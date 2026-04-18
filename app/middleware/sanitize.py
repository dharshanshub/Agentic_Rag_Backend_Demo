import json

from json_repair import repair_json  # type: ignore[import-untyped]
from starlette.types import ASGIApp, Receive, Scope, Send

from app.utils.logger import get_logger

logger = get_logger(__name__)


class SanitizeRequestBodyMiddleware:
    """
    Fixes malformed JSON request bodies (e.g. unescaped inner quotes)
    before FastAPI parses them, using json-repair.
    Only applies to requests with Content-Type: application/json.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_type = headers.get(b"content-type", b"").decode()

        if "application/json" not in content_type:
            await self.app(scope, receive, send)
            return

        # Read the raw body
        message = await receive()
        raw: bytes = message.get("body", b"")

        # Nothing to sanitise — pass through immediately
        if not raw.strip():
            await self.app(scope, receive, send)
            return

        # Attempt repair if JSON is invalid
        final_body = raw
        try:
            decoded = raw.decode("utf-8")
            json.loads(decoded)                             # valid — use as-is
        except (json.JSONDecodeError, UnicodeDecodeError):
            repaired = str(repair_json(decoded, return_objects=False))
            final_body = repaired.encode("utf-8") if repaired else raw
            logger.debug("Repaired malformed JSON body for %s", scope.get("path"))

        # Inject the fixed body back into the ASGI receive stream
        async def patched_receive() -> dict:
            return {"type": "http.request", "body": final_body, "more_body": False}

        await self.app(scope, patched_receive, send)
