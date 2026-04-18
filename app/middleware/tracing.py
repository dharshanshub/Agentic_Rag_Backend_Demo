import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Injects a correlation ID into every request/response cycle.

    Behaviour:
    - If the incoming request already carries an ``X-Request-ID`` header
      (set by an upstream gateway or client), that value is reused so the
      trace can be correlated end-to-end.
    - Otherwise a new UUID4 is generated for this request.
    - The ID is stored on ``request.state.request_id`` so any downstream
      handler or middleware can read it without re-parsing headers.
    - The ID is echoed back in the ``X-Request-ID`` response header so
      callers can correlate their own logs with server-side logs.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
