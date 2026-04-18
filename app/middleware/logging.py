import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Paths excluded from request logging to reduce noise in production.
# Health probes are hit constantly by load balancers / k8s — no signal value.
_SKIP_PATHS: frozenset[str] = frozenset({"/health/live", "/health/ready"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request with method, path, status code, latency,
    and the correlation ID injected by RequestTracingMiddleware.

    Log format (INFO):
        POST /api/v1/ingestion/process → 200 | 1243ms | request_id=abc123

    Unhandled exceptions propagate normally after being noted; the
    global exception handler in main.py is responsible for returning
    a sanitised 500 response to the client.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        # request_id is set by RequestTracingMiddleware which runs before this
        request_id: str = getattr(request.state, "request_id", "-")

        try:
            response: Response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "%s %s → UNHANDLED EXCEPTION | %.1fms | request_id=%s",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        log_fn = logger.warning if response.status_code >= 400 else logger.info
        log_fn(
            "%s %s → %d | %.1fms | request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response
