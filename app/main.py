from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.sanitize import SanitizeRequestBodyMiddleware
from app.middleware.tracing import RequestTracingMiddleware
from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.ingestion import router as ingestion_router
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Middleware ---
    # FastAPI applies middleware in reverse registration order:
    # last added = outermost = runs first on every request.
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(SanitizeRequestBodyMiddleware)

    # CORS must be outermost so browser preflight OPTIONS requests are
    # handled before any other middleware or route logic runs.
    cors_origins = [
        origin.strip()
        for origin in settings.CORS_ALLOWED_ORIGINS.split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,       # driven by env var
        allow_methods=["GET", "POST"],    # only what this API exposes
        allow_headers=["Content-Type"],   # only header the frontend sends
        allow_credentials=False,          # no cookie/session auth in use
        max_age=600,                      # browser caches preflight for 10 min
    )

    # --- Global exception handler ---
    # Catches any unhandled exception that escapes route handlers,
    # returns a sanitised 500 so stack traces never reach the client.
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.error(
            "Unhandled exception | request_id=%s | %s: %s",
            request_id,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    app.add_exception_handler(Exception, unhandled_exception_handler)

    # --- Routers ---
    # Health probes — no auth (must be reachable by load balancers / k8s)
    app.include_router(
        health_router,
        prefix="/health",
        tags=["Health"],
    )

    # Ingestion routes
    app.include_router(
        ingestion_router,
        prefix="/api/v1/ingestion",
        tags=["Ingestion"],
    )

    # Chat / RAG routes
    app.include_router(
        chat_router,
        prefix="/api/v1/chat",
        tags=["Chat"],
    )

    return app


app = create_app()
