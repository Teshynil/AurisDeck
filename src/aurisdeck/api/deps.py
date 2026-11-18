"""FastAPI state dependencies, authentication helpers, and standard JSON error handlers."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from aurisdeck.auth.service import AuthError, AuthService
from aurisdeck.backends.errors import AurisDeckBackendError
from aurisdeck.config import Settings
from aurisdeck.db.repositories import JobRepo, SegmentRepo, SystemSettingsRepo, UserRepo
from aurisdeck.i18n.loader import LocaleLoader
from aurisdeck.storage.paths import DataPaths
from aurisdeck.workers.coordinator import Coordinator
from aurisdeck.workers.events import EventBus

SESSION_COOKIE_NAME = "aurisdeck_session"
logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Structured API exception serialized as error_code + error_params JSON."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        *,
        error_params: dict[str, Any] | None = None,
        detail: str = "",
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.error_params: dict[str, Any] = dict(error_params) if error_params else {}
        self.detail = detail or error_code
        super().__init__(self.detail)


def error_payload(
    error_code: str,
    error_params: dict[str, Any] | None = None,
    detail: str = "",
) -> dict[str, Any]:
    """Build the canonical AurisDeck error payload without translated prose."""
    return {
        "error_code": error_code,
        "error_params": dict(error_params) if error_params else {},
        "detail": detail,
    }


def is_https_request(request: Request) -> bool:
    """Detect whether the client connection is HTTPS directly or via reverse proxy."""
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    first_proto = forwarded_proto.split(",")[0].strip().lower() if forwarded_proto else ""
    return request.url.scheme == "https" or first_proto == "https"


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_paths(request: Request) -> DataPaths:
    return request.app.state.paths


def get_user_repo(request: Request) -> UserRepo:
    return request.app.state.user_repo


def get_system_settings_repo(request: Request) -> SystemSettingsRepo:
    return request.app.state.system_settings_repo


def get_job_repo(request: Request) -> JobRepo:
    return request.app.state.job_repo


def get_segment_repo(request: Request) -> SegmentRepo:
    return request.app.state.segment_repo


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


def get_event_bus(request: Request) -> EventBus:
    return request.app.state.event_bus


def get_coordinator(request: Request) -> Coordinator:
    return request.app.state.coordinator


def get_locale_loader(request: Request) -> LocaleLoader:
    return request.app.state.locale_loader


def extract_bearer_token(request: Request) -> str | None:
    """Extract a Bearer token from the Authorization header if present."""
    header = request.headers.get("authorization")
    if header is None:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def resolve_current_user(request: Request) -> dict[str, Any]:
    """Resolve the current user using session cookie, Bearer token, or auth_mode='none'."""
    auth_service = get_auth_service(request)
    cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
    bearer_token = extract_bearer_token(request)
    return auth_service.resolve_user(cookie_token=cookie_token, bearer_token=bearer_token)


def get_current_user(request: Request) -> dict[str, Any]:
    """FastAPI dependency returning the authenticated user dict."""
    return resolve_current_user(request)


def require_admin(
    user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    """FastAPI dependency enforcing admin role on system endpoints."""
    if user.get("role") != "admin":
        raise ApiError(
            status_code=403,
            error_code="FORBIDDEN",
            detail="Admin role is required",
        )
    return user


def register_exception_handlers(app: FastAPI) -> None:
    """Register standard JSON error handlers for Auth, Backend, HTTP, and validation errors."""

    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error(
                "API %s %s -> %s (%s): %s",
                request.method,
                request.url.path,
                exc.status_code,
                exc.error_code,
                exc.detail,
            )
        else:
            logger.warning(
                "API %s %s -> %s (%s): %s",
                request.method,
                request.url.path,
                exc.status_code,
                exc.error_code,
                exc.detail,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(exc.error_code, exc.error_params, exc.detail),
        )

    @app.exception_handler(AuthError)
    async def _handle_auth_error(request: Request, exc: AuthError) -> JSONResponse:
        logger.warning("Auth error on %s %s (%s)", request.method, request.url.path, exc.error_code)
        return JSONResponse(
            status_code=401,
            content=error_payload(exc.error_code, exc.error_params, str(exc)),
        )

    @app.exception_handler(AurisDeckBackendError)
    async def _handle_backend_error(
        request: Request, exc: AurisDeckBackendError
    ) -> JSONResponse:
        logger.exception(
            "Backend error on %s %s (%s): %s",
            request.method,
            request.url.path,
            exc.error_code,
            exc.detail or str(exc),
        )
        return JSONResponse(
            status_code=500,
            content=error_payload(exc.error_code, exc.error_params, exc.detail or str(exc)),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_payload(
                "VALIDATION_ERROR",
                {"errors": jsonable_encoder(exc.errors())},
                "Request validation failed",
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error_code" in exc.detail:
            payload = error_payload(
                str(exc.detail["error_code"]),
                exc.detail.get("error_params"),
                str(exc.detail.get("detail", "")),
            )
        else:
            code = f"HTTP_{exc.status_code}"
            payload = error_payload(code, {}, str(exc.detail or code))
        return JSONResponse(status_code=exc.status_code, content=payload)
