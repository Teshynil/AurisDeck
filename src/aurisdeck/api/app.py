"""FastAPI application factory, lifespan wiring, and core REST endpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from aurisdeck import __version__
from aurisdeck.api.deps import (
    SESSION_COOKIE_NAME,
    ApiError,
    get_auth_service,
    get_coordinator,
    get_current_user,
    get_locale_loader,
    get_paths,
    get_settings,
    get_user_repo,
    is_https_request,
    register_exception_handlers,
    require_admin,
    resolve_current_user,
)
from aurisdeck.auth.service import AuthError, AuthService
from aurisdeck.config import Settings
from aurisdeck.db.migrations import run_migrations
from aurisdeck.db.repositories import JobRepo, SegmentRepo, UserRepo
from aurisdeck.domain.models import SlugStr
from aurisdeck.i18n.loader import LocaleLoader
from aurisdeck.logging_setup import configure_main_logging, set_log_level
from aurisdeck.storage.defaults import seed_default_profiles
from aurisdeck.storage.paths import DataPaths, atomic_write_text
from aurisdeck.storage.stores import ProfileStore
from aurisdeck.workers.coordinator import Coordinator
from aurisdeck.workers.devices import DeviceInfo
from aurisdeck.workers.events import EventBus

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_BUILTIN_LOCALES_DIR = _WEB_DIR / "locales"
_STATIC_DIR = _WEB_DIR / "static"
_SYSTEM_SETTINGS_FILENAME = "system_settings.json"


class LoginRequest(BaseModel):
    """Shared-password login request body."""

    model_config = ConfigDict(extra="forbid")

    password: str = ""


class PlayerPreferencesPatch(BaseModel):
    """Partial update payload for user media player preferences."""

    model_config = ConfigDict(extra="forbid")

    playback_rate: float | None = Field(default=None, ge=0.25, le=4.0)
    auto_scroll: bool | None = None


class MePreferencesPatch(BaseModel):
    """Partial update payload for Account Settings (`users.preferences_json`)."""

    model_config = ConfigDict(extra="forbid")

    locale: SlugStr | None = None
    default_profile_id: SlugStr | None = None
    player: PlayerPreferencesPatch | None = None
    preferences: MePreferencesPatch | None = None


class AdminSettingsPatch(BaseModel):
    """Partial update payload for global system/admin runtime settings."""

    model_config = ConfigDict(extra="forbid")

    model_idle_unload_seconds: int | None = Field(default=None, ge=0)
    cpu_threads: int | None = Field(default=None, ge=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None


def _system_settings_path(paths: DataPaths) -> Path:
    return paths.root / _SYSTEM_SETTINGS_FILENAME


def _current_admin_settings(settings: Settings) -> dict[str, Any]:
    return {
        "model_idle_unload_seconds": int(settings.model_idle_unload_seconds),
        "cpu_threads": int(settings.cpu_threads),
        "log_level": str(settings.log_level),
    }


def _load_persisted_system_settings(paths: DataPaths, settings: Settings) -> dict[str, Any]:
    path = _system_settings_path(paths)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            parsed = AdminSettingsPatch.model_validate(raw)
            if parsed.model_idle_unload_seconds is not None:
                settings.model_idle_unload_seconds = parsed.model_idle_unload_seconds
            if parsed.cpu_threads is not None:
                settings.cpu_threads = parsed.cpu_threads
            if parsed.log_level is not None:
                settings.log_level = parsed.log_level
    return _current_admin_settings(settings)


def _save_persisted_system_settings(paths: DataPaths, settings: Settings) -> dict[str, Any]:
    payload = _current_admin_settings(settings)
    atomic_write_text(
        _system_settings_path(paths),
        json.dumps(payload, indent=2) + "\n",
        paths.tmp,
    )
    return payload


def _apply_preferences_patch(
    current: dict[str, Any],
    patch: MePreferencesPatch,
) -> dict[str, Any]:
    updated = dict(current)
    nested = patch.preferences
    if nested is not None:
        updated = _apply_preferences_patch(updated, nested)

    if patch.locale is not None:
        updated["locale"] = patch.locale
    if patch.default_profile_id is not None:
        updated["default_profile_id"] = patch.default_profile_id
    if patch.player is not None:
        existing_player = updated.get("player")
        player_prefs = dict(existing_player) if isinstance(existing_player, dict) else {}
        if patch.player.playback_rate is not None:
            player_prefs["playback_rate"] = float(patch.player.playback_rate)
        if patch.player.auto_scroll is not None:
            player_prefs["auto_scroll"] = bool(patch.player.auto_scroll)
        updated["player"] = player_prefs
    return updated


def _build_core_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/auth/status")
    async def auth_status(request: Request) -> dict[str, Any]:
        settings = get_settings(request)
        authenticated = False
        try:
            resolve_current_user(request)
            authenticated = True
        except AuthError:
            authenticated = False
        return {
            "auth_mode": settings.auth_mode,
            "authenticated": authenticated,
        }

    @router.post("/auth/login")
    async def auth_login(
        body: LoginRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        auth_service = get_auth_service(request)
        user_repo = get_user_repo(request)
        if not auth_service.verify_shared_password(body.password):
            raise ApiError(
                status_code=401,
                error_code="AUTH_INVALID_CREDENTIALS",
                detail="Invalid password",
            )

        default_user = user_repo.get_by_username("default")
        user_id = default_user.id if default_user is not None else "default"
        cookie_value = auth_service.create_session_cookie(user_id=user_id)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=cookie_value,
            httponly=True,
            samesite="lax",
            secure=is_https_request(request),
            path="/",
        )
        return {"authenticated": True}

    @router.post("/auth/logout")
    async def auth_logout(request: Request, response: Response) -> dict[str, Any]:
        response.delete_cookie(
            key=SESSION_COOKIE_NAME,
            httponly=True,
            samesite="lax",
            secure=is_https_request(request),
            path="/",
        )
        return {"authenticated": False}

    @router.get("/locales")
    async def list_locales(
        locale_loader: Annotated[LocaleLoader, Depends(get_locale_loader)],
    ) -> JSONResponse:
        return JSONResponse(
            content={"locales": locale_loader.list_locales()},
            headers={"Cache-Control": "public, max-age=60"},
        )

    @router.get("/locales/{code}.json")
    async def get_locale_file(
        code: str,
        locale_loader: Annotated[LocaleLoader, Depends(get_locale_loader)],
    ) -> JSONResponse:
        try:
            data = locale_loader.get_locale(code)
        except ValueError as exc:
            raise ApiError(
                status_code=400,
                error_code="VALIDATION_ERROR",
                error_params={"code": code},
                detail=str(exc),
            ) from exc
        except FileNotFoundError as exc:
            raise ApiError(
                status_code=404,
                error_code="LOCALE_NOT_FOUND",
                error_params={"code": code},
                detail=f"Locale {code!r} not found",
            ) from exc

        return JSONResponse(
            content=data,
            headers={"Cache-Control": "public, max-age=60"},
        )

    @router.get("/me")
    async def get_me(
        user: Annotated[dict[str, Any], Depends(get_current_user)],
    ) -> dict[str, Any]:
        return user

    @router.patch("/me")
    async def patch_me(
        body: MePreferencesPatch,
        user: Annotated[dict[str, Any], Depends(get_current_user)],
        user_repo: Annotated[UserRepo, Depends(get_user_repo)],
    ) -> dict[str, Any]:
        next_preferences = _apply_preferences_patch(user.get("preferences", {}), body)
        updated = user_repo.update_preferences(user["id"], next_preferences)
        if updated is None:
            raise ApiError(
                status_code=404,
                error_code="AUTH_USER_NOT_FOUND",
                error_params={"username": user.get("username", "")},
                detail="User not found",
            )
        return {
            "id": updated.id,
            "username": updated.username,
            "display_name": updated.display_name,
            "role": updated.role,
            "preferences": dict(updated.preferences),
            "created_at": updated.created_at,
            "updated_at": updated.updated_at,
        }

    @router.get("/admin/settings")
    async def get_admin_settings(
        request: Request,
        _admin: Annotated[dict[str, Any], Depends(require_admin)],
    ) -> dict[str, Any]:
        settings = get_settings(request)
        payload = _current_admin_settings(settings)
        request.app.state.admin_settings = payload
        return payload

    @router.patch("/admin/settings")
    async def patch_admin_settings(
        body: AdminSettingsPatch,
        request: Request,
        _admin: Annotated[dict[str, Any], Depends(require_admin)],
    ) -> dict[str, Any]:
        settings = get_settings(request)
        paths = get_paths(request)
        if body.model_idle_unload_seconds is not None:
            settings.model_idle_unload_seconds = body.model_idle_unload_seconds
        if body.cpu_threads is not None:
            settings.cpu_threads = body.cpu_threads
        if body.log_level is not None:
            settings.log_level = body.log_level
            set_log_level(settings.log_level)
        payload = _save_persisted_system_settings(paths, settings)
        request.app.state.admin_settings = payload
        return payload

    @router.get("/system/devices")
    async def get_system_devices(
        _admin: Annotated[dict[str, Any], Depends(require_admin)],
        coordinator: Annotated[Coordinator, Depends(get_coordinator)],
    ) -> dict[str, list[dict[str, Any]]]:
        devices = [asdict(device) for device in coordinator.list_devices()]
        return {"devices": devices}

    @router.post("/system/unload-models")
    async def unload_system_models(
        _admin: Annotated[dict[str, Any], Depends(require_admin)],
        coordinator: Annotated[Coordinator, Depends(get_coordinator)],
    ) -> dict[str, Any]:
        coordinator.unload_all_models()
        return {"status": "ok", "unloaded": True}

    return router


def create_app(
    settings: Settings | None = None,
    *,
    coordinator: Coordinator | None = None,
    devices: list[DeviceInfo] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    resolved_settings = settings or Settings()
    paths = DataPaths(resolved_settings.data_dir)
    user_repo = UserRepo(paths.db_path)
    job_repo = JobRepo(paths.db_path)
    segment_repo = SegmentRepo(paths.db_path)
    event_bus = EventBus()
    locale_loader = LocaleLoader(_BUILTIN_LOCALES_DIR, paths.locales)
    auth_service = AuthService(user_repo=user_repo, settings=resolved_settings)
    resolved_coordinator = coordinator or Coordinator(
        settings=resolved_settings,
        paths=paths,
        job_repo=job_repo,
        segment_repo=segment_repo,
        event_bus=event_bus,
        devices=devices,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        paths.ensure_dirs()
        app.state.admin_settings = _load_persisted_system_settings(paths, resolved_settings)
        configure_main_logging(
            paths,
            level=resolved_settings.log_level,
            max_bytes=resolved_settings.log_max_bytes,
            backup_count=resolved_settings.log_backup_count,
        )
        run_migrations(paths.db_path)
        seed_default_profiles(ProfileStore(paths))
        app.state.coordinator.start()
        try:
            yield
        finally:
            app.state.coordinator.stop()

    app = FastAPI(
        title="AurisDeck",
        version=__version__,
        lifespan=lifespan,
    )
    # Assign root_path after instantiation to avoid FastAPIDeprecationWarning;
    # runtime ASGI scope root_path is configured via uvicorn.run(root_path=...).
    app.root_path = resolved_settings.base_path

    app.state.settings = resolved_settings
    app.state.paths = paths
    app.state.user_repo = user_repo
    app.state.job_repo = job_repo
    app.state.segment_repo = segment_repo
    app.state.auth_service = auth_service
    app.state.event_bus = event_bus
    app.state.coordinator = resolved_coordinator
    app.state.locale_loader = locale_loader
    app.state.admin_settings = _current_admin_settings(resolved_settings)

    register_exception_handlers(app)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(_build_core_router())

    from aurisdeck.api.routes_config import router as config_router
    from aurisdeck.api.routes_events import router as events_router
    from aurisdeck.api.routes_jobs import router as jobs_router

    app.include_router(config_router)
    app.include_router(events_router)
    app.include_router(jobs_router)

    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

    return app
