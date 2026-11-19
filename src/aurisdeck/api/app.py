"""FastAPI application factory, lifespan wiring, and core REST endpoints."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response
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
    get_system_settings_repo,
    get_user_repo,
    is_https_request,
    register_exception_handlers,
    require_admin,
    resolve_current_user,
)
from aurisdeck.auth.service import AuthError, AuthService, hash_secret
from aurisdeck.backends.hf_verify import verify_hf_token
from aurisdeck.config import Settings
from aurisdeck.db.migrations import run_migrations
from aurisdeck.db.repositories import JobRepo, SegmentRepo, SystemSettingsRepo, UserRepo
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
    """Login request body supporting username+password or legacy shared password."""

    model_config = ConfigDict(extra="forbid")

    username: str | None = None
    password: str = ""


class SetupVerifyTokenRequest(BaseModel):
    """Payload for verifying a Hugging Face token."""

    model_config = ConfigDict(extra="forbid")

    token: str | None = None


class SetupCompleteRequest(BaseModel):
    """First-run setup wizard submission payload."""

    model_config = ConfigDict(extra="forbid")

    locale: SlugStr = "es"
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=6, max_length=256)
    hf_token: str | None = None


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
    hf_token: str | None = None


class ModelPullRequest(BaseModel):
    """Payload to download or update a cached Whisper or Hugging Face model."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=256)


def _system_settings_path(paths: DataPaths) -> Path:
    return paths.root / _SYSTEM_SETTINGS_FILENAME


def _mask_token(token: str | None) -> str | None:
    if not token:
        return None
    cleaned = token.strip()
    if not cleaned:
        return None
    if len(cleaned) <= 7:
        return "••••"
    prefix = "hf_" if cleaned.startswith("hf_") else ""
    return f"{prefix}••••{cleaned[-4:]}"


def _current_admin_settings(
    settings: Settings,
    system_settings_repo: SystemSettingsRepo | None = None,
) -> dict[str, Any]:
    token: str | None = None
    if system_settings_repo is not None:
        try:
            token = system_settings_repo.get_hf_token()
        except sqlite3.OperationalError:
            token = None
    return {
        "model_idle_unload_seconds": int(settings.model_idle_unload_seconds),
        "cpu_threads": int(settings.cpu_threads),
        "log_level": str(settings.log_level),
        "hf_token_configured": bool(token),
        "hf_token_masked": _mask_token(token),
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


def _save_persisted_system_settings(
    paths: DataPaths,
    settings: Settings,
    system_settings_repo: SystemSettingsRepo | None = None,
) -> dict[str, Any]:
    file_payload = {
        "model_idle_unload_seconds": int(settings.model_idle_unload_seconds),
        "cpu_threads": int(settings.cpu_threads),
        "log_level": str(settings.log_level),
    }
    atomic_write_text(
        _system_settings_path(paths),
        json.dumps(file_payload, indent=2) + "\n",
        paths.tmp,
    )
    return _current_admin_settings(settings, system_settings_repo)


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
        auth_service = get_auth_service(request)
        system_settings_repo = get_system_settings_repo(request)
        authenticated = False
        try:
            resolve_current_user(request)
            authenticated = True
        except AuthError:
            authenticated = False
        try:
            setup_required = not system_settings_repo.is_setup_completed()
        except sqlite3.OperationalError:
            setup_required = True
        return {
            "auth_mode": auth_service.effective_auth_mode,
            "authenticated": authenticated,
            "setup_required": setup_required,
        }

    @router.post("/auth/login")
    async def auth_login(
        body: LoginRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        auth_service = get_auth_service(request)
        user = auth_service.verify_credentials(body.username, body.password)
        if user is None:
            raise ApiError(
                status_code=401,
                error_code="AUTH_INVALID_CREDENTIALS",
                detail="Invalid username or password",
            )

        cookie_value = auth_service.create_session_cookie(user_id=user.id)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=cookie_value,
            httponly=True,
            samesite="lax",
            secure=is_https_request(request),
            path="/",
        )
        return {"authenticated": True}

    @router.get("/setup/status")
    async def setup_status(
        request: Request,
        locale_loader: Annotated[LocaleLoader, Depends(get_locale_loader)],
    ) -> dict[str, Any]:
        system_settings_repo = get_system_settings_repo(request)
        try:
            setup_required = not system_settings_repo.is_setup_completed()
        except sqlite3.OperationalError:
            setup_required = True
        return {
            "setup_required": setup_required,
            "locales": locale_loader.list_locales(),
        }

    @router.post("/setup/verify-hf-token")
    async def setup_verify_hf_token(
        body: SetupVerifyTokenRequest,
        request: Request,
    ) -> dict[str, Any]:
        system_settings_repo = get_system_settings_repo(request)
        if system_settings_repo.is_setup_completed():
            raise ApiError(
                status_code=409,
                error_code="SETUP_ALREADY_COMPLETED",
                detail="Initial setup has already been completed",
            )
        settings = get_settings(request)
        result = verify_hf_token(body.token, hf_endpoint=settings.hf_endpoint)
        return {
            "valid": result.valid,
            "status": result.status,
            "missing_models": result.missing_models,
            "username": result.username,
            "detail": result.detail,
        }

    @router.post("/setup/complete")
    async def setup_complete(
        body: SetupCompleteRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        system_settings_repo = get_system_settings_repo(request)
        if system_settings_repo.is_setup_completed():
            raise ApiError(
                status_code=409,
                error_code="SETUP_ALREADY_COMPLETED",
                detail="Initial setup has already been completed",
            )

        cleaned_username = body.username.strip()
        if not cleaned_username or any(ch.isspace() for ch in cleaned_username):
            raise ApiError(
                status_code=422,
                error_code="VALIDATION_ERROR",
                error_params={"field": "username"},
                detail="Username must not be empty or contain spaces",
            )

        user_repo = get_user_repo(request)
        auth_service = get_auth_service(request)
        admin_user = user_repo.setup_initial_admin(
            cleaned_username,
            hash_secret(body.password),
            locale=body.locale,
        )
        if body.hf_token is not None and body.hf_token.strip():
            system_settings_repo.set_hf_token(body.hf_token.strip())
        else:
            system_settings_repo.set_hf_token(None)
        system_settings_repo.set_auth_mode("password")
        system_settings_repo.set_setup_completed(True)

        settings = get_settings(request)
        request.app.state.admin_settings = _current_admin_settings(settings, system_settings_repo)

        cookie_value = auth_service.create_session_cookie(user_id=admin_user.id)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=cookie_value,
            httponly=True,
            samesite="lax",
            secure=is_https_request(request),
            path="/",
        )
        return {
            "setup_completed": True,
            "authenticated": True,
            "user": {
                "id": admin_user.id,
                "username": admin_user.username,
                "display_name": admin_user.display_name,
                "role": admin_user.role,
                "preferences": dict(admin_user.preferences),
            },
        }

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
        system_settings_repo = get_system_settings_repo(request)
        payload = _current_admin_settings(settings, system_settings_repo)
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
        system_settings_repo = get_system_settings_repo(request)
        if body.model_idle_unload_seconds is not None:
            settings.model_idle_unload_seconds = body.model_idle_unload_seconds
        if body.cpu_threads is not None:
            settings.cpu_threads = body.cpu_threads
        if body.log_level is not None:
            settings.log_level = body.log_level
            set_log_level(settings.log_level)
        if "hf_token" in body.model_fields_set:
            system_settings_repo.set_hf_token(body.hf_token)
        payload = _save_persisted_system_settings(paths, settings, system_settings_repo)
        request.app.state.admin_settings = payload
        return payload

    @router.post("/admin/settings/verify-hf-token")
    async def admin_verify_hf_token(
        body: SetupVerifyTokenRequest,
        request: Request,
        _admin: Annotated[dict[str, Any], Depends(require_admin)],
    ) -> dict[str, Any]:
        settings = get_settings(request)
        system_settings_repo = get_system_settings_repo(request)
        token_to_check = body.token if body.token is not None else system_settings_repo.get_hf_token()
        result = verify_hf_token(token_to_check, hf_endpoint=settings.hf_endpoint)
        return {
            "valid": result.valid,
            "status": result.status,
            "missing_models": result.missing_models,
            "username": result.username,
            "detail": result.detail,
        }

    @router.get("/system/devices")
    async def get_system_devices(
        _admin: Annotated[dict[str, Any], Depends(require_admin)],
        coordinator: Annotated[Coordinator, Depends(get_coordinator)],
    ) -> dict[str, list[dict[str, Any]]]:
        if hasattr(coordinator, "get_device_snapshots"):
            devices = coordinator.get_device_snapshots()
        else:
            devices = [asdict(device) for device in coordinator.list_devices()]
        return {"devices": devices}

    @router.post("/system/unload-models")
    async def unload_system_models(
        _admin: Annotated[dict[str, Any], Depends(require_admin)],
        coordinator: Annotated[Coordinator, Depends(get_coordinator)],
    ) -> dict[str, Any]:
        coordinator.unload_all_models()
        return {"status": "ok", "unloaded": True}

    @router.get("/system/models")
    async def list_system_models(
        _admin: Annotated[dict[str, Any], Depends(require_admin)],
        paths: Annotated[DataPaths, Depends(get_paths)],
    ) -> dict[str, Any]:
        from aurisdeck.storage.models_cache import list_cached_models

        return await anyio.to_thread.run_sync(list_cached_models, paths.models)

    @router.post("/system/models/pull")
    async def pull_system_model(
        body: ModelPullRequest,
        _admin: Annotated[dict[str, Any], Depends(require_admin)],
        paths: Annotated[DataPaths, Depends(get_paths)],
        settings: Annotated[Settings, Depends(get_settings)],
        system_settings_repo: Annotated[SystemSettingsRepo, Depends(get_system_settings_repo)],
    ) -> dict[str, Any]:
        from aurisdeck.storage.models_cache import pull_or_update_model

        hf_token = system_settings_repo.get_hf_token()
        try:
            return await anyio.to_thread.run_sync(
                lambda: pull_or_update_model(
                    body.model,
                    paths.models,
                    hf_token=hf_token,
                    hf_endpoint=settings.hf_endpoint,
                )
            )
        except ValueError as exc:
            raise ApiError(
                status_code=400,
                error_code="VALIDATION_ERROR",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise ApiError(
                status_code=502,
                error_code="MODEL_LOAD_FAILED",
                error_params={"model": body.model},
                detail=str(exc),
            ) from exc

    @router.delete("/system/models")
    async def delete_system_model(
        repo_id: Annotated[str, Query(min_length=1, max_length=256)],
        _admin: Annotated[dict[str, Any], Depends(require_admin)],
        paths: Annotated[DataPaths, Depends(get_paths)],
        coordinator: Annotated[Coordinator, Depends(get_coordinator)],
    ) -> dict[str, Any]:
        from aurisdeck.storage.models_cache import delete_cached_model

        if hasattr(coordinator, "get_device_snapshots"):
            busy = any(dev.get("state") == "busy" for dev in coordinator.get_device_snapshots())
            if busy:
                raise ApiError(
                    status_code=409,
                    error_code="MODEL_IN_USE",
                    detail="Cannot delete cached models while a transcription job is running",
                )

        coordinator.unload_all_models()
        try:
            return await anyio.to_thread.run_sync(delete_cached_model, repo_id, paths.models)
        except FileNotFoundError as exc:
            raise ApiError(
                status_code=404,
                error_code="MODEL_NOT_FOUND",
                error_params={"repo_id": repo_id},
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise ApiError(
                status_code=400,
                error_code="VALIDATION_ERROR",
                detail=str(exc),
            ) from exc

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
    system_settings_repo = SystemSettingsRepo(paths.db_path)
    job_repo = JobRepo(paths.db_path)
    segment_repo = SegmentRepo(paths.db_path)
    event_bus = EventBus()
    locale_loader = LocaleLoader(_BUILTIN_LOCALES_DIR, paths.locales)
    auth_service = AuthService(
        user_repo=user_repo,
        settings=resolved_settings,
        system_settings_repo=system_settings_repo,
    )
    resolved_coordinator = coordinator or Coordinator(
        settings=resolved_settings,
        paths=paths,
        job_repo=job_repo,
        segment_repo=segment_repo,
        event_bus=event_bus,
        devices=devices,
        hf_token_provider=system_settings_repo.get_hf_token,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        paths.ensure_dirs()
        configure_main_logging(
            paths,
            level=resolved_settings.log_level,
            max_bytes=resolved_settings.log_max_bytes,
            backup_count=resolved_settings.log_backup_count,
        )
        run_migrations(paths.db_path)
        app.state.admin_settings = _load_persisted_system_settings(paths, resolved_settings)
        app.state.admin_settings = _current_admin_settings(resolved_settings, system_settings_repo)
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
    app.state.system_settings_repo = system_settings_repo
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
