"""Core API, i18n LocaleLoader, auth cookie, /me, and /admin tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aurisdeck import __version__
from aurisdeck.api import create_app
from aurisdeck.api.deps import ApiError, require_admin
from aurisdeck.config import Settings
from aurisdeck.i18n import LocaleLoader
from aurisdeck.storage.paths import DataPaths
from aurisdeck.storage.stores import ProfileStore
from aurisdeck.workers.devices import DeviceInfo

BUILTIN_LOCALES_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "aurisdeck" / "web" / "locales"
)

REQUIRED_STAGES = {
    "queued",
    "preprocessing",
    "downloading_model",
    "loading_model",
    "transcribing",
    "diarizing",
    "merging",
    "exporting",
    "completed",
    "completed_partial",
    "cancelling",
    "cancelled",
    "failed",
}

REQUIRED_ERRORS = {
    "AUTH_REQUIRED",
    "AUTH_INVALID_TOKEN",
    "AUTH_INVALID_CREDENTIALS",
    "AUTH_PASSWORD_NOT_CONFIGURED",
    "AUTH_USER_NOT_FOUND",
    "MEDIA_INVALID",
    "DIARIZATION_HF_TOKEN_MISSING",
    "DIARIZATION_GATED_TERMS",
    "DIARIZATION_DEPENDENCY_MISSING",
    "DIARIZATION_INTERRUPTED",
    "GPU_OOM",
    "MODEL_LOAD_FAILED",
    "SERVER_INTERRUPTED",
    "JOB_NOT_FOUND",
    "PROFILE_NOT_FOUND",
    "INVALID_ARCHIVE",
}


class FakeCoordinator:
    """Lightweight coordinator stub avoiding subprocess spawns in API unit tests."""

    def __init__(self, devices: list[DeviceInfo] | None = None) -> None:
        self.started = False
        self.stopped = False
        self.unload_calls = 0
        self._devices = devices or [
            DeviceInfo(
                id="cuda:0",
                kind="cuda",
                index=0,
                name="Fake RTX GPU",
                vram_total_mb=16384,
                vram_free_mb=15000,
            ),
            DeviceInfo(id="cpu", kind="cpu", index=None, name="CPU (8 threads)"),
        ]

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def list_devices(self) -> list[DeviceInfo]:
        return list(self._devices)

    def unload_all_models(self) -> None:
        self.unload_calls += 1


def _collect_leaf_paths(obj: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                paths.update(_collect_leaf_paths(value, next_prefix))
            else:
                paths.add(next_prefix)
    return paths


def test_builtin_locales_have_exact_recursive_key_parity() -> None:
    es_data = json.loads((BUILTIN_LOCALES_DIR / "es.json").read_text(encoding="utf-8"))
    en_data = json.loads((BUILTIN_LOCALES_DIR / "en.json").read_text(encoding="utf-8"))

    assert es_data["_meta"] == {"code": "es", "name": "Español", "direction": "ltr"}
    assert en_data["_meta"] == {"code": "en", "name": "English", "direction": "ltr"}

    es_keys = _collect_leaf_paths(es_data)
    en_keys = _collect_leaf_paths(en_data)
    assert es_keys == en_keys

    assert REQUIRED_STAGES.issubset(set(es_data["stages"].keys()))
    assert REQUIRED_ERRORS.issubset(set(es_data["errors"].keys()))
    assert "account" in es_data["settings"]
    assert "system" in es_data["settings"]


def test_locale_loader_merges_custom_locales_and_blocks_traversal(tmp_path: Path) -> None:
    custom_dir = tmp_path / "locales"
    custom_dir.mkdir(parents=True, exist_ok=True)

    (custom_dir / "es.json").write_text(
        json.dumps(
            {
                "_meta": {"name": "Español (Custom)"},
                "nav": {"simple": "Transcripción rápida"},
                "custom_section": {"greeting": "Hola"},
            }
        ),
        encoding="utf-8",
    )
    (custom_dir / "fr.json").write_text(
        json.dumps(
            {
                "_meta": {"code": "fr", "name": "Français", "direction": "ltr"},
                "nav": {"simple": "Transcrire"},
            }
        ),
        encoding="utf-8",
    )

    loader = LocaleLoader(BUILTIN_LOCALES_DIR, custom_dir)
    listed = {item["code"]: item for item in loader.list_locales()}

    assert set(listed.keys()) == {"en", "es", "fr"}
    assert listed["es"]["name"] == "Español (Custom)"
    assert listed["es"]["direction"] == "ltr"
    assert listed["fr"] == {"code": "fr", "name": "Français", "direction": "ltr"}

    merged_es = loader.get_locale("es")
    assert merged_es["nav"]["simple"] == "Transcripción rápida"
    assert merged_es["nav"]["library"] == "Biblioteca"
    assert merged_es["custom_section"]["greeting"] == "Hola"

    fr = loader.get_locale("fr")
    assert fr["nav"]["simple"] == "Transcrire"

    with pytest.raises(FileNotFoundError):
        loader.get_locale("de")

    for bad_code in ("../es", "..", "es/en", "es.json", ""):
        with pytest.raises(ValueError):
            loader.get_locale(bad_code)


def test_healthz_locales_and_me_in_none_auth_mode(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.ensure_dirs()
    (paths.locales / "fr.json").write_text(
        json.dumps(
            {
                "_meta": {"code": "fr", "name": "Français", "direction": "ltr"},
                "nav": {"simple": "Transcrire"},
            }
        ),
        encoding="utf-8",
    )

    fake_coord = FakeCoordinator()
    settings = Settings(data_dir=tmp_path, auth_mode="none", base_path="/auris")
    app = create_app(settings, coordinator=fake_coord)  # type: ignore[arg-type]

    assert app.root_path == "/auris"

    with TestClient(app) as client:
        assert fake_coord.started is True

        # Default profiles seeded on startup
        assert not ProfileStore(paths).is_empty()

        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "version": __version__}

        status_resp = client.get("/api/v1/auth/status")
        assert status_resp.status_code == 200
        assert status_resp.json() == {
            "auth_mode": "none",
            "authenticated": True,
            "setup_required": True,
        }

        locales_resp = client.get("/api/v1/locales")
        assert locales_resp.status_code == 200
        assert "cache-control" in locales_resp.headers
        codes = {item["code"] for item in locales_resp.json()["locales"]}
        assert codes == {"en", "es", "fr"}

        fr_resp = client.get("/api/v1/locales/fr.json")
        assert fr_resp.status_code == 200
        assert "cache-control" in fr_resp.headers
        assert fr_resp.json()["nav"]["simple"] == "Transcrire"

        missing_locale = client.get("/api/v1/locales/xx.json")
        assert missing_locale.status_code == 404
        assert missing_locale.json()["error_code"] == "LOCALE_NOT_FOUND"

        bad_locale = client.get("/api/v1/locales/es.evil.json")
        assert bad_locale.status_code == 400
        assert bad_locale.json()["error_code"] == "VALIDATION_ERROR"

        encoded_traversal = client.get("/api/v1/locales/..%2Fes.json")
        assert encoded_traversal.status_code in {400, 404}

        me_get = client.get("/api/v1/me")
        assert me_get.status_code == 200
        me_data = me_get.json()
        assert me_data["id"] == "default"
        assert me_data["role"] == "admin"
        assert me_data["preferences"]["locale"] == "es"

        me_patch = client.patch(
            "/api/v1/me",
            json={
                "locale": "en",
                "default_profile_id": "en-general",
                "player": {"playback_rate": 1.5, "auto_scroll": False},
            },
        )
        assert me_patch.status_code == 200
        patched_prefs = me_patch.json()["preferences"]
        assert patched_prefs["locale"] == "en"
        assert patched_prefs["default_profile_id"] == "en-general"
        assert patched_prefs["player"] == {"playback_rate": 1.5, "auto_scroll": False}

        # Verify persistence across GET /api/v1/me
        me_after = client.get("/api/v1/me")
        assert me_after.status_code == 200
        assert me_after.json()["preferences"] == patched_prefs

        # Invalid preference validation returns structured error_code
        invalid_me = client.patch("/api/v1/me", json={"player": {"playback_rate": 99.0}})
        assert invalid_me.status_code == 422
        assert invalid_me.json()["error_code"] == "VALIDATION_ERROR"

    assert fake_coord.stopped is True


def test_password_auth_cookie_not_secure_on_http_and_protects_endpoints(tmp_path: Path) -> None:
    fake_coord = FakeCoordinator()
    settings = Settings(
        data_dir=tmp_path,
        auth_mode="password",
        auth_password="super-secret-password",
    )
    app = create_app(settings, coordinator=fake_coord)  # type: ignore[arg-type]

    with TestClient(app) as client:
        # Unauthenticated status
        status_before = client.get("/api/v1/auth/status")
        assert status_before.status_code == 200
        assert status_before.json() == {
            "auth_mode": "password",
            "authenticated": False,
            "setup_required": True,
        }

        # Public routes stay accessible
        assert client.get("/healthz").status_code == 200
        assert client.get("/api/v1/locales").status_code == 200

        # Protected routes fail with 401 AUTH_REQUIRED
        me_unauth = client.get("/api/v1/me")
        assert me_unauth.status_code == 401
        assert me_unauth.json()["error_code"] == "AUTH_REQUIRED"

        admin_unauth = client.get("/api/v1/admin/settings")
        assert admin_unauth.status_code == 401
        assert admin_unauth.json()["error_code"] == "AUTH_REQUIRED"

        # Invalid password returns 401 AUTH_INVALID_CREDENTIALS
        bad_login = client.post("/api/v1/auth/login", json={"password": "wrong"})
        assert bad_login.status_code == 401
        assert bad_login.json()["error_code"] == "AUTH_INVALID_CREDENTIALS"

        # Valid login over HTTP must NOT mark Set-Cookie Secure
        good_login = client.post("/api/v1/auth/login", json={"password": "super-secret-password"})
        assert good_login.status_code == 200
        assert good_login.json() == {"authenticated": True}
        set_cookie = good_login.headers.get("set-cookie", "")
        assert "aurisdeck_session=" in set_cookie
        assert "httponly" in set_cookie.lower()
        assert "samesite=lax" in set_cookie.lower()
        assert "secure" not in set_cookie.lower()

        # Authenticated session works
        status_after = client.get("/api/v1/auth/status")
        assert status_after.status_code == 200
        assert status_after.json() == {
            "auth_mode": "password",
            "authenticated": True,
            "setup_required": True,
        }

        me_auth = client.get("/api/v1/me")
        assert me_auth.status_code == 200
        assert me_auth.json()["id"] == "default"

        # HTTPS behind reverse proxy SHOULD mark Set-Cookie Secure
        https_login = client.post(
            "/api/v1/auth/login",
            json={"password": "super-secret-password"},
            headers={"x-forwarded-proto": "https"},
        )
        assert https_login.status_code == 200
        assert "secure" in https_login.headers.get("set-cookie", "").lower()

        # Logout clears session cookie
        logout = client.post("/api/v1/auth/logout")
        assert logout.status_code == 200
        assert logout.json() == {"authenticated": False}

        me_after_logout = client.get("/api/v1/me")
        assert me_after_logout.status_code == 401
        assert me_after_logout.json()["error_code"] == "AUTH_REQUIRED"


def test_bearer_token_and_admin_endpoints_and_persistence(tmp_path: Path) -> None:
    fake_coord = FakeCoordinator()
    settings = Settings(
        data_dir=tmp_path,
        auth_mode="password",
        auth_password="pw",
    )
    app = create_app(settings, coordinator=fake_coord)  # type: ignore[arg-type]

    with TestClient(app) as client:
        token = app.state.auth_service.rotate_api_token("default")

        # Invalid bearer token returns AUTH_INVALID_TOKEN
        bad_bearer = client.get(
            "/api/v1/admin/settings",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert bad_bearer.status_code == 401
        assert bad_bearer.json()["error_code"] == "AUTH_INVALID_TOKEN"

        headers = {"Authorization": f"Bearer {token}"}

        admin_get = client.get("/api/v1/admin/settings", headers=headers)
        assert admin_get.status_code == 200
        assert admin_get.json() == {
            "model_idle_unload_seconds": 120,
            "cpu_threads": 0,
            "log_level": "INFO",
            "hf_token_configured": False,
            "hf_token_masked": None,
        }

        admin_patch = client.patch(
            "/api/v1/admin/settings",
            json={"model_idle_unload_seconds": 300, "cpu_threads": 6, "log_level": "DEBUG"},
            headers=headers,
        )
        assert admin_patch.status_code == 200
        assert admin_patch.json() == {
            "model_idle_unload_seconds": 300,
            "cpu_threads": 6,
            "log_level": "DEBUG",
            "hf_token_configured": False,
            "hf_token_masked": None,
        }
        assert settings.model_idle_unload_seconds == 300
        assert settings.cpu_threads == 6
        assert settings.log_level == "DEBUG"

        devices_resp = client.get("/api/v1/system/devices", headers=headers)
        assert devices_resp.status_code == 200
        devices = devices_resp.json()["devices"]
        assert [dev["id"] for dev in devices] == ["cuda:0", "cpu"]

        unload_resp = client.post("/api/v1/system/unload-models", headers=headers)
        assert unload_resp.status_code == 200
        assert unload_resp.json()["unloaded"] is True
        assert fake_coord.unload_calls == 1

    # Persisted system_settings.json is reloaded on app restart
    fake_coord2 = FakeCoordinator()
    restarted_settings = Settings(data_dir=tmp_path, auth_mode="none")
    restarted_app = create_app(restarted_settings, coordinator=fake_coord2)  # type: ignore[arg-type]
    with TestClient(restarted_app) as restarted_client:
        reloaded = restarted_client.get("/api/v1/admin/settings")
        assert reloaded.status_code == 200
        assert reloaded.json() == {
            "model_idle_unload_seconds": 300,
            "cpu_threads": 6,
            "log_level": "DEBUG",
            "hf_token_configured": False,
            "hf_token_masked": None,
        }

    # require_admin rejects non-admin users with 403 FORBIDDEN
    with pytest.raises(ApiError) as exc_info:
        require_admin({"id": "u2", "role": "member"})
    assert exc_info.value.status_code == 403
    assert exc_info.value.error_code == "FORBIDDEN"
