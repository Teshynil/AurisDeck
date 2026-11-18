"""Tests for first-run setup wizard, DB-only HF_TOKEN storage, and token verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aurisdeck.api.app import create_app
from aurisdeck.backends.hf_verify import verify_hf_token
from aurisdeck.config import Settings
from aurisdeck.workers.devices import DeviceInfo


class _FakeHttpError(Exception):
    def __init__(self, status_code: int, message: str = "error") -> None:
        super().__init__(message)
        self.response = type("Resp", (), {"status_code": status_code})()


class _FakeHfApi:
    def __init__(
        self,
        *,
        valid_token: str = "hf_valid_1234",
        gated_missing: set[str] | None = None,
    ) -> None:
        self._valid_token = valid_token
        self._gated_missing = gated_missing or set()

    def whoami(self, token: str | None = None) -> dict[str, Any]:
        if token != self._valid_token:
            raise _FakeHttpError(401, "401 Unauthorized")
        return {"name": "tester"}

    def model_info(
        self,
        *,
        repo_id: str,
        token: str | None = None,
        files_metadata: bool = False,
    ) -> dict[str, Any]:
        if repo_id in self._gated_missing:
            raise _FakeHttpError(403, "403 Forbidden: gated repo")
        return {"id": repo_id}


def _build_test_app(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", auth_mode="none")
    devices = [DeviceInfo(id="cpu", kind="cpu", index=None, name="CPU")]
    return create_app(settings, devices=devices)


def test_verify_hf_token_states() -> None:
    assert verify_hf_token(None).status == "missing_token"
    assert verify_hf_token("   ").status == "missing_token"

    bad = verify_hf_token("hf_wrong", api_client=_FakeHfApi())
    assert not bad.valid
    assert bad.status == "invalid_token"

    gated = verify_hf_token(
        "hf_valid_1234",
        api_client=_FakeHfApi(gated_missing={"pyannote/segmentation-3.0"}),
    )
    assert not gated.valid
    assert gated.status == "gated_terms_required"
    assert gated.missing_models == ["pyannote/segmentation-3.0"]

    ok = verify_hf_token("hf_valid_1234", api_client=_FakeHfApi())
    assert ok.valid
    assert ok.status == "ok"
    assert ok.username == "tester"


def test_env_hf_token_is_ignored_and_db_token_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_from_env_should_be_ignored")
    monkeypatch.setenv("AURISDECK_HF_TOKEN", "hf_from_env_should_be_ignored_too")

    app = _build_test_app(tmp_path)
    with TestClient(app) as client:
        assert app.state.system_settings_repo.get_hf_token() is None
        admin_res = client.get("/api/v1/admin/settings")
        assert admin_res.status_code == 200
        assert admin_res.json()["hf_token_configured"] is False
        assert admin_res.json()["hf_token_masked"] is None


def test_setup_wizard_flow_and_admin_hf_token_management(tmp_path: Path) -> None:
    app = _build_test_app(tmp_path)
    with TestClient(app) as client:
        status_res = client.get("/api/v1/setup/status")
        assert status_res.status_code == 200
        assert status_res.json()["setup_required"] is True

        complete_res = client.post(
            "/api/v1/setup/complete",
            json={
                "locale": "en",
                "username": "teshynil",
                "password": "supersecretpassword",
                "hf_token": "hf_secret_987654321",
            },
        )
        assert complete_res.status_code == 200, complete_res.text
        body = complete_res.json()
        assert body["setup_completed"] is True
        assert body["authenticated"] is True
        assert body["user"]["username"] == "teshynil"
        assert body["user"]["role"] == "admin"
        assert body["user"]["preferences"]["locale"] == "en"

        # Calling setup/complete again must return 409
        repeat_res = client.post(
            "/api/v1/setup/complete",
            json={
                "locale": "es",
                "username": "other",
                "password": "otherpassword",
            },
        )
        assert repeat_res.status_code == 409

        # Admin settings must never expose the raw token
        admin_res = client.get("/api/v1/admin/settings")
        assert admin_res.status_code == 200
        admin_json = admin_res.json()
        assert admin_json["hf_token_configured"] is True
        assert admin_json["hf_token_masked"] == "hf_••••4321"
        assert "hf_secret_987654321" not in str(admin_json)

        # Logout and verify login requires username + password
        client.post("/api/v1/auth/logout")
        unauth_res = client.get("/api/v1/admin/settings")
        assert unauth_res.status_code == 401

        bad_login = client.post(
            "/api/v1/auth/login",
            json={"username": "teshynil", "password": "wrong"},
        )
        assert bad_login.status_code == 401

        ok_login = client.post(
            "/api/v1/auth/login",
            json={"username": "teshynil", "password": "supersecretpassword"},
        )
        assert ok_login.status_code == 200

        # Admin can clear the token
        clear_res = client.patch("/api/v1/admin/settings", json={"hf_token": None})
        assert clear_res.status_code == 200
        assert clear_res.json()["hf_token_configured"] is False
        assert clear_res.json()["hf_token_masked"] is None


def test_cli_reset_password_defaults_to_primary_admin_and_list_users(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from aurisdeck.cli import main as cli_main

    data_dir = tmp_path / "data"
    app = _build_test_app(tmp_path)
    with TestClient(app) as client:
        res = client.post(
            "/api/v1/setup/complete",
            json={
                "locale": "es",
                "username": "myadmin",
                "password": "initialpassword",
            },
        )
        assert res.status_code == 200

    runner = CliRunner()
    list_res = runner.invoke(cli_main, ["auth", "list-users", "--data-dir", str(data_dir)])
    assert list_res.exit_code == 0, list_res.output
    assert "myadmin\tadmin" in list_res.output

    reset_res = runner.invoke(
        cli_main,
        ["auth", "reset-password", "--password", "newpassword123", "--data-dir", str(data_dir)],
    )
    assert reset_res.exit_code == 0, reset_res.output
    assert "myadmin" in reset_res.output

    with TestClient(app) as client:
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": "myadmin", "password": "newpassword123"},
        )
        assert login_res.status_code == 200
