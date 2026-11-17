"""Tests for AurisDeck auth primitives, scrypt hashing, session cookies, and AuthService."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from aurisdeck.auth import (
    AuthInvalidCredentialsError,
    AuthInvalidTokenError,
    AuthPasswordNotConfiguredError,
    AuthRequiredError,
    AuthService,
    AuthUserNotFoundError,
    hash_secret,
    hash_token,
    sign_session_cookie,
    verify_secret,
    verify_session_cookie,
)
from aurisdeck.config import Settings
from aurisdeck.db import run_migrations
from aurisdeck.db.repositories import UserRepo


def _make_service(
    tmp_path: Path,
    *,
    auth_mode: str = "none",
    auth_password: str | None = None,
    secret_path: Path | None = None,
) -> tuple[AuthService, UserRepo, Path]:
    db_path = tmp_path / "aurisdeck.db"
    run_migrations(db_path)
    repo = UserRepo(db_path)
    settings = Settings(
        data_dir=tmp_path,
        auth_mode=auth_mode,  # type: ignore[arg-type]
        auth_password=auth_password,
    )
    key_path = secret_path or (tmp_path / ".session_secret")
    service = AuthService(user_repo=repo, settings=settings, secret_key_path=key_path)
    return service, repo, key_path


def test_scrypt_hash_and_verify_roundtrip() -> None:
    h1 = hash_secret("correct-horse-battery-staple")
    h2 = hash_secret("correct-horse-battery-staple")

    assert h1.startswith("scrypt$")
    assert h1 != h2
    assert verify_secret("correct-horse-battery-staple", h1) is True
    assert verify_secret("correct-horse-battery-staple", h2) is True
    assert verify_secret("wrong-password", h1) is False
    assert verify_secret("", h1) is False

    unicode_hash = hash_secret("contraseña-ñ-🔐")
    assert verify_secret("contraseña-ñ-🔐", unicode_hash) is True
    assert verify_secret("contrasena-n", unicode_hash) is False


def test_scrypt_rejects_empty_plain_and_malformed_hashes() -> None:
    with pytest.raises(AuthInvalidCredentialsError) as exc_info:
        hash_secret("")
    assert exc_info.value.error_code == "AUTH_INVALID_CREDENTIALS"

    assert verify_secret("secret", "") is False
    assert verify_secret("secret", "plain$salt$hash") is False
    assert verify_secret("secret", "scrypt$notanint$8$1$aa$bb") is False
    assert verify_secret("secret", "scrypt$16384$8$1$%%%$%%%") is False
    assert verify_secret("secret", "scrypt$15$8$1$YQ$YQ") is False


def test_session_cookie_sign_verify_tamper_and_expiry() -> None:
    secret = "super-secret-signing-key"
    cookie = sign_session_cookie("default", secret, ttl_seconds=120)

    assert "." in cookie
    assert verify_session_cookie(cookie, secret) == "default"
    assert verify_session_cookie(cookie, "different-secret") is None

    payload_b64, sig_b64 = cookie.split(".")
    tampered_sig = sig_b64[:-1] + ("A" if sig_b64[-1] != "A" else "B")
    assert verify_session_cookie(f"{payload_b64}.{tampered_sig}", secret) is None
    assert verify_session_cookie("malformed", secret) is None
    assert verify_session_cookie("", secret) is None

    expired = sign_session_cookie("default", secret, ttl_seconds=-1)
    assert verify_session_cookie(expired, secret) is None


def test_secret_key_is_created_with_0600_and_persists_across_restarts(tmp_path: Path) -> None:
    service1, repo, key_path = _make_service(tmp_path, auth_mode="password")
    assert key_path.is_file()
    mode = stat.S_IMODE(key_path.stat().st_mode)
    assert mode == 0o600

    cookie = service1.create_session_cookie("default")

    settings = Settings(data_dir=tmp_path, auth_mode="password")
    service2 = AuthService(user_repo=repo, settings=settings, secret_key_path=key_path)
    assert service2.secret_key == service1.secret_key
    user = service2.resolve_user(cookie_token=cookie, bearer_token=None)
    assert user["id"] == "default"
    assert user["role"] == "admin"


def test_resolve_user_in_none_mode(tmp_path: Path) -> None:
    service, _, _ = _make_service(tmp_path, auth_mode="none")

    user = service.resolve_user(cookie_token=None, bearer_token=None)
    assert user["id"] == "default"
    assert user["username"] == "default"
    assert user["role"] == "admin"
    assert user["preferences"]["locale"] == "es"
    assert "password_hash" not in user
    assert "api_token_hash" not in user

    # Stale or invalid cookies are ignored in auth_mode='none'
    user_with_bad_cookie = service.resolve_user(cookie_token="invalid.cookie", bearer_token=None)
    assert user_with_bad_cookie["id"] == "default"


def test_resolve_user_in_password_mode_requires_valid_cookie(tmp_path: Path) -> None:
    service, _, _ = _make_service(tmp_path, auth_mode="password", auth_password="env-password")

    with pytest.raises(AuthRequiredError) as missing_exc:
        service.resolve_user(cookie_token=None, bearer_token=None)
    assert missing_exc.value.error_code == "AUTH_REQUIRED"

    with pytest.raises(AuthRequiredError) as invalid_exc:
        service.resolve_user(cookie_token="bogus.cookie", bearer_token=None)
    assert invalid_exc.value.error_code == "AUTH_REQUIRED"

    expired = sign_session_cookie("default", service.secret_key, ttl_seconds=-5)
    with pytest.raises(AuthRequiredError) as expired_exc:
        service.resolve_user(cookie_token=expired, bearer_token=None)
    assert expired_exc.value.error_code == "AUTH_REQUIRED"

    valid_cookie = service.create_session_cookie("default")
    user = service.resolve_user(cookie_token=valid_cookie, bearer_token=None)
    assert user["id"] == "default"
    assert "password_hash" not in user
    assert "api_token_hash" not in user


def test_bearer_token_rotation_and_resolution(tmp_path: Path) -> None:
    service, repo, _ = _make_service(tmp_path, auth_mode="password")

    token1 = service.rotate_api_token("default")
    stored_user = repo.get_by_username("default")
    assert stored_user is not None
    assert stored_user.api_token_hash == hash_token(token1)
    assert stored_user.api_token_hash != token1

    resolved = service.resolve_user(cookie_token=None, bearer_token=token1)
    assert resolved["id"] == "default"

    with pytest.raises(AuthInvalidTokenError) as bad_exc:
        service.resolve_user(cookie_token=None, bearer_token="wrong-token")
    assert bad_exc.value.error_code == "AUTH_INVALID_TOKEN"

    with pytest.raises(AuthInvalidTokenError) as empty_exc:
        service.resolve_user(cookie_token=None, bearer_token="   ")
    assert empty_exc.value.error_code == "AUTH_INVALID_TOKEN"

    # Rotating invalidates old token and activates new one
    token2 = service.rotate_api_token("default")
    assert token2 != token1
    with pytest.raises(AuthInvalidTokenError):
        service.resolve_user(cookie_token=None, bearer_token=token1)
    assert service.resolve_user(cookie_token=None, bearer_token=token2)["id"] == "default"


def test_bearer_token_is_still_validated_in_none_mode(tmp_path: Path) -> None:
    service, _, _ = _make_service(tmp_path, auth_mode="none")
    token = service.rotate_api_token("default")

    assert service.resolve_user(cookie_token=None, bearer_token=token)["id"] == "default"
    with pytest.raises(AuthInvalidTokenError) as exc_info:
        service.resolve_user(cookie_token=None, bearer_token="invalid-cli-token")
    assert exc_info.value.error_code == "AUTH_INVALID_TOKEN"


def test_verify_shared_password_and_reset_password_precedence(tmp_path: Path) -> None:
    service_no_pw, _, _ = _make_service(tmp_path, auth_mode="password", auth_password=None)
    with pytest.raises(AuthPasswordNotConfiguredError) as not_cfg_exc:
        service_no_pw.verify_shared_password("anything")
    assert not_cfg_exc.value.error_code == "AUTH_PASSWORD_NOT_CONFIGURED"

    service, repo, _ = _make_service(
        tmp_path / "with_env",
        auth_mode="password",
        auth_password="initial-env-password",
    )
    assert service.verify_shared_password("initial-env-password") is True
    assert service.verify_shared_password("wrong") is False

    service.reset_password("db-override-password")
    stored = repo.get_by_username("default")
    assert stored is not None
    assert stored.password_hash is not None
    assert stored.password_hash.startswith("scrypt$")
    assert "db-override-password" not in stored.password_hash

    assert service.verify_shared_password("db-override-password") is True
    assert service.verify_shared_password("initial-env-password") is False

    with pytest.raises(AuthUserNotFoundError) as missing_user_exc:
        service.reset_password("abc", username="nobody")
    assert missing_user_exc.value.error_code == "AUTH_USER_NOT_FOUND"

    with pytest.raises(AuthInvalidCredentialsError) as empty_pw_exc:
        service.reset_password("")
    assert empty_pw_exc.value.error_code == "AUTH_INVALID_CREDENTIALS"
