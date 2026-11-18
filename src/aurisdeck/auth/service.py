"""Authentication helpers, scrypt password hashing, session cookies, and AuthService."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

from aurisdeck.config import Settings
from aurisdeck.db.repositories import SystemSettingsRepo, UserRecord, UserRepo

DEFAULT_SESSION_TTL_SECONDS = 86400 * 14

_SCRYPT_N = 16384
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_SALT_BYTES = 16
_SCRYPT_DKLEN = 32
_SCRYPT_MAXMEM = 64 * 1024 * 1024


class AuthError(Exception):
    """Base exception for authentication failures exposing an i18n error_code."""

    error_code: str = "AUTH_ERROR"

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        error_params: dict[str, Any] | None = None,
    ) -> None:
        if error_code is not None:
            self.error_code = error_code
        self.error_params: dict[str, Any] = dict(error_params) if error_params else {}
        super().__init__(message or self.error_code)


class AuthRequiredError(AuthError):
    """Raised when password mode requires a valid session or bearer token."""

    error_code = "AUTH_REQUIRED"


class AuthInvalidTokenError(AuthError):
    """Raised when a supplied Bearer API token is malformed or unknown."""

    error_code = "AUTH_INVALID_TOKEN"


class AuthInvalidCredentialsError(AuthError):
    """Raised when credentials fail validation or password verification."""

    error_code = "AUTH_INVALID_CREDENTIALS"


class AuthPasswordNotConfiguredError(AuthError):
    """Raised when password auth is active without a DB hash or env password."""

    error_code = "AUTH_PASSWORD_NOT_CONFIGURED"


class AuthUserNotFoundError(AuthError):
    """Raised when the requested user account does not exist in SQLite."""

    error_code = "AUTH_USER_NOT_FOUND"


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64u_decode(encoded: str) -> bytes:
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + padding)


def hash_secret(plain: str) -> str:
    """Hash a plaintext password with scrypt and a per-password cryptographic salt."""
    if not isinstance(plain, str) or not plain:
        raise AuthInvalidCredentialsError("Password must not be empty")
    salt = secrets.token_bytes(_SCRYPT_SALT_BYTES)
    digest = hashlib.scrypt(
        plain.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        maxmem=_SCRYPT_MAXMEM,
        dklen=_SCRYPT_DKLEN,
    )
    return (
        f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}"
        f"${_b64u_encode(salt)}${_b64u_encode(digest)}"
    )


def verify_secret(plain: str, encoded: str) -> bool:
    """Verify a plaintext password against a stored scrypt hash in constant time."""
    if not plain or not encoded:
        return False
    parts = encoded.split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return False
    try:
        n = int(parts[1])
        r = int(parts[2])
        p = int(parts[3])
        salt = _b64u_decode(parts[4])
        expected = _b64u_decode(parts[5])
    except (ValueError, binascii.Error):
        return False

    if n <= 1 or (n & (n - 1)) != 0 or r < 1 or p < 1 or not salt or not expected:
        return False
    if len(expected) > 128 or 128 * n * r > _SCRYPT_MAXMEM:
        return False

    try:
        derived = hashlib.scrypt(
            plain.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            maxmem=_SCRYPT_MAXMEM,
            dklen=len(expected),
        )
    except ValueError:
        return False
    return hmac.compare_digest(derived, expected)


def hash_token(token: str) -> str:
    """Compute a deterministic SHA-256 hex digest for Bearer API tokens."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def sign_session_cookie(
    user_id: str,
    secret_key: str,
    ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
) -> str:
    """Create an HMAC-SHA256 signed session cookie token with an expiration timestamp."""
    if not user_id:
        raise ValueError("user_id must not be empty")
    if not secret_key:
        raise ValueError("secret_key must not be empty")
    payload = {
        "uid": user_id,
        "exp": time.time() + ttl_seconds,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = _b64u_encode(payload_bytes)
    signature = hmac.new(
        secret_key.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_b64}.{_b64u_encode(signature)}"


def verify_session_cookie(cookie_val: str, secret_key: str) -> str | None:
    """Validate an HMAC-SHA256 signed session cookie and return user_id if valid."""
    if not cookie_val or not secret_key or "." not in cookie_val:
        return None
    parts = cookie_val.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    payload_b64, sig_b64 = parts

    try:
        actual_sig = _b64u_decode(sig_b64)
    except (ValueError, binascii.Error):
        return None

    expected_sig = hmac.new(
        secret_key.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(actual_sig, expected_sig):
        return None

    try:
        payload = json.loads(_b64u_decode(payload_b64).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return None

    if not isinstance(payload, dict):
        return None
    user_id = payload.get("uid")
    expires_at = payload.get("exp")
    if not isinstance(user_id, str) or not user_id:
        return None
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        return None
    if time.time() >= float(expires_at):
        return None
    return user_id


def _user_to_dict(user: UserRecord) -> dict[str, Any]:
    """Serialize UserRecord into a safe CurrentUser dict without credential hashes."""
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "preferences": dict(user.preferences),
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


def _load_or_create_secret(secret_key_path: Path) -> str:
    """Load an existing session secret or atomically create one with 0600 permissions."""
    secret_key_path = Path(secret_key_path)
    if secret_key_path.is_file():
        existing = secret_key_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing

    secret_key_path.parent.mkdir(parents=True, exist_ok=True)
    generated = secrets.token_urlsafe(32)

    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=secret_key_path.parent,
        prefix=f".{secret_key_path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(f"{generated}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, secret_key_path)
        os.chmod(secret_key_path, 0o600)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return generated


class AuthService:
    """Framework-agnostic authentication, token resolution, and credential management."""

    def __init__(
        self,
        user_repo: UserRepo,
        settings: Settings,
        secret_key_path: Path | None = None,
        system_settings_repo: SystemSettingsRepo | None = None,
    ) -> None:
        self._user_repo = user_repo
        self._settings = settings
        self._system_settings_repo = system_settings_repo
        self._secret_key_path = (
            Path(secret_key_path)
            if secret_key_path is not None
            else settings.data_dir / ".session_secret"
        )
        self._secret_key = _load_or_create_secret(self._secret_key_path)

    @property
    def effective_auth_mode(self) -> str:
        """Return active auth mode, honoring DB password mode once initial setup completes."""
        if self._system_settings_repo is not None:
            db_mode = self._system_settings_repo.get_auth_mode()
            if db_mode == "password":
                return "password"
        return self._settings.auth_mode

    @property
    def secret_key(self) -> str:
        """Return the persistent HMAC secret used to sign session cookies."""
        return self._secret_key

    def create_session_cookie(
        self,
        user_id: str = "default",
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    ) -> str:
        """Issue a signed session cookie for the specified user id."""
        return sign_session_cookie(user_id, self._secret_key, ttl_seconds=ttl_seconds)

    def verify_session(self, cookie_val: str | None) -> str | None:
        """Verify a session cookie using the persisted secret key."""
        if not cookie_val:
            return None
        return verify_session_cookie(cookie_val, self._secret_key)

    def resolve_user(
        self,
        cookie_token: str | None,
        bearer_token: str | None,
    ) -> dict[str, Any]:
        """Resolve CurrentUser from Bearer token, auth_mode='none', or session cookie."""
        if bearer_token is not None:
            cleaned = bearer_token.strip()
            if not cleaned:
                raise AuthInvalidTokenError("Invalid bearer token")
            user = self._user_repo.find_by_api_token_hash(hash_token(cleaned))
            if user is None:
                raise AuthInvalidTokenError("Invalid bearer token")
            return _user_to_dict(user)

        if self.effective_auth_mode == "none":
            default_user = self._user_repo.get_by_id("default") or self._user_repo.get_by_username("default")
            if default_user is None:
                raise AuthUserNotFoundError(
                    "Default user not found",
                    error_params={"username": "default"},
                )
            return _user_to_dict(default_user)

        user_id = self.verify_session(cookie_token)
        if user_id is None:
            raise AuthRequiredError("Authentication required")

        user = self._user_repo.get_by_id(user_id)
        if user is None:
            raise AuthRequiredError("Authentication required")
        return _user_to_dict(user)

    def verify_shared_password(self, plain: str) -> bool:
        """Verify the shared login password against DB hash first, then env fallback."""
        user = self._user_repo.get_by_id("default") or self._user_repo.get_by_username("default")
        if user is None:
            raise AuthUserNotFoundError(
                "Default user not found",
                error_params={"username": "default"},
            )

        if user.password_hash:
            return verify_secret(plain, user.password_hash)

        configured = self._settings.auth_password
        if not configured:
            raise AuthPasswordNotConfiguredError(
                "Password authentication is enabled but no password is configured"
            )
        return hmac.compare_digest(plain.encode("utf-8"), configured.encode("utf-8"))

    def verify_credentials(self, username: str | None, plain: str) -> UserRecord | None:
        """Verify username+password or fallback to shared password when username is omitted."""
        cleaned = username.strip() if username else ""
        if cleaned:
            user = self._user_repo.get_by_username(cleaned)
            if user is None or not user.password_hash:
                return None
            if verify_secret(plain, user.password_hash):
                return user
            return None

        if self.verify_shared_password(plain):
            return self._user_repo.get_by_id("default") or self._user_repo.get_by_username("default")
        return None

    def reset_password(self, new_password: str, username: str = "default") -> None:
        """Hash and store a new password for the target user account."""
        if not new_password:
            raise AuthInvalidCredentialsError("Password must not be empty")
        user = self._user_repo.get_by_username(username)
        if user is None:
            raise AuthUserNotFoundError(
                f"User {username!r} not found",
                error_params={"username": username},
            )
        self._user_repo.set_password_hash(user.id, hash_secret(new_password))

    def rotate_api_token(self, username: str = "default") -> str:
        """Generate a new Bearer API token, store its SHA-256 hash, and return plaintext."""
        user = self._user_repo.get_by_username(username)
        if user is None:
            raise AuthUserNotFoundError(
                f"User {username!r} not found",
                error_params={"username": username},
            )
        raw_token = f"adk_{secrets.token_urlsafe(32)}"
        self._user_repo.set_api_token_hash(user.id, hash_token(raw_token))
        return raw_token
