"""Authentication service, password hashing, and session/token helpers."""

from aurisdeck.auth.service import (
    DEFAULT_SESSION_TTL_SECONDS,
    AuthError,
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

__all__ = [
    "DEFAULT_SESSION_TTL_SECONDS",
    "AuthError",
    "AuthInvalidCredentialsError",
    "AuthInvalidTokenError",
    "AuthPasswordNotConfiguredError",
    "AuthRequiredError",
    "AuthService",
    "AuthUserNotFoundError",
    "hash_secret",
    "hash_token",
    "sign_session_cookie",
    "verify_secret",
    "verify_session_cookie",
]
