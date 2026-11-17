"""Centralized application settings for AurisDeck."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from AURISDECK_* and standard Hugging Face env vars."""

    model_config = SettingsConfigDict(
        env_prefix="AURISDECK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path = Path("./data")
    auth_mode: Literal["none", "password"] = "none"
    auth_password: str | None = None
    base_path: str = ""
    model_idle_unload_seconds: int = Field(default=120, ge=0)
    cpu_threads: int = Field(default=0, ge=0)

    hf_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_ENDPOINT", "AURISDECK_HF_ENDPOINT", "hf_endpoint"),
    )
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "AURISDECK_HF_TOKEN", "hf_token"),
    )

    @field_validator("base_path", mode="before")
    @classmethod
    def _normalize_base_path(cls, value: object) -> str:
        """Normalize base_path without a trailing slash (and empty for root)."""
        if value is None:
            return ""
        normalized = str(value).strip().rstrip("/")
        if not normalized:
            return ""
        if not normalized.startswith("/"):
            return f"/{normalized}"
        return normalized
