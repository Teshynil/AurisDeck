"""Pydantic v2 domain contracts for profiles, glossaries, replacements, and effective context."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    NonNegativeInt,
    field_validator,
    model_validator,
)

_SLUG_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_slug(value: str) -> str:
    """Validate that an entity identifier is a safe slug matching [a-zA-Z0-9_-]+."""
    if not isinstance(value, str) or not _SLUG_RE.fullmatch(value):
        raise ValueError(
            f"Invalid slug id {value!r}: must match [a-zA-Z0-9_-]+ without '/' or '..'"
        )
    return value


SlugStr = Annotated[str, AfterValidator(validate_slug)]


class ProfileInputConfig(BaseModel):
    """Input ingestion mode configuration for a transcription profile."""

    mode: Literal["single", "multitrack", "batch"] = "single"


class ProfilePreprocessingConfig(BaseModel):
    """Audio preprocessing options applied prior to ASR."""

    normalize_audio: bool = False
    generate_mix: bool = False


class ProfileAsrConfig(BaseModel):
    """ASR engine parameters."""

    backend: str = Field(default="faster-whisper", min_length=1)
    model: str = Field(default="large-v3-turbo", min_length=1)
    language: str | None = "es"
    compute_type: str = Field(default="float16", min_length=1)
    beam_size: int = Field(default=5, ge=1)
    vad_filter: bool = True
    word_timestamps: bool = True


class ProfileDiarizationConfig(BaseModel):
    """Speaker diarization parameters."""

    enabled: bool = False
    backend: str = Field(default="pyannote-3.1", min_length=1)
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_speaker_bounds(self) -> ProfileDiarizationConfig:
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.max_speakers < self.min_speakers
        ):
            raise ValueError("max_speakers must be greater than or equal to min_speakers")
        return self


class ProfileContextConfig(BaseModel):
    """Vocabulary bias and post-ASR replacement configuration."""

    initial_prompt: str = ""
    hotwords: list[str] = Field(default_factory=list)
    glossary_ids: list[SlugStr] = Field(default_factory=list)
    replacement_ids: list[SlugStr] = Field(default_factory=list)
    inline_replacements: dict[str, str] = Field(default_factory=dict)
    apply_replacements: bool = True


class ProfileHardwareConfig(BaseModel):
    """Device placement and CPU fallback configuration."""

    gpus: Literal["auto", "cpu"] | Annotated[list[NonNegativeInt], Field(min_length=1)] = "auto"
    cpu_fallback: bool = True
    cpu_compute_type: str = Field(default="int8", min_length=1)
    cpu_threads: int = Field(default=0, ge=0)


class SubtitlesOutputConfig(BaseModel):
    """Subtitle cue formatting and speaker tag styling."""

    voice_style: Literal["v_tag", "prefix", "none"] = "v_tag"
    max_cue_seconds: float = Field(default=6.0, gt=0)
    max_chars_per_line: int = Field(default=68, ge=1)
    max_lines_per_cue: int = Field(default=2, ge=1)


class ProfileOutputConfig(BaseModel):
    """Subtitle rendering options."""

    subtitles: SubtitlesOutputConfig = Field(default_factory=SubtitlesOutputConfig)


class ProfileSchema(BaseModel):
    """Unified transcription profile JSON schema (schema_version=1)."""

    schema_version: Literal[1] = 1
    id: SlugStr
    owner_user_id: str = Field(default="default", min_length=1)
    visibility: Literal["shared", "private"] = "shared"
    display_name: str = Field(min_length=1)
    simple_visible: bool = True
    input: ProfileInputConfig = Field(default_factory=ProfileInputConfig)
    preprocessing: ProfilePreprocessingConfig = Field(default_factory=ProfilePreprocessingConfig)
    asr: ProfileAsrConfig = Field(default_factory=ProfileAsrConfig)
    diarization: ProfileDiarizationConfig = Field(default_factory=ProfileDiarizationConfig)
    context: ProfileContextConfig = Field(default_factory=ProfileContextConfig)
    hardware: ProfileHardwareConfig = Field(default_factory=ProfileHardwareConfig)
    output: ProfileOutputConfig = Field(default_factory=ProfileOutputConfig)


class GlossarySchema(BaseModel):
    """Reusable shared glossary of domain terms."""

    id: SlugStr
    name: str = Field(min_length=1)
    owner_user_id: str = Field(default="default", min_length=1)
    visibility: Literal["shared", "private"] = "shared"
    terms: list[str] = Field(default_factory=list)

    @field_validator("terms")
    @classmethod
    def _clean_terms(cls, values: list[str]) -> list[str]:
        return [item.strip() for item in values if isinstance(item, str) and item.strip()]


class ReplacementDictSchema(BaseModel):
    """Reusable post-ASR text replacement dictionary."""

    id: SlugStr
    name: str = Field(min_length=1)
    owner_user_id: str = Field(default="default", min_length=1)
    visibility: Literal["shared", "private"] = "shared"
    rules: dict[str, str] = Field(default_factory=dict)
    case_sensitive: bool = False

    @field_validator("rules")
    @classmethod
    def _validate_rules(cls, rules: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for source, target in rules.items():
            if not source.strip():
                raise ValueError("Replacement source key cannot be empty")
            cleaned[source] = target
        return cleaned


class EffectiveContext(BaseModel):
    """Resolved prompt, hotwords, and replacement rules for a transcription job."""

    initial_prompt: str = ""
    hotwords: list[str] = Field(default_factory=list)
    replacements: dict[str, str] = Field(default_factory=dict)
    case_sensitive: bool = False
    apply_replacements: bool = True


def migrate_profile(raw: dict[str, Any]) -> ProfileSchema:
    """Normalize and migrate a raw profile dictionary into ProfileSchema v1."""
    if not isinstance(raw, dict):
        raise TypeError("Profile payload must be a JSON object")

    data = dict(raw)
    version = data.get("schema_version")

    if version is None or version == 0:
        data["schema_version"] = 1
        if "display_name" not in data and isinstance(data.get("name"), str):
            data["display_name"] = data.pop("name")
        if "diarization" not in data and "diarize" in data:
            data["diarization"] = {"enabled": bool(data.pop("diarize"))}
    elif version != 1:
        raise ValueError(f"Unsupported profile schema_version: {version!r}")

    return ProfileSchema.model_validate(data)
