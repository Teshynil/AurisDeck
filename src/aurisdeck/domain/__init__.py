"""Domain schemas and validation helpers for AurisDeck."""

from aurisdeck.domain.models import (
    EffectiveContext,
    GlossarySchema,
    ProfileAsrConfig,
    ProfileContextConfig,
    ProfileDiarizationConfig,
    ProfileHardwareConfig,
    ProfileInputConfig,
    ProfileOutputConfig,
    ProfilePreprocessingConfig,
    ProfileSchema,
    ReplacementDictSchema,
    SubtitlesOutputConfig,
    migrate_profile,
    validate_slug,
)

__all__ = [
    "EffectiveContext",
    "GlossarySchema",
    "ProfileAsrConfig",
    "ProfileContextConfig",
    "ProfileDiarizationConfig",
    "ProfileHardwareConfig",
    "ProfileInputConfig",
    "ProfileOutputConfig",
    "ProfilePreprocessingConfig",
    "ProfileSchema",
    "ReplacementDictSchema",
    "SubtitlesOutputConfig",
    "migrate_profile",
    "validate_slug",
]
