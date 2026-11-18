"""Initial generic profile seeding for AurisDeck."""

from __future__ import annotations

from aurisdeck.domain.models import (
    ProfileAsrConfig,
    ProfileDiarizationConfig,
    ProfileInputConfig,
    ProfileOutputConfig,
    ProfilePreprocessingConfig,
    ProfileSchema,
    SubtitlesOutputConfig,
)
from aurisdeck.storage.stores import ProfileStore


def default_profiles() -> list[ProfileSchema]:
    """Return the 4 built-in generic starter profiles."""
    return [
        ProfileSchema(
            id="es-tecnico",
            display_name="Español + Términos Técnicos",
            simple_visible=True,
            input=ProfileInputConfig(mode="single"),
            asr=ProfileAsrConfig(
                model="large-v3-turbo",
                language="es",
                compute_type="float16",
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            ),
            diarization=ProfileDiarizationConfig(enabled=False),
            output=ProfileOutputConfig(
                subtitles=SubtitlesOutputConfig(
                    voice_style="v_tag",
                    max_cue_seconds=6.0,
                    max_chars_per_line=68,
                    max_lines_per_cue=2,
                ),
            ),
        ),
        ProfileSchema(
            id="subtitulos-video",
            display_name="Subtítulos de Video",
            simple_visible=True,
            input=ProfileInputConfig(mode="single"),
            asr=ProfileAsrConfig(
                model="large-v3-turbo",
                language="es",
                compute_type="float16",
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            ),
            diarization=ProfileDiarizationConfig(enabled=False),
            output=ProfileOutputConfig(
                subtitles=SubtitlesOutputConfig(
                    voice_style="none",
                    max_cue_seconds=5.0,
                    max_chars_per_line=42,
                    max_lines_per_cue=2,
                ),
            ),
        ),
        ProfileSchema(
            id="multipista-sync",
            display_name="Multipista Sincronizada",
            simple_visible=True,
            input=ProfileInputConfig(mode="multitrack"),
            preprocessing=ProfilePreprocessingConfig(generate_mix=True),
            asr=ProfileAsrConfig(
                model="large-v3-turbo",
                language="es",
                compute_type="float16",
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            ),
            diarization=ProfileDiarizationConfig(enabled=False),
            output=ProfileOutputConfig(
                subtitles=SubtitlesOutputConfig(
                    voice_style="v_tag",
                    max_cue_seconds=6.0,
                    max_chars_per_line=68,
                    max_lines_per_cue=2,
                ),
            ),
        ),
        ProfileSchema(
            id="en-general",
            display_name="English General",
            simple_visible=True,
            input=ProfileInputConfig(mode="single"),
            asr=ProfileAsrConfig(
                model="large-v3-turbo",
                language="en",
                compute_type="float16",
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            ),
            diarization=ProfileDiarizationConfig(enabled=False),
            output=ProfileOutputConfig(
                subtitles=SubtitlesOutputConfig(
                    voice_style="v_tag",
                    max_cue_seconds=6.0,
                    max_chars_per_line=68,
                    max_lines_per_cue=2,
                ),
            ),
        ),
    ]


def seed_default_profiles(profile_store: ProfileStore) -> list[ProfileSchema]:
    """Seed the 4 initial profiles only when profiles/ has no existing profiles."""
    if not profile_store.is_empty():
        return []
    created: list[ProfileSchema] = []
    for profile in default_profiles():
        created.append(profile_store.save(profile))
    return created
