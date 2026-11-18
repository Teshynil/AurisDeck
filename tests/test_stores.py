"""Tests for aurisdeck.domain.models, aurisdeck.storage.stores, and defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aurisdeck.domain.models import (
    GlossarySchema,
    ProfileSchema,
    ReplacementDictSchema,
    migrate_profile,
)
from aurisdeck.storage.defaults import seed_default_profiles
from aurisdeck.storage.paths import DataPaths
from aurisdeck.storage.stores import (
    GlossaryStore,
    ProfileStore,
    ReplacementStore,
    build_effective_context,
    import_glossary_text,
)


def test_profile_schema_matches_architecture_v1_json() -> None:
    raw_arch_profile = {
        "schema_version": 1,
        "id": "es-tecnico",
        "owner_user_id": "default",
        "visibility": "shared",
        "display_name": "Español + Términos Técnicos",
        "display_name_i18n": {
            "es": "Español + Términos Técnicos",
            "en": "Spanish + Technical Terms",
        },
        "simple_visible": True,
        "input": {"mode": "single"},
        "preprocessing": {"normalize_audio": False},
        "asr": {
            "backend": "faster-whisper",
            "model": "large-v3-turbo",
            "language": "es",
            "compute_type": "float16",
            "beam_size": 5,
            "vad_filter": True,
            "word_timestamps": True,
        },
        "diarization": {
            "enabled": False,
            "backend": "pyannote-3.1",
            "min_speakers": None,
            "max_speakers": None,
        },
        "context": {
            "initial_prompt": "",
            "hotwords": [],
            "glossary_ids": [],
            "replacement_ids": [],
            "inline_replacements": {},
            "apply_replacements": True,
        },
        "hardware": {
            "gpus": "auto",
            "cpu_fallback": True,
            "cpu_compute_type": "int8",
            "cpu_threads": 0,
        },
        "output": {
            "formats": ["vtt", "md"],
            "subtitles": {
                "voice_style": "v_tag",
                "max_cue_seconds": 6,
                "max_chars_per_line": 68,
                "max_lines_per_cue": 2,
            },
        },
    }

    profile = ProfileSchema.model_validate(raw_arch_profile)
    assert profile.schema_version == 1
    assert profile.id == "es-tecnico"
    assert profile.diarization.backend == "pyannote-3.1"
    assert profile.output.subtitles.voice_style == "v_tag"
    assert profile.output.subtitles.max_cue_seconds == 6.0
    assert profile.context.glossary_ids == []
    assert profile.context.replacement_ids == []
    assert profile.hardware.gpus == "auto"

    # Hardware gpus variants: "cpu" and explicit GPU index list
    cpu_profile = profile.model_copy(
        update={"hardware": profile.hardware.model_copy(update={"gpus": "cpu"})}
    )
    assert cpu_profile.hardware.gpus == "cpu"

    gpu_list_profile = ProfileSchema.model_validate(
        {**raw_arch_profile, "hardware": {"gpus": [0, 1]}}
    )
    assert gpu_list_profile.hardware.gpus == [0, 1]


@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        "../escape",
        "a/b",
        r"a\b",
        "foo..bar",
        "foo.bar",
        "with space",
        ".hidden",
    ],
)
def test_slug_validation_rejects_unsafe_ids(bad_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ProfileSchema(id=bad_id, display_name="Test")

    with pytest.raises(ValidationError):
        GlossarySchema(id=bad_id, name="Test")

    with pytest.raises(ValidationError):
        ReplacementDictSchema(id=bad_id, name="Test")

    with pytest.raises(ValidationError):
        ProfileSchema(
            id="valid-id",
            display_name="Test",
            context={"glossary_ids": [bad_id]},
        )

    paths = DataPaths(tmp_path / "data")
    paths.ensure_dirs()
    store = ProfileStore(paths)
    with pytest.raises(ValueError):
        store.get(bad_id)
    with pytest.raises(ValueError):
        store.delete(bad_id)


def test_profile_validation_constraints() -> None:
    with pytest.raises(ValidationError):
        ProfileSchema(
            id="bad-speakers",
            display_name="Bad",
            diarization={"enabled": True, "min_speakers": 4, "max_speakers": 2},
        )

    with pytest.raises(ValidationError):
        ProfileSchema(
            id="bad-gpus-empty",
            display_name="Bad",
            hardware={"gpus": []},
        )

    with pytest.raises(ValidationError):
        ProfileSchema(
            id="bad-gpus-neg",
            display_name="Bad",
            hardware={"gpus": [-1]},
        )


def test_migrate_profile_handles_legacy_and_rejects_unknown_version() -> None:
    legacy_raw = {
        "id": "legacy-es",
        "name": "Perfil Legacy",
        "diarize": True,
        "asr": {"language": "es"},
    }
    snapshot = dict(legacy_raw)
    migrated = migrate_profile(legacy_raw)

    assert legacy_raw == snapshot  # caller dict not mutated
    assert migrated.schema_version == 1
    assert migrated.display_name == "Perfil Legacy"
    assert migrated.diarization.enabled is True
    assert migrated.asr.backend == "faster-whisper"
    assert migrated.output.subtitles.voice_style == "v_tag"

    zero_version = migrate_profile({"schema_version": 0, "id": "v0", "display_name": "V0"})
    assert zero_version.schema_version == 1

    with pytest.raises(ValueError, match="Unsupported profile schema_version"):
        migrate_profile({"schema_version": 99, "id": "future", "display_name": "Future"})


def test_stores_crud_and_atomic_persistence(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path / "data")
    paths.ensure_dirs()

    profile_store = ProfileStore(paths)
    glossary_store = GlossaryStore(paths)
    replacement_store = ReplacementStore(paths)

    assert profile_store.list() == []
    assert profile_store.is_empty() is True

    profile = ProfileSchema(id="custom_1", display_name="Custom Profile")
    profile_store.save(profile)
    assert profile_store.exists("custom_1") is True
    assert profile_store.is_empty() is False
    loaded_profile = profile_store.get("custom_1")
    assert loaded_profile is not None
    assert loaded_profile.display_name == "Custom Profile"
    assert [p.id for p in profile_store.list()] == ["custom_1"]

    # Legacy file written directly on disk should migrate transparently on get/list
    legacy_path = paths.profiles / "legacy-disk.json"
    legacy_path.write_text(
        json.dumps({"id": "legacy-disk", "name": "On Disk Legacy"}),
        encoding="utf-8",
    )
    loaded_legacy = profile_store.get("legacy-disk")
    assert loaded_legacy is not None
    assert loaded_legacy.schema_version == 1
    assert loaded_legacy.display_name == "On Disk Legacy"

    glossary = GlossarySchema(id="tech_terms", name="Tech Terms", terms=["FastAPI", "  SQLite  "])
    glossary_store.save(glossary)
    assert glossary_store.get("tech_terms") == GlossarySchema(
        id="tech_terms",
        name="Tech Terms",
        terms=["FastAPI", "SQLite"],
    )

    rep = ReplacementDictSchema(
        id="abbr",
        name="Abbreviations",
        rules={"k8s": "Kubernetes"},
        case_sensitive=False,
    )
    replacement_store.save(rep)
    assert replacement_store.get("abbr") == rep

    assert profile_store.delete("custom_1") is True
    assert profile_store.delete("custom_1") is False
    assert profile_store.get("custom_1") is None
    assert list(paths.tmp.iterdir()) == []


def test_import_glossary_text_txt_and_csv() -> None:
    txt_raw = "\ufeff# Comentario inicial\n\n  Kubernetes  \nFastAPI\n# otro\nKubernetes\n  SQLite WAL \n"
    assert import_glossary_text(txt_raw, ".txt") == ["Kubernetes", "FastAPI", "SQLite WAL"]
    assert import_glossary_text(txt_raw, "TXT") == ["Kubernetes", "FastAPI", "SQLite WAL"]

    csv_raw = (
        "# header comment\n"
        "CTranslate2,ASR inference engine\n"
        '"Auris, Deck",Platform name\n'
        ",empty first col\n"
        "CTranslate2,duplicate\n"
        "PyAnnote,Diarization\n"
    )
    assert import_glossary_text(csv_raw, ".csv") == ["CTranslate2", "Auris, Deck", "PyAnnote"]

    with pytest.raises(ValueError, match="Unsupported glossary extension"):
        import_glossary_text("foo\nbar", ".json")


def test_seed_default_profiles_is_idempotent_only_when_empty(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path / "data")
    paths.ensure_dirs()
    profile_store = ProfileStore(paths)

    seeded = seed_default_profiles(profile_store)
    assert [p.id for p in seeded] == [
        "es-tecnico",
        "subtitulos-video",
        "multipista-sync",
        "en-general",
    ]
    assert len(profile_store.list()) == 4

    # Second run is a no-op
    assert seed_default_profiles(profile_store) == []

    # Deleting one profile does NOT trigger re-seeding because profiles/ is not empty
    assert profile_store.delete("en-general") is True
    assert seed_default_profiles(profile_store) == []
    assert sorted(p.id for p in profile_store.list()) == [
        "es-tecnico",
        "multipista-sync",
        "subtitulos-video",
    ]


def test_build_effective_context_merges_and_truncates_safely(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path / "data")
    paths.ensure_dirs()
    glossary_store = GlossaryStore(paths)
    replacement_store = ReplacementStore(paths)

    glossary_store.save(
        GlossarySchema(
            id="g1",
            name="G1",
            terms=["Docker", "CTranslate2", "FastAPI"],
        )
    )
    glossary_store.save(
        GlossarySchema(
            id="g2",
            name="G2",
            terms=["FastAPI", "PyAnnote"],
        )
    )
    replacement_store.save(
        ReplacementDictSchema(
            id="r1",
            name="R1",
            rules={"fast api": "FastAPI", "k8s": "Kubernetes"},
            case_sensitive=False,
        )
    )
    replacement_store.save(
        ReplacementDictSchema(
            id="r2",
            name="R2",
            rules={"k8s": "K8s Cluster", "whisper": "Whisper"},
            case_sensitive=True,
        )
    )

    profile = ProfileSchema(
        id="dev",
        display_name="Dev",
        context={
            "initial_prompt": " ".join(["palabra"] * 100),  # 799 chars
            "hotwords": ["AurisDeck", "Docker"],
            "glossary_ids": ["g1", "missing-glossary", "g2"],
            "replacement_ids": ["r1", "missing-rep", "r2"],
            "inline_replacements": {"whisper": "faster-whisper"},
            "apply_replacements": True,
        },
    )

    ctx = build_effective_context(
        profile,
        glossary_store,
        replacement_store,
        extra_hotwords=["PyAnnote", "WebVTT"],
        extra_prompt=" ".join(["extra"] * 30),
    )

    assert ctx.hotwords == [
        "AurisDeck",
        "Docker",
        "CTranslate2",
        "FastAPI",
        "PyAnnote",
        "WebVTT",
    ]
    assert len(ctx.initial_prompt) <= 850
    assert ctx.initial_prompt.endswith(("palabra", "extra"))
    assert not ctx.initial_prompt.endswith(("palabr", "extr"))
    assert ctx.replacements == {
        "fast api": "FastAPI",
        "k8s": "K8s Cluster",
        "whisper": "faster-whisper",
    }
    assert ctx.case_sensitive is True

    # Verify exact boundary word preservation
    short_profile = ProfileSchema(
        id="boundary",
        display_name="Boundary",
        context={"initial_prompt": "uno dos tres cuatro"},
    )
    exact_boundary = build_effective_context(
        short_profile, glossary_store, replacement_store, max_prompt_chars=7
    )
    assert exact_boundary.initial_prompt == "uno dos"

    mid_word_boundary = build_effective_context(
        short_profile, glossary_store, replacement_store, max_prompt_chars=10
    )
    assert mid_word_boundary.initial_prompt == "uno dos"

    # apply_replacements=False disables replacements
    disabled_profile = profile.model_copy(
        update={"context": profile.context.model_copy(update={"apply_replacements": False})}
    )
    disabled_ctx = build_effective_context(disabled_profile, glossary_store, replacement_store)
    assert disabled_ctx.replacements == {}
    assert disabled_ctx.apply_replacements is False
