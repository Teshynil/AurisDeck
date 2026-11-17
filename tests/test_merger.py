"""Tests for backend contracts, typed errors, and TranscriptMerger."""

from __future__ import annotations

import ast
from pathlib import Path

import aurisdeck.backends as backends_pkg
import aurisdeck.pipeline as pipeline_pkg
from aurisdeck.backends import (
    ASRBackend,
    ASRCapabilities,
    ASRContext,
    ASRResult,
    ASRSegment,
    ASRWord,
    AurisDeckBackendError,
    DiarizerBackend,
    GatedModelTermsError,
    GpuOomError,
    JobCancelledError,
    MissingCredentialsError,
    ModelLoadError,
    SpeakerTurn,
)
from aurisdeck.export.cue_segmenter import WordTiming
from aurisdeck.pipeline import TranscriptMerger


def test_backend_errors_defaults_and_overrides() -> None:
    base = AurisDeckBackendError("CUSTOM_ERR", {"k": "v"}, "Something failed")
    assert base.error_code == "CUSTOM_ERR"
    assert base.error_params == {"k": "v"}
    assert base.detail == "Something failed"
    assert str(base) == "Something failed"

    assert MissingCredentialsError().error_code == "DIARIZATION_HF_TOKEN_MISSING"
    assert str(MissingCredentialsError()) == "DIARIZATION_HF_TOKEN_MISSING"
    assert GatedModelTermsError().error_code == "DIARIZATION_GATED_TERMS"
    assert GpuOomError().error_code == "GPU_OOM"
    assert ModelLoadError().error_code == "MODEL_LOAD_FAILED"
    assert JobCancelledError().error_code == "JOB_CANCELLED"

    dep_err = ModelLoadError("DIARIZATION_DEPENDENCY_MISSING", detail="pyannote missing")
    assert dep_err.error_code == "DIARIZATION_DEPENDENCY_MISSING"
    assert dep_err.detail == "pyannote missing"
    assert isinstance(dep_err, AurisDeckBackendError)


def test_backend_protocols_and_zero_ml_imports() -> None:
    class _DummyASR:
        capabilities = ASRCapabilities(
            supports_hotwords=True,
            supports_initial_prompt=True,
            supports_word_timestamps=True,
        )

        def ensure_loaded(self) -> None:
            return None

        def transcribe(self, audio_path: Path, context: ASRContext) -> ASRResult:
            return ASRResult(language_detected=context.language, duration_ms=0, segments=[])

        def unload(self) -> None:
            return None

    class _DummyDiarizer:
        def ensure_loaded(self) -> None:
            return None

        def diarize(
            self,
            audio_path: Path,
            min_speakers: int | None = None,
            max_speakers: int | None = None,
        ) -> list[SpeakerTurn]:
            return []

        def unload(self) -> None:
            return None

    assert isinstance(_DummyASR(), ASRBackend)
    assert isinstance(_DummyDiarizer(), DiarizerBackend)

    forbidden = {"torch", "faster_whisper", "ctranslate2", "pyannote"}
    package_dirs = [
        Path(backends_pkg.__file__).parent,
        Path(pipeline_pkg.__file__).parent,
    ]
    for directory in package_dirs:
        for py_file in directory.glob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".")[0] not in forbidden
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".")[0] not in forbidden


def test_merge_single_without_turns_assigns_default_speaker() -> None:
    asr = ASRResult(
        language_detected="es",
        duration_ms=4000,
        segments=[
            ASRSegment(
                start_ms=0,
                end_ms=1800,
                text="Hola kubernetis",
                words=[
                    ASRWord(0, 800, "Hola"),
                    ASRWord(850, 1800, "kubernetis"),
                ],
            ),
            ASRSegment(
                start_ms=2000,
                end_ms=3500,
                text="Segundo segmento",
                words=[],
            ),
        ],
    )

    for turns in (None, []):
        merged = TranscriptMerger.merge_single(
            asr,
            turns=turns,
            replacements={"kubernetis": "Kubernetes"},
        )
        assert [item.seq for item in merged] == [0, 1]
        assert all(item.speaker_key == "SPEAKER_00" for item in merged)
        assert merged[0].raw_text == "Hola kubernetis"
        assert merged[0].text == "Hola Kubernetes"
        assert merged[0].words == [
            WordTiming(0, 800, "Hola"),
            WordTiming(850, 1800, "kubernetis"),
        ]
        assert merged[1].raw_text == "Segundo segmento"
        assert merged[1].text == "Segundo segmento"


def test_merge_single_word_level_split_and_gap_fallback() -> None:
    asr = ASRResult(
        language_detected="es",
        duration_ms=5000,
        segments=[
            ASRSegment(
                start_ms=0,
                end_ms=4000,
                text=" Hola whisper   gracias  mundo",
                words=[
                    ASRWord(100, 700, " Hola"),
                    ASRWord(750, 1300, "  whisper "),
                    # Gap between turns: [0, 1500] (mid=750) vs [2200, 4200] (mid=3200).
                    # Word midpoint is 1900 -> closer to turn 1 (|1900-750|=1150 vs |1900-3200|=1300).
                    ASRWord(1800, 2000, " amigo"),
                    ASRWord(2300, 3000, " gracias"),
                    ASRWord(3100, 3900, "  mundo"),
                ],
            ),
        ],
    )
    turns = [
        SpeakerTurn(start_ms=0, end_ms=1500, speaker_key="SPEAKER_00"),
        SpeakerTurn(start_ms=2200, end_ms=4200, speaker_key="SPEAKER_01"),
    ]

    merged = TranscriptMerger.merge_single(
        asr,
        turns=turns,
        replacements={"whisper": "Whisper"},
    )

    assert len(merged) == 2
    assert [item.seq for item in merged] == [0, 1]

    assert merged[0].speaker_key == "SPEAKER_00"
    assert merged[0].start_ms == 100
    assert merged[0].end_ms == 2000
    assert merged[0].raw_text == "Hola whisper amigo"
    assert merged[0].text == "Hola Whisper amigo"
    assert "  " not in merged[0].raw_text
    assert len(merged[0].words) == 3

    assert merged[1].speaker_key == "SPEAKER_01"
    assert merged[1].start_ms == 2300
    assert merged[1].end_ms == 3900
    assert merged[1].raw_text == "gracias mundo"
    assert merged[1].text == "gracias mundo"
    assert len(merged[1].words) == 2


def test_merge_single_segment_without_words_uses_turn_overlap() -> None:
    asr = ASRResult(
        language_detected="en",
        duration_ms=5000,
        segments=[
            ASRSegment(start_ms=100, end_ms=1200, text="First turn", words=[]),
            ASRSegment(start_ms=1300, end_ms=2800, text="Second turn", words=[]),
        ],
    )
    turns = [
        SpeakerTurn(start_ms=0, end_ms=1400, speaker_key="SPEAKER_00"),
        SpeakerTurn(start_ms=1400, end_ms=3000, speaker_key="SPEAKER_01"),
    ]

    merged = TranscriptMerger.merge_single(asr, turns=turns, replacements={})
    assert [item.speaker_key for item in merged] == ["SPEAKER_00", "SPEAKER_01"]
    assert [item.seq for item in merged] == [0, 1]


def test_merge_multitrack_offsets_chronological_sort_and_replacements() -> None:
    track_alice = ASRResult(
        language_detected="es",
        duration_ms=3000,
        segments=[
            ASRSegment(
                start_ms=500,
                end_ms=1500,
                text="Hola fast api",
                words=[ASRWord(500, 900, "Hola"), ASRWord(950, 1500, "fast api")],
            ),
            ASRSegment(
                start_ms=2000,
                end_ms=2800,
                text="Adiós",
                words=[],
            ),
        ],
    )
    track_bob = ASRResult(
        language_detected="es",
        duration_ms=2000,
        segments=[
            ASRSegment(
                start_ms=100,
                end_ms=900,
                text="Turno temprano de fast api",
                words=[ASRWord(100, 400, "Turno"), ASRWord(450, 900, "temprano")],
            ),
        ],
    )

    merged = TranscriptMerger.merge_multitrack(
        tracks=[
            ("trk-alice", "SPEAKER_00", 1000, track_alice),
            ("trk-bob", "SPEAKER_01", 200, track_bob),
        ],
        replacements={"fast api": "FastAPI"},
        case_sensitive=True,
    )

    assert [item.seq for item in merged] == [0, 1, 2]

    # Bob: 100+200=300 .. 900+200=1100
    assert merged[0].speaker_key == "SPEAKER_01"
    assert merged[0].start_ms == 300
    assert merged[0].end_ms == 1100
    assert merged[0].words == [WordTiming(300, 600, "Turno"), WordTiming(650, 1100, "temprano")]
    assert merged[0].raw_text == "Turno temprano de fast api"
    assert merged[0].text == "Turno temprano de FastAPI"

    # Alice seg 1: 500+1000=1500 .. 1500+1000=2500
    assert merged[1].speaker_key == "SPEAKER_00"
    assert merged[1].start_ms == 1500
    assert merged[1].end_ms == 2500
    assert merged[1].words == [WordTiming(1500, 1900, "Hola"), WordTiming(1950, 2500, "fast api")]
    assert merged[1].raw_text == "Hola fast api"
    assert merged[1].text == "Hola FastAPI"

    # Alice seg 2: 2000+1000=3000 .. 2800+1000=3800
    assert merged[2].speaker_key == "SPEAKER_00"
    assert merged[2].start_ms == 3000
    assert merged[2].end_ms == 3800
