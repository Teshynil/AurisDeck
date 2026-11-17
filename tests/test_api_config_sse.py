"""Tests for configuration CRUD endpoints (/profiles, /glossaries, /replacements) and SSE."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aurisdeck.api import create_app
from aurisdeck.api.routes_config import router as config_router
from aurisdeck.api.routes_events import router as events_router
from aurisdeck.api.sse import SSEBroker, format_sse_event
from aurisdeck.config import Settings
from aurisdeck.storage.stores import GlossaryStore, ProfileStore, ReplacementStore
from aurisdeck.workers.devices import DeviceInfo
from aurisdeck.workers.events import EventBus


class _FakeCoordinator:
    """Minimal coordinator stub so API tests never spawn worker subprocesses."""

    def __init__(self) -> None:
        self._devices = [DeviceInfo(id="cpu", kind="cpu", index=None, name="CPU (4 threads)")]

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def list_devices(self) -> list[DeviceInfo]:
        return list(self._devices)

    def unload_all_models(self) -> None:
        return None


def _build_test_app(
    tmp_path: Path,
    *,
    auth_mode: str = "none",
    auth_password: str = "",
) -> FastAPI:
    settings = Settings(
        data_dir=tmp_path,
        auth_mode=auth_mode,  # type: ignore[arg-type]
        auth_password=auth_password,
    )
    app = create_app(settings, coordinator=_FakeCoordinator())  # type: ignore[arg-type]
    registered_paths = {getattr(route, "path", None) for route in app.routes}
    if "/api/v1/profiles" not in registered_paths:
        app.include_router(config_router)
    if "/api/v1/events" not in registered_paths:
        app.include_router(events_router)
    return app


def _sample_profile(profile_id: str = "custom-es", display_name: str = "Custom ES") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": profile_id,
        "owner_user_id": "default",
        "visibility": "shared",
        "display_name": display_name,
        "display_name_i18n": {"es": display_name, "en": display_name},
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
            "initial_prompt": "AurisDeck session",
            "hotwords": ["AurisDeck"],
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
            "formats": ["vtt", "json"],
            "subtitles": {
                "voice_style": "v_tag",
                "max_cue_seconds": 6.0,
                "max_chars_per_line": 68,
                "max_lines_per_cue": 2,
            },
        },
    }


def test_config_and_events_require_auth_when_password_enabled(tmp_path: Path) -> None:
    app = _build_test_app(tmp_path, auth_mode="password", auth_password="secret")
    with TestClient(app) as client:
        for path in (
            "/api/v1/profiles",
            "/api/v1/glossaries",
            "/api/v1/replacements",
            "/api/v1/events",
        ):
            resp = client.get(path)
            assert resp.status_code == 401
            assert resp.json()["error_code"] == "AUTH_REQUIRED"


def test_profiles_crud_and_error_codes(tmp_path: Path) -> None:
    app = _build_test_app(tmp_path)
    with TestClient(app) as client:
        # Seeded defaults are listed
        listed = client.get("/api/v1/profiles")
        assert listed.status_code == 200
        seeded_ids = {item["id"] for item in listed.json()}
        assert "es-tecnico" in seeded_ids

        # 404 on missing profile
        missing = client.get("/api/v1/profiles/does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["error_code"] == "PROFILE_NOT_FOUND"

        # 422 on invalid slug path param
        invalid = client.get("/api/v1/profiles/bad.slug")
        assert invalid.status_code == 422
        assert invalid.json()["error_code"] == "VALIDATION_ERROR"

        # Create (201)
        payload = _sample_profile("meeting-es", "Reunión ES")
        created = client.post("/api/v1/profiles", json=payload)
        assert created.status_code == 201
        assert created.json()["id"] == "meeting-es"
        assert created.json()["display_name"] == "Reunión ES"

        # Duplicate POST -> 409 PROFILE_ALREADY_EXISTS
        dup = client.post("/api/v1/profiles", json=payload)
        assert dup.status_code == 409
        assert dup.json()["error_code"] == "PROFILE_ALREADY_EXISTS"

        # GET created
        fetched = client.get("/api/v1/profiles/meeting-es")
        assert fetched.status_code == 200
        assert fetched.json()["context"]["hotwords"] == ["AurisDeck"]

        # PUT updates and enforces id == profile_id
        update_payload = _sample_profile("ignored-id", "Reunión Actualizada")
        update_payload["asr"]["beam_size"] = 3
        updated = client.put("/api/v1/profiles/meeting-es", json=update_payload)
        assert updated.status_code == 200
        assert updated.json()["id"] == "meeting-es"
        assert updated.json()["display_name"] == "Reunión Actualizada"
        assert updated.json()["asr"]["beam_size"] == 3
        assert ProfileStore(app.state.paths).exists("ignored-id") is False

        # PUT on nonexistent -> 404 PROFILE_NOT_FOUND
        missing_put = client.put("/api/v1/profiles/missing-prof", json=_sample_profile("missing-prof"))
        assert missing_put.status_code == 404
        assert missing_put.json()["error_code"] == "PROFILE_NOT_FOUND"

        # DELETE
        deleted = client.delete("/api/v1/profiles/meeting-es")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True, "id": "meeting-es"}

        # DELETE again -> 404 PROFILE_NOT_FOUND
        deleted_again = client.delete("/api/v1/profiles/meeting-es")
        assert deleted_again.status_code == 404
        assert deleted_again.json()["error_code"] == "PROFILE_NOT_FOUND"


def test_glossaries_crud_and_import_txt_csv_and_raw_text(tmp_path: Path) -> None:
    app = _build_test_app(tmp_path)
    with TestClient(app) as client:
        # Empty list initially
        assert client.get("/api/v1/glossaries").json() == []

        # 404 missing
        missing = client.get("/api/v1/glossaries/no-such-glossary")
        assert missing.status_code == 404
        assert missing.json()["error_code"] == "GLOSSARY_NOT_FOUND"

        # POST create (201)
        glossary_payload = {
            "id": "tech-terms",
            "name": "Términos Técnicos",
            "owner_user_id": "default",
            "visibility": "shared",
            "terms": ["FastAPI", "CTranslate2", "FastAPI"],
        }
        created = client.post("/api/v1/glossaries", json=glossary_payload)
        assert created.status_code == 201
        assert created.json()["terms"] == ["FastAPI", "CTranslate2", "FastAPI"]

        # Duplicate POST -> 409
        dup = client.post("/api/v1/glossaries", json=glossary_payload)
        assert dup.status_code == 409
        assert dup.json()["error_code"] == "GLOSSARY_ALREADY_EXISTS"

        # PUT update enforcing id == glossary_id
        updated = client.put(
            "/api/v1/glossaries/tech-terms",
            json={
                "id": "other-id",
                "name": "Tech Updated",
                "terms": ["Whisper", "Pyannote"],
            },
        )
        assert updated.status_code == 200
        assert updated.json()["id"] == "tech-terms"
        assert updated.json()["name"] == "Tech Updated"
        assert updated.json()["terms"] == ["Whisper", "Pyannote"]

        # Import .txt via multipart file upload (creates new glossary)
        txt_bytes = b"# header comment\nAurisDeck\nSilero VAD\nAurisDeck\n\nWebVTT\n"
        imp_txt = client.post(
            "/api/v1/glossaries/import",
            files={"file": ("lore-pack.txt", txt_bytes, "text/plain")},
            data={"glossary_id": "lore-txt", "display_name": "Lore TXT"},
        )
        assert imp_txt.status_code == 200
        assert imp_txt.json()["id"] == "lore-txt"
        assert imp_txt.json()["name"] == "Lore TXT"
        assert imp_txt.json()["terms"] == ["AurisDeck", "Silero VAD", "WebVTT"]

        # Import .csv via multipart file upload (updates existing glossary)
        csv_bytes = b"Kaelen,protagonist\nThalassia,city\nKaelen,duplicate\n"
        imp_csv = client.post(
            "/api/v1/glossaries/import",
            files={"file": ("lore.csv", csv_bytes, "text/csv")},
            data={"glossary_id": "lore-txt", "display_name": "Lore CSV Updated"},
        )
        assert imp_csv.status_code == 200
        assert imp_csv.json()["id"] == "lore-txt"
        assert imp_csv.json()["name"] == "Lore CSV Updated"
        assert imp_csv.json()["terms"] == ["Kaelen", "Thalassia"]
        assert GlossaryStore(app.state.paths).get("lore-txt") is not None

        # Import via raw_text form fields (plain text / CSV)
        imp_raw = client.post(
            "/api/v1/glossaries/import",
            data={
                "raw_text": "alpha,one\nbeta,two\nalpha,dup\n",
                "glossary_id": "raw-csv",
                "display_name": "Raw CSV",
                "ext": "csv",
            },
        )
        assert imp_raw.status_code == 200
        assert imp_raw.json()["id"] == "raw-csv"
        assert imp_raw.json()["terms"] == ["alpha", "beta"]

        # Invalid import requests
        no_input = client.post("/api/v1/glossaries/import", data={"glossary_id": "empty"})
        assert no_input.status_code == 400
        assert no_input.json()["error_code"] == "VALIDATION_ERROR"

        bad_ext = client.post(
            "/api/v1/glossaries/import",
            data={"raw_text": "foo", "glossary_id": "bad-ext", "ext": "json"},
        )
        assert bad_ext.status_code == 400
        assert bad_ext.json()["error_code"] == "VALIDATION_ERROR"

        # DELETE & 404 afterwards
        deleted = client.delete("/api/v1/glossaries/tech-terms")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True, "id": "tech-terms"}

        after_del = client.get("/api/v1/glossaries/tech-terms")
        assert after_del.status_code == 404
        assert after_del.json()["error_code"] == "GLOSSARY_NOT_FOUND"


def test_replacements_crud_and_error_codes(tmp_path: Path) -> None:
    app = _build_test_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/v1/replacements").json() == []

        missing = client.get("/api/v1/replacements/no-dict")
        assert missing.status_code == 404
        assert missing.json()["error_code"] == "REPLACEMENT_NOT_FOUND"

        payload = {
            "id": "phonetic-fixes",
            "name": "Correcciones fonéticas",
            "owner_user_id": "default",
            "visibility": "shared",
            "rules": {"auris deck": "AurisDeck", "whispper": "Whisper"},
            "case_sensitive": False,
        }
        created = client.post("/api/v1/replacements", json=payload)
        assert created.status_code == 201
        assert created.json()["id"] == "phonetic-fixes"

        dup = client.post("/api/v1/replacements", json=payload)
        assert dup.status_code == 409
        assert dup.json()["error_code"] == "REPLACEMENT_ALREADY_EXISTS"

        listed = client.get("/api/v1/replacements")
        assert listed.status_code == 200
        assert [item["id"] for item in listed.json()] == ["phonetic-fixes"]

        updated = client.put(
            "/api/v1/replacements/phonetic-fixes",
            json={
                "id": "wrong-id",
                "name": "Correcciones v2",
                "rules": {"silero bat": "Silero VAD"},
                "case_sensitive": True,
            },
        )
        assert updated.status_code == 200
        assert updated.json()["id"] == "phonetic-fixes"
        assert updated.json()["case_sensitive"] is True
        assert updated.json()["rules"] == {"silero bat": "Silero VAD"}
        assert ReplacementStore(app.state.paths).exists("wrong-id") is False

        deleted = client.delete("/api/v1/replacements/phonetic-fixes")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True, "id": "phonetic-fixes"}

        missing_del = client.delete("/api/v1/replacements/phonetic-fixes")
        assert missing_del.status_code == 404
        assert missing_del.json()["error_code"] == "REPLACEMENT_NOT_FOUND"


def test_sse_broker_events_keepalive_and_unsubscribe() -> None:
    async def _run() -> None:
        bus = EventBus()
        broker = SSEBroker(bus, heartbeat_seconds=0.05)

        formatted = format_sse_event(7, "job.progress", {"v": 1, "payload": {"pct": 50}})
        assert formatted == 'id: 7\nevent: job.progress\ndata: {"v":1,"payload":{"pct":50}}\n\n'

        gen = broker.stream()
        # Start streaming and wait for initial keepalive when no events are published
        first_chunk = await anext(gen)
        assert first_chunk == ": keepalive\n\n"
        assert len(bus._subscribers) == 1

        bus.publish("job.created", job_id="job-1", payload={"status": "queued"})
        bus.publish("job.progress", job_id="job-1", payload={"processed_ms": 1000})

        ev1 = await anext(gen)
        ev2 = await anext(gen)

        lines1 = ev1.strip().splitlines()
        assert lines1[0] == "id: 1"
        assert lines1[1] == "event: job.created"
        assert lines1[2].startswith("data: ")
        parsed1 = json.loads(lines1[2].removeprefix("data: "))
        assert parsed1["type"] == "job.created"
        assert parsed1["job_id"] == "job-1"
        assert parsed1["payload"] == {"status": "queued"}

        lines2 = ev2.strip().splitlines()
        assert lines2[0] == "id: 2"
        assert lines2[1] == "event: job.progress"
        parsed2 = json.loads(lines2[2].removeprefix("data: "))
        assert parsed2["payload"] == {"processed_ms": 1000}

        await gen.aclose()
        assert len(bus._subscribers) == 0

    asyncio.run(_run())


def test_sse_broker_stops_when_request_disconnected() -> None:
    class _FakeRequest:
        def __init__(self) -> None:
            self.checks = 0

        async def is_disconnected(self) -> bool:
            self.checks += 1
            return self.checks >= 2

    async def _run() -> None:
        bus = EventBus()
        broker = SSEBroker(bus, heartbeat_seconds=0.02)
        req = _FakeRequest()
        chunks = [chunk async for chunk in broker.stream(req)]  # type: ignore[arg-type]
        assert chunks == [] or chunks == [": keepalive\n\n"]
        assert len(bus._subscribers) == 0

    asyncio.run(_run())


def test_events_endpoint_headers_keepalive_and_streaming(tmp_path: Path) -> None:
    app = _build_test_app(tmp_path)
    app.state.sse_broker = SSEBroker(app.state.event_bus, heartbeat_seconds=0.05)

    def _publish_after_keepalive() -> None:
        app.state.event_bus.publish(
            "job.completed",
            job_id="job-xyz",
            payload={"formats": ["vtt"]},
        )

    with TestClient(app) as client, client.stream("GET", "/api/v1/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"

        seen_keepalive = False
        event_lines: list[str] = []
        for line in resp.iter_lines():
            if line == ": keepalive":
                if not seen_keepalive:
                    seen_keepalive = True
                    _publish_after_keepalive()
                continue
            if not line:
                if len(event_lines) >= 3:
                    break
                continue
            event_lines.append(line)
            if len(event_lines) >= 3:
                break

        assert seen_keepalive is True
        assert event_lines[0] == "id: 1"
        assert event_lines[1] == "event: job.completed"
        assert event_lines[2].startswith("data: ")
        payload = json.loads(event_lines[2].removeprefix("data: "))
        assert payload["job_id"] == "job-xyz"
        assert payload["payload"] == {"formats": ["vtt"]}


def test_events_endpoint_via_testclient_sync_stream(tmp_path: Path) -> None:
    app = _build_test_app(tmp_path)
    app.state.sse_broker = SSEBroker(app.state.event_bus, heartbeat_seconds=0.05)

    def _publish_soon() -> None:
        app.state.event_bus.publish("device.status", payload={"device_id": "cuda:0"})

    with TestClient(app) as client, client.stream("GET", "/api/v1/events") as resp:
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"
        seen_event = False
        published = False
        for line in resp.iter_lines():
            if line == ": keepalive" and not published:
                published = True
                _publish_soon()
                continue
            if line.startswith("event: device.status"):
                seen_event = True
                break
        assert seen_event is True
