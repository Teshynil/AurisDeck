"""Offline web asset serving, CSP/CDN hygiene, and API route coexistence tests."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from aurisdeck.api import create_app
from aurisdeck.config import Settings

STATIC_DIR = Path(__file__).resolve().parent.parent / "src" / "aurisdeck" / "web" / "static"

EXPECTED_STATIC_ASSETS = (
    "css/app.css",
    "js/store.js",
    "js/i18n.js",
    "js/api.js",
    "js/sse.js",
)

FORBIDDEN_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bnew\s+Function\s*\("),
)


def test_root_serves_index_html_and_static_assets_offline(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        root_resp = client.get("/")
        assert root_resp.status_code == 200
        assert "text/html" in root_resp.headers.get("content-type", "")
        html = root_resp.text

        assert '<meta name="base-path"' in html
        assert 'href="./css/app.css"' in html
        assert 'src="./js/app.js"' in html
        assert 'id="login-modal"' in html
        assert 'id="toast-container"' in html
        assert 'accept=".m4a,' in html

        for view_id in (
            "view-transcribe",
            "view-library",
            "view-viewer",
            "view-profiles",
            "view-settings",
        ):
            assert f'id="{view_id}"' in html

        for rel_path in EXPECTED_STATIC_ASSETS:
            asset_resp = client.get(f"/{rel_path}")
            assert asset_resp.status_code == 200, f"Expected 200 for /{rel_path}"
            assert len(asset_resp.text.strip()) > 0
            if rel_path.endswith(".css"):
                assert "text/css" in asset_resp.headers.get("content-type", "")
            elif rel_path.endswith(".js"):
                content_type = asset_resp.headers.get("content-type", "")
                assert "javascript" in content_type

        # StaticFiles mount at "/" must not shadow /healthz or /api/v1/* endpoints
        health_resp = client.get("/healthz")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        auth_resp = client.get("/api/v1/auth/status")
        assert auth_resp.status_code == 200
        assert auth_resp.json() == {"auth_mode": "none", "authenticated": True}

        locales_resp = client.get("/api/v1/locales")
        assert locales_resp.status_code == 200
        codes = {item["code"] for item in locales_resp.json()["locales"]}
        assert {"en", "es"}.issubset(codes)

        me_resp = client.get("/api/v1/me")
        assert me_resp.status_code == 200
        assert me_resp.json()["id"] == "default"

        jobs_resp = client.get("/api/v1/jobs")
        assert jobs_resp.status_code == 200
        assert "jobs" in jobs_resp.json()


def test_web_static_files_have_no_external_cdns_or_unsafe_eval() -> None:
    assert STATIC_DIR.is_dir()
    files = sorted(path for path in STATIC_DIR.rglob("*") if path.is_file())
    assert files, "No static files found in web/static"

    for path in files:
        content = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            match = pattern.search(content)
            assert match is None, f"Forbidden pattern {pattern.pattern!r} found in {path}"
        assert "innerHTML" not in content, f"innerHTML found in {path}"
