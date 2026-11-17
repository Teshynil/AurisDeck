"""Extensible JSON locale loader with built-in defaults and /data/locales overrides."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

_LOCALE_CODE_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_locale_code(code: str) -> str:
    """Validate that a locale code is safe against path traversal."""
    if not isinstance(code, str) or not _LOCALE_CODE_RE.fullmatch(code):
        raise ValueError(f"Invalid locale code: {code!r}")
    return code


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override onto base without mutating inputs."""
    merged: dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_safe_json_path(directory: Path, code: str) -> Path:
    """Resolve `directory / f'{code}.json'` ensuring it stays within directory."""
    resolved_dir = directory.resolve()
    candidate = (resolved_dir / f"{code}.json").resolve()
    if candidate.parent != resolved_dir:
        raise ValueError(f"Locale path escapes base directory: {code!r}")
    return candidate


def _load_json_dict(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Locale file must contain a JSON object: {path}")
    return data


def _extract_meta(data: dict[str, Any], code: str) -> dict[str, Any]:
    raw_meta = data.get("_meta")
    meta: dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    meta.setdefault("code", code)
    meta.setdefault("name", code)
    meta.setdefault("direction", "ltr")
    return meta


class LocaleLoader:
    """Loads built-in locales and deep-merges persistent overrides from data/locales."""

    def __init__(self, builtin_dir: Path | str, custom_dir: Path | str) -> None:
        self.builtin_dir = Path(builtin_dir)
        self.custom_dir = Path(custom_dir)

    def _discover_codes(self) -> list[str]:
        codes: set[str] = set()
        for directory in (self.builtin_dir, self.custom_dir):
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                if path.is_file() and _LOCALE_CODE_RE.fullmatch(path.stem):
                    codes.add(path.stem)
        return sorted(codes)

    def list_locales(self) -> list[dict[str, Any]]:
        """List `_meta` blocks for all built-in and custom locales."""
        locales: list[dict[str, Any]] = []
        for code in self._discover_codes():
            builtin_path = _resolve_safe_json_path(self.builtin_dir, code)
            custom_path = _resolve_safe_json_path(self.custom_dir, code)

            meta: dict[str, Any] = {"code": code, "name": code, "direction": "ltr"}
            if builtin_path.is_file():
                meta = deep_merge(meta, _extract_meta(_load_json_dict(builtin_path), code))
            if custom_path.is_file():
                meta = deep_merge(meta, _extract_meta(_load_json_dict(custom_path), code))
            locales.append(meta)
        return locales

    def get_locale(self, code: str) -> dict[str, Any]:
        """Load and deep-merge a locale dictionary by code."""
        valid_code = validate_locale_code(code)
        builtin_path = _resolve_safe_json_path(self.builtin_dir, valid_code)
        custom_path = _resolve_safe_json_path(self.custom_dir, valid_code)

        has_builtin = builtin_path.is_file()
        has_custom = custom_path.is_file()
        if not has_builtin and not has_custom:
            raise FileNotFoundError(f"Locale not found: {valid_code}")

        base = _load_json_dict(builtin_path) if has_builtin else {}
        if has_custom:
            override = _load_json_dict(custom_path)
            merged = deep_merge(base, override)
        else:
            merged = base

        merged["_meta"] = _extract_meta(merged, valid_code)
        return merged
