"""Disk-backed JSON stores for profiles, glossaries, replacements, and ContextBuilder."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from aurisdeck.domain.models import (
    EffectiveContext,
    GlossarySchema,
    ProfileSchema,
    ReplacementDictSchema,
    migrate_profile,
    validate_slug,
)
from aurisdeck.storage.paths import DataPaths, atomic_write_text, sanitize_filename

_MAX_PROMPT_CHARS = 850


class _JsonEntity(Protocol):
    """Structural contract for persisted domain models with a slug id."""

    id: str

    def model_dump_json(self, *, indent: int | None = None) -> str: ...


class _JsonEntityStore[ModelT: _JsonEntity]:
    """Generic atomic JSON store backed by a DataPaths subdirectory."""

    def __init__(self, paths: DataPaths, directory: Path) -> None:
        self._paths = paths
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    def _path_for(self, entity_id: str) -> Path:
        slug = validate_slug(entity_id)
        safe_stem = sanitize_filename(slug)
        return self._directory / f"{safe_stem}.json"

    def _deserialize(self, raw: dict[str, Any]) -> ModelT:
        raise NotImplementedError

    def is_empty(self) -> bool:
        """Return True if no JSON entities exist in this store directory."""
        return not self._directory.exists() or not any(self._directory.glob("*.json"))

    def exists(self, entity_id: str) -> bool:
        """Return True if an entity with the given slug exists on disk."""
        return self._path_for(entity_id).is_file()

    def list(self) -> list[ModelT]:
        """Load and validate all stored entities sorted by filename."""
        if not self._directory.exists():
            return []
        items: list[ModelT] = []
        for path in sorted(self._directory.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            items.append(self._deserialize(raw))
        return items

    def get(self, entity_id: str) -> ModelT | None:
        """Load and validate an entity by id, or return None if missing."""
        path = self._path_for(entity_id)
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return self._deserialize(raw)

    def save(self, entity: ModelT) -> ModelT:
        """Persist an entity atomically using temp staging + os.replace."""
        path = self._path_for(entity.id)
        payload = entity.model_dump_json(indent=2) + "\n"
        atomic_write_text(path, payload, self._paths.tmp)
        return entity

    def delete(self, entity_id: str) -> bool:
        """Delete an entity by id. Returns True if a file was removed."""
        path = self._path_for(entity_id)
        if not path.is_file():
            return False
        path.unlink()
        return True


class ProfileStore(_JsonEntityStore[ProfileSchema]):
    """Persistent JSON store for transcription profiles."""

    def __init__(self, paths: DataPaths) -> None:
        super().__init__(paths, paths.profiles)

    def _deserialize(self, raw: dict[str, Any]) -> ProfileSchema:
        return migrate_profile(raw)


class GlossaryStore(_JsonEntityStore[GlossarySchema]):
    """Persistent JSON store for shared glossaries."""

    def __init__(self, paths: DataPaths) -> None:
        super().__init__(paths, paths.glossaries)

    def _deserialize(self, raw: dict[str, Any]) -> GlossarySchema:
        return GlossarySchema.model_validate(raw)


class ReplacementStore(_JsonEntityStore[ReplacementDictSchema]):
    """Persistent JSON store for post-ASR replacement dictionaries."""

    def __init__(self, paths: DataPaths) -> None:
        super().__init__(paths, paths.replacements)

    def _deserialize(self, raw: dict[str, Any]) -> ReplacementDictSchema:
        return ReplacementDictSchema.model_validate(raw)


def _dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        cleaned = raw.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def import_glossary_text(raw_text: str, ext: str) -> list[str]:
    """Parse glossary terms from .txt (one per line) or .csv (first column)."""
    normalized_ext = ext.strip().lower().lstrip(".")
    if normalized_ext not in {"txt", "csv"}:
        raise ValueError(f"Unsupported glossary extension: {ext!r}")

    cleaned_text = (raw_text or "").lstrip("\ufeff")
    candidates: list[str] = []

    if normalized_ext == "txt":
        for line in cleaned_text.splitlines():
            term = line.strip()
            if term and not term.startswith("#"):
                candidates.append(term)
    else:
        reader = csv.reader(io.StringIO(cleaned_text))
        for row in reader:
            if not row:
                continue
            term = row[0].strip()
            if term and not term.startswith("#"):
                candidates.append(term)

    return _dedupe_preserving_order(candidates)


def _truncate_prompt_safely(prompt: str, max_chars: int) -> str:
    """Truncate prompt at or below max_chars without splitting a word in half."""
    if max_chars <= 0 or not prompt:
        return ""
    if len(prompt) <= max_chars:
        return prompt
    candidate = prompt[:max_chars]
    if prompt[max_chars].isspace():
        return candidate.rstrip()
    last_ws = max(candidate.rfind(" "), candidate.rfind("\n"), candidate.rfind("\t"))
    if last_ws > 0:
        return candidate[:last_ws].rstrip()
    return candidate


def build_effective_context(
    profile: ProfileSchema,
    glossary_store: GlossaryStore | None = None,
    replacement_store: ReplacementStore | None = None,
    extra_hotwords: Iterable[str] | None = None,
    extra_prompt: str | None = None,
    *,
    max_prompt_chars: int = _MAX_PROMPT_CHARS,
) -> EffectiveContext:
    """Build deduplicated hotwords, word-safe prompt, and merged replacements for a profile."""
    raw_hotwords: list[str] = list(profile.context.hotwords)

    if glossary_store is not None:
        for glossary_id in profile.context.glossary_ids:
            glossary = glossary_store.get(glossary_id)
            if glossary is not None:
                raw_hotwords.extend(glossary.terms)

    if extra_hotwords is not None:
        if isinstance(extra_hotwords, str):
            raw_hotwords.append(extra_hotwords)
        else:
            raw_hotwords.extend(extra_hotwords)

    hotwords = _dedupe_preserving_order(raw_hotwords)

    prompt_parts = [
        part.strip()
        for part in (profile.context.initial_prompt, extra_prompt)
        if isinstance(part, str) and part.strip()
    ]
    combined_prompt = " ".join(prompt_parts)
    initial_prompt = _truncate_prompt_safely(combined_prompt, max_prompt_chars)

    replacements: dict[str, str] = {}
    case_sensitive = False
    if profile.context.apply_replacements:
        case_sensitive_flags: list[bool] = []
        if replacement_store is not None:
            for dict_id in profile.context.replacement_ids:
                replacement_dict = replacement_store.get(dict_id)
                if replacement_dict is not None:
                    replacements.update(replacement_dict.rules)
                    case_sensitive_flags.append(replacement_dict.case_sensitive)
        replacements.update(profile.context.inline_replacements)
        case_sensitive = any(case_sensitive_flags)

    return EffectiveContext(
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        replacements=replacements,
        case_sensitive=case_sensitive,
        apply_replacements=profile.context.apply_replacements,
    )
