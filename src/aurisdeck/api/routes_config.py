"""REST endpoints for reusable configuration entities: profiles, glossaries, and replacements."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from aurisdeck.api.deps import ApiError, get_current_user, get_paths
from aurisdeck.domain.models import (
    GlossarySchema,
    ProfileSchema,
    ReplacementDictSchema,
    SlugStr,
    validate_slug,
)
from aurisdeck.storage.paths import DataPaths
from aurisdeck.storage.stores import (
    GlossaryStore,
    ProfileStore,
    ReplacementStore,
    _JsonEntityStore,
    import_glossary_text,
)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(get_current_user)])

_SLUGIFY_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class DeleteResult(BaseModel):
    """Confirmation payload returned when a configuration entity is deleted."""

    deleted: bool = True
    id: str


def _get_or_404[ModelT: (ProfileSchema, GlossarySchema, ReplacementDictSchema)](
    store: _JsonEntityStore[ModelT],
    entity_id: str,
    *,
    error_code: str,
    param_key: str,
    label: str,
) -> ModelT:
    entity = store.get(entity_id)
    if entity is None:
        raise ApiError(
            status_code=404,
            error_code=error_code,
            error_params={param_key: entity_id},
            detail=f"{label} {entity_id!r} not found",
        )
    return entity


def _create_entity[ModelT: (ProfileSchema, GlossarySchema, ReplacementDictSchema)](
    store: _JsonEntityStore[ModelT],
    entity: ModelT,
    *,
    error_code: str,
    param_key: str,
    label: str,
) -> ModelT:
    if store.exists(entity.id):
        raise ApiError(
            status_code=409,
            error_code=error_code,
            error_params={param_key: entity.id},
            detail=f"{label} {entity.id!r} already exists",
        )
    return store.save(entity)


def _update_entity[ModelT: (ProfileSchema, GlossarySchema, ReplacementDictSchema)](
    store: _JsonEntityStore[ModelT],
    entity_id: str,
    entity: ModelT,
    *,
    error_code: str,
    param_key: str,
    label: str,
) -> ModelT:
    if not store.exists(entity_id):
        raise ApiError(
            status_code=404,
            error_code=error_code,
            error_params={param_key: entity_id},
            detail=f"{label} {entity_id!r} not found",
        )
    normalized = entity.model_copy(update={"id": entity_id})
    return store.save(normalized)


def _delete_or_404[ModelT: (ProfileSchema, GlossarySchema, ReplacementDictSchema)](
    store: _JsonEntityStore[ModelT],
    entity_id: str,
    *,
    error_code: str,
    param_key: str,
    label: str,
) -> DeleteResult:
    if not store.delete(entity_id):
        raise ApiError(
            status_code=404,
            error_code=error_code,
            error_params={param_key: entity_id},
            detail=f"{label} {entity_id!r} not found",
        )
    return DeleteResult(deleted=True, id=entity_id)


def _resolve_import_glossary_id(
    glossary_id: str | None,
    filename: str | None,
    display_name: str | None,
) -> str:
    if glossary_id is not None and glossary_id.strip():
        candidate = glossary_id.strip()
        try:
            return validate_slug(candidate)
        except ValueError as exc:
            raise ApiError(
                status_code=422,
                error_code="VALIDATION_ERROR",
                error_params={"glossary_id": candidate},
                detail=str(exc),
            ) from exc

    source = Path(filename).stem if filename else (display_name or "")
    derived = _SLUGIFY_RE.sub("-", source.strip()).strip("-_")
    if not derived:
        raise ApiError(
            status_code=422,
            error_code="VALIDATION_ERROR",
            detail="glossary_id is required when it cannot be derived from filename or display_name",
        )
    return validate_slug(derived)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@router.get("/profiles", response_model=list[ProfileSchema])
async def list_profiles(
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> list[ProfileSchema]:
    return ProfileStore(paths).list()


@router.get("/profiles/{profile_id}", response_model=ProfileSchema)
async def get_profile(
    profile_id: SlugStr,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> ProfileSchema:
    return _get_or_404(
        ProfileStore(paths),
        profile_id,
        error_code="PROFILE_NOT_FOUND",
        param_key="profile_id",
        label="Profile",
    )


@router.post("/profiles", response_model=ProfileSchema, status_code=201)
async def create_profile(
    body: ProfileSchema,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> ProfileSchema:
    return _create_entity(
        ProfileStore(paths),
        body,
        error_code="PROFILE_ALREADY_EXISTS",
        param_key="profile_id",
        label="Profile",
    )


@router.put("/profiles/{profile_id}", response_model=ProfileSchema)
async def update_profile(
    profile_id: SlugStr,
    body: ProfileSchema,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> ProfileSchema:
    return _update_entity(
        ProfileStore(paths),
        profile_id,
        body,
        error_code="PROFILE_NOT_FOUND",
        param_key="profile_id",
        label="Profile",
    )


@router.delete("/profiles/{profile_id}", response_model=DeleteResult)
async def delete_profile(
    profile_id: SlugStr,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> DeleteResult:
    return _delete_or_404(
        ProfileStore(paths),
        profile_id,
        error_code="PROFILE_NOT_FOUND",
        param_key="profile_id",
        label="Profile",
    )


# ---------------------------------------------------------------------------
# Glossaries
# ---------------------------------------------------------------------------


@router.get("/glossaries", response_model=list[GlossarySchema])
async def list_glossaries(
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> list[GlossarySchema]:
    return GlossaryStore(paths).list()


@router.get("/glossaries/{glossary_id}", response_model=GlossarySchema)
async def get_glossary(
    glossary_id: SlugStr,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> GlossarySchema:
    return _get_or_404(
        GlossaryStore(paths),
        glossary_id,
        error_code="GLOSSARY_NOT_FOUND",
        param_key="glossary_id",
        label="Glossary",
    )


@router.post("/glossaries", response_model=GlossarySchema, status_code=201)
async def create_glossary(
    body: GlossarySchema,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> GlossarySchema:
    return _create_entity(
        GlossaryStore(paths),
        body,
        error_code="GLOSSARY_ALREADY_EXISTS",
        param_key="glossary_id",
        label="Glossary",
    )


@router.post("/glossaries/import", response_model=GlossarySchema)
async def import_glossary(
    paths: Annotated[DataPaths, Depends(get_paths)],
    user: Annotated[dict[str, Any], Depends(get_current_user)],
    file: Annotated[UploadFile | None, File()] = None,
    raw_text: Annotated[str | None, Form()] = None,
    glossary_id: Annotated[str | None, Form()] = None,
    display_name: Annotated[str | None, Form()] = None,
    name: Annotated[str | None, Form()] = None,
    ext: Annotated[str | None, Form()] = None,
    visibility: Annotated[Literal["shared", "private"] | None, Form()] = None,
) -> GlossarySchema:
    if file is None and raw_text is None:
        raise ApiError(
            status_code=400,
            error_code="VALIDATION_ERROR",
            detail="Either file or raw_text must be provided",
        )

    filename = file.filename if file is not None else None
    if file is not None:
        raw_bytes = await file.read()
        try:
            text_content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ApiError(
                status_code=400,
                error_code="VALIDATION_ERROR",
                detail="Uploaded glossary file must be valid UTF-8 text",
            ) from exc
        resolved_ext = ext.strip() if ext and ext.strip() else (Path(filename or "").suffix or "txt")
    else:
        text_content = raw_text or ""
        resolved_ext = ext.strip() if ext and ext.strip() else "txt"

    try:
        terms = import_glossary_text(text_content, resolved_ext)
    except ValueError as exc:
        raise ApiError(
            status_code=400,
            error_code="VALIDATION_ERROR",
            error_params={"ext": resolved_ext},
            detail=str(exc),
        ) from exc

    requested_name = (display_name or name or "").strip() or None
    resolved_id = _resolve_import_glossary_id(glossary_id, filename, requested_name)

    store = GlossaryStore(paths)
    existing = store.get(resolved_id)
    resolved_name = (
        requested_name
        or (existing.name if existing is not None else None)
        or (Path(filename).stem.strip() if filename else None)
        or resolved_id
    )
    resolved_owner = (
        existing.owner_user_id if existing is not None else str(user.get("id") or "default")
    )
    resolved_visibility: Literal["shared", "private"] = (
        visibility if visibility is not None else (existing.visibility if existing else "shared")
    )

    glossary = GlossarySchema(
        id=resolved_id,
        name=resolved_name,
        owner_user_id=resolved_owner,
        visibility=resolved_visibility,
        terms=terms,
    )
    return store.save(glossary)


@router.put("/glossaries/{glossary_id}", response_model=GlossarySchema)
async def update_glossary(
    glossary_id: SlugStr,
    body: GlossarySchema,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> GlossarySchema:
    return _update_entity(
        GlossaryStore(paths),
        glossary_id,
        body,
        error_code="GLOSSARY_NOT_FOUND",
        param_key="glossary_id",
        label="Glossary",
    )


@router.delete("/glossaries/{glossary_id}", response_model=DeleteResult)
async def delete_glossary(
    glossary_id: SlugStr,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> DeleteResult:
    return _delete_or_404(
        GlossaryStore(paths),
        glossary_id,
        error_code="GLOSSARY_NOT_FOUND",
        param_key="glossary_id",
        label="Glossary",
    )


# ---------------------------------------------------------------------------
# Replacement dictionaries
# ---------------------------------------------------------------------------


@router.get("/replacements", response_model=list[ReplacementDictSchema])
async def list_replacements(
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> list[ReplacementDictSchema]:
    return ReplacementStore(paths).list()


@router.get("/replacements/{dict_id}", response_model=ReplacementDictSchema)
async def get_replacement(
    dict_id: SlugStr,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> ReplacementDictSchema:
    return _get_or_404(
        ReplacementStore(paths),
        dict_id,
        error_code="REPLACEMENT_NOT_FOUND",
        param_key="dict_id",
        label="Replacement dictionary",
    )


@router.post("/replacements", response_model=ReplacementDictSchema, status_code=201)
async def create_replacement(
    body: ReplacementDictSchema,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> ReplacementDictSchema:
    return _create_entity(
        ReplacementStore(paths),
        body,
        error_code="REPLACEMENT_ALREADY_EXISTS",
        param_key="dict_id",
        label="Replacement dictionary",
    )


@router.put("/replacements/{dict_id}", response_model=ReplacementDictSchema)
async def update_replacement(
    dict_id: SlugStr,
    body: ReplacementDictSchema,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> ReplacementDictSchema:
    return _update_entity(
        ReplacementStore(paths),
        dict_id,
        body,
        error_code="REPLACEMENT_NOT_FOUND",
        param_key="dict_id",
        label="Replacement dictionary",
    )


@router.delete("/replacements/{dict_id}", response_model=DeleteResult)
async def delete_replacement(
    dict_id: SlugStr,
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> DeleteResult:
    return _delete_or_404(
        ReplacementStore(paths),
        dict_id,
        error_code="REPLACEMENT_NOT_FOUND",
        param_key="dict_id",
        label="Replacement dictionary",
    )
