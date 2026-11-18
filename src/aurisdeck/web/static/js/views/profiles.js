/**
 * Advanced configuration view: JSON Profiles, Shared Glossaries, and Post-ASR Replacements.
 */

import { api } from "../api.js";
import { t, translateError } from "../i18n.js";
import { store } from "../store.js";

const SLUG_RE = /^[a-zA-Z0-9_-]+$/;
const ASR_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"];
const ASR_LANGUAGES = ["es", "en", "auto"];
const COMPUTE_TYPES = ["float16", "int8_float16", "int8"];
const INPUT_MODES = ["single", "multitrack", "batch"];
const OUTPUT_FORMATS = ["vtt", "srt", "md", "json"];
const VOICE_STYLES = ["v_tag", "prefix", "none"];
const TOKEN_LIMIT = 224;

const containerCleanups = new WeakMap();

function el(tag, attrs = null, ...children) {
  const node = document.createElement(tag);
  if (attrs && typeof attrs === "object") {
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null || value === false) {
        continue;
      }
      if (key === "className") {
        node.className = String(value);
      } else if (key === "textContent") {
        node.textContent = String(value);
      } else if (key === "value") {
        node.value = String(value);
      } else if (key === "checked") {
        node.checked = Boolean(value);
      } else if (key === "disabled") {
        node.disabled = Boolean(value);
      } else if (key === "readOnly") {
        node.readOnly = Boolean(value);
      } else if (key === "required") {
        node.required = Boolean(value);
      } else if (key === "dataset" && typeof value === "object") {
        for (const [dKey, dVal] of Object.entries(value)) {
          if (dVal !== undefined && dVal !== null) {
            node.dataset[dKey] = String(dVal);
          }
        }
      } else if (key.startsWith("on") && typeof value === "function") {
        node.addEventListener(key.slice(2).toLowerCase(), value);
      } else if (value === true) {
        node.setAttribute(key, "");
      } else {
        node.setAttribute(key, String(value));
      }
    }
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) {
      continue;
    }
    if (typeof child === "string" || typeof child === "number") {
      node.appendChild(document.createTextNode(String(child)));
    } else {
      node.appendChild(child);
    }
  }
  return node;
}

function estimatePromptTokens(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) {
    return 0;
  }
  const words = trimmed.split(/\s+/).length;
  const charEstimate = Math.ceil(trimmed.length / 4);
  return Math.max(words, charEstimate);
}

function createEmptyProfile(ownerId = "default") {
  return {
    schema_version: 1,
    id: "",
    owner_user_id: ownerId || "default",
    visibility: "shared",
    display_name: "",
    simple_visible: true,
    input: { mode: "single" },
    preprocessing: { normalize_audio: false, generate_mix: false },
    asr: {
      backend: "faster-whisper",
      model: "large-v3-turbo",
      language: "es",
      compute_type: "float16",
      beam_size: 5,
      vad_filter: true,
      word_timestamps: true,
    },
    diarization: {
      enabled: false,
      backend: "pyannote-3.1",
      min_speakers: null,
      max_speakers: null,
    },
    context: {
      initial_prompt: "",
      hotwords: [],
      glossary_ids: [],
      replacement_ids: [],
      inline_replacements: {},
      apply_replacements: true,
    },
    hardware: {
      gpus: "auto",
      cpu_fallback: true,
      cpu_compute_type: "int8",
      cpu_threads: 0,
    },
    output: {
      subtitles: {
        voice_style: "v_tag",
        max_cue_seconds: 6.0,
        max_chars_per_line: 68,
        max_lines_per_cue: 2,
      },
    },
  };
}

function normalizeProfile(raw, fallbackOwner = "default") {
  const base = createEmptyProfile(fallbackOwner);
  const src = raw && typeof raw === "object" ? raw : {};
  const input = src.input && typeof src.input === "object" ? src.input : {};
  const pre = src.preprocessing && typeof src.preprocessing === "object" ? src.preprocessing : {};
  const asr = src.asr && typeof src.asr === "object" ? src.asr : {};
  const diar = src.diarization && typeof src.diarization === "object" ? src.diarization : {};
  const ctx = src.context && typeof src.context === "object" ? src.context : {};
  const hw = src.hardware && typeof src.hardware === "object" ? src.hardware : {};
  const out = src.output && typeof src.output === "object" ? src.output : {};
  const subs = out.subtitles && typeof out.subtitles === "object" ? out.subtitles : {};

  let normalizedGpus = "auto";
  if (hw.gpus === "cpu" || hw.gpus === "auto") {
    normalizedGpus = hw.gpus;
  } else if (Array.isArray(hw.gpus)) {
    const indices = hw.gpus
      .map((val) => Number.parseInt(val, 10))
      .filter((num) => Number.isInteger(num) && num >= 0);
    normalizedGpus = indices.length > 0 ? Array.from(new Set(indices)) : "auto";
  }

  const rawFormats = Array.isArray(out.formats)
    ? out.formats.filter((fmt) => OUTPUT_FORMATS.includes(fmt))
    : [];

  return {
    schema_version: 1,
    id: typeof src.id === "string" ? src.id : "",
    owner_user_id: typeof src.owner_user_id === "string" && src.owner_user_id ? src.owner_user_id : base.owner_user_id,
    visibility: src.visibility === "private" ? "private" : "shared",
    display_name: typeof src.display_name === "string" ? src.display_name : "",
    simple_visible: src.simple_visible !== undefined ? Boolean(src.simple_visible) : true,
    input: {
      mode: INPUT_MODES.includes(input.mode) ? input.mode : "single",
    },
    preprocessing: {
      normalize_audio: Boolean(pre.normalize_audio),
      generate_mix: Boolean(pre.generate_mix),
    },
    asr: {
      backend: typeof asr.backend === "string" && asr.backend ? asr.backend : "faster-whisper",
      model: typeof asr.model === "string" && asr.model ? asr.model : "large-v3-turbo",
      language: asr.language === null || asr.language === undefined || asr.language === "" || asr.language === "auto"
        ? null
        : String(asr.language),
      compute_type: typeof asr.compute_type === "string" && asr.compute_type ? asr.compute_type : "float16",
      beam_size: Number.isFinite(Number(asr.beam_size)) && Number(asr.beam_size) >= 1 ? Math.round(Number(asr.beam_size)) : 5,
      vad_filter: asr.vad_filter !== undefined ? Boolean(asr.vad_filter) : true,
      word_timestamps: asr.word_timestamps !== undefined ? Boolean(asr.word_timestamps) : true,
    },
    diarization: {
      enabled: Boolean(diar.enabled),
      backend: typeof diar.backend === "string" && diar.backend ? diar.backend : "pyannote-3.1",
      min_speakers: diar.min_speakers !== null && diar.min_speakers !== undefined && diar.min_speakers !== ""
        ? Number(diar.min_speakers)
        : null,
      max_speakers: diar.max_speakers !== null && diar.max_speakers !== undefined && diar.max_speakers !== ""
        ? Number(diar.max_speakers)
        : null,
    },
    context: {
      initial_prompt: typeof ctx.initial_prompt === "string" ? ctx.initial_prompt : "",
      hotwords: Array.isArray(ctx.hotwords) ? ctx.hotwords.map((item) => String(item)).filter(Boolean) : [],
      glossary_ids: Array.isArray(ctx.glossary_ids) ? ctx.glossary_ids.map((item) => String(item)).filter(Boolean) : [],
      replacement_ids: Array.isArray(ctx.replacement_ids) ? ctx.replacement_ids.map((item) => String(item)).filter(Boolean) : [],
      inline_replacements:
        ctx.inline_replacements && typeof ctx.inline_replacements === "object" && !Array.isArray(ctx.inline_replacements)
          ? { ...ctx.inline_replacements }
          : {},
      apply_replacements: ctx.apply_replacements !== undefined ? Boolean(ctx.apply_replacements) : true,
    },
    hardware: {
      gpus: normalizedGpus,
      cpu_fallback: hw.cpu_fallback !== undefined ? Boolean(hw.cpu_fallback) : true,
      cpu_compute_type: typeof hw.cpu_compute_type === "string" && hw.cpu_compute_type ? hw.cpu_compute_type : "int8",
      cpu_threads: Number.isFinite(Number(hw.cpu_threads)) && Number(hw.cpu_threads) >= 0 ? Math.round(Number(hw.cpu_threads)) : 0,
    },
    output: {
      subtitles: {
        voice_style: VOICE_STYLES.includes(subs.voice_style) ? subs.voice_style : "v_tag",
        max_cue_seconds: Number.isFinite(Number(subs.max_cue_seconds)) && Number(subs.max_cue_seconds) > 0
          ? Number(subs.max_cue_seconds)
          : 6.0,
        max_chars_per_line: Number.isFinite(Number(subs.max_chars_per_line)) && Number(subs.max_chars_per_line) >= 1
          ? Math.round(Number(subs.max_chars_per_line))
          : 68,
        max_lines_per_cue: Number.isFinite(Number(subs.max_lines_per_cue)) && Number(subs.max_lines_per_cue) >= 1
          ? Math.round(Number(subs.max_lines_per_cue))
          : 2,
      },
    },
  };
}

function pairsObjectToRows(obj) {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) {
    return [];
  }
  return Object.entries(obj).map(([source, target]) => ({
    source: String(source),
    target: String(target ?? ""),
  }));
}

function rowsToPairsObject(rows) {
  const result = {};
  for (const row of rows) {
    const src = String(row.source || "").trim();
    const tgt = String(row.target ?? "");
    if (!src && !tgt.trim()) {
      continue;
    }
    if (!src) {
      return { error: "empty_source", data: null };
    }
    result[row.source] = tgt;
  }
  return { error: null, data: result };
}

function makeUniqueSlug(baseId, existingIds) {
  const cleanBase = String(baseId || "profile").replace(/[^a-zA-Z0-9_-]+/g, "-") || "profile";
  const taken = new Set(existingIds);
  let candidate = `${cleanBase}-copy`;
  let idx = 2;
  while (taken.has(candidate)) {
    candidate = `${cleanBase}-copy-${idx}`;
    idx += 1;
  }
  return candidate;
}

function createEmptyGlossary(ownerId = "default") {
  return {
    id: "",
    name: "",
    owner_user_id: ownerId || "default",
    visibility: "shared",
    termsText: "",
  };
}

function createEmptyReplacement(ownerId = "default") {
  return {
    id: "",
    name: "",
    owner_user_id: ownerId || "default",
    visibility: "shared",
    case_sensitive: false,
    rows: [],
  };
}

function localizedProfileName(profile) {
  return profile?.display_name || profile?.id || "";
}

/**
 * Mount the Advanced configuration view into `container` (#view-advanced).
 * @param {HTMLElement} container
 */
export function mount(container) {
  if (!container) {
    return () => {};
  }

  const prevCleanup = containerCleanups.get(container);
  if (typeof prevCleanup === "function") {
    prevCleanup();
  }

  const initial = store.getState();
  const initialUserId = initial.user?.id || "default";
  const initialProfiles = Array.isArray(initial.profiles) ? initial.profiles : [];
  const initialGlossaries = Array.isArray(initial.glossaries) ? initial.glossaries : [];
  const initialReplacements = Array.isArray(initial.replacements) ? initial.replacements : [];

  const local = {
    subTab: "profiles",
    selectedProfileId: initialProfiles[0]?.id || "",
    isNewProfile: initialProfiles.length === 0,
    draftProfile: initialProfiles[0]
      ? normalizeProfile(initialProfiles[0], initialUserId)
      : createEmptyProfile(initialUserId),
    inlineRows: initialProfiles[0]
      ? pairsObjectToRows(initialProfiles[0].context?.inline_replacements)
      : [],
    rawJsonMode: false,
    rawJsonText: initialProfiles[0]
      ? JSON.stringify(normalizeProfile(initialProfiles[0], initialUserId), null, 2)
      : JSON.stringify(createEmptyProfile(initialUserId), null, 2),
    profileError: "",
    savingProfile: false,

    selectedGlossaryId: initialGlossaries[0]?.id || "",
    isNewGlossary: initialGlossaries.length === 0,
    draftGlossary: initialGlossaries[0]
      ? {
          id: initialGlossaries[0].id,
          name: initialGlossaries[0].name || initialGlossaries[0].display_name || "",
          owner_user_id: initialGlossaries[0].owner_user_id || initialUserId,
          visibility: initialGlossaries[0].visibility || "shared",
          termsText: Array.isArray(initialGlossaries[0].terms) ? initialGlossaries[0].terms.join("\n") : "",
        }
      : createEmptyGlossary(initialUserId),
    glossaryError: "",
    savingGlossary: false,
    importingGlossary: false,

    selectedReplacementId: initialReplacements[0]?.id || "",
    isNewReplacement: initialReplacements.length === 0,
    draftReplacement: initialReplacements[0]
      ? {
          id: initialReplacements[0].id,
          name: initialReplacements[0].name || initialReplacements[0].display_name || "",
          owner_user_id: initialReplacements[0].owner_user_id || initialUserId,
          visibility: initialReplacements[0].visibility || "shared",
          case_sensitive: Boolean(initialReplacements[0].case_sensitive),
          rows: pairsObjectToRows(initialReplacements[0].rules),
        }
      : createEmptyReplacement(initialUserId),
    replacementError: "",
    savingReplacement: false,

    hasAttemptedInitialFetch: false,
    fetchingResources: false,
  };

  let gpuSectionEl = null;

  function loadProfileIntoDraft(profile) {
    const ownerId = store.getState().user?.id || "default";
    if (!profile) {
      local.isNewProfile = true;
      local.selectedProfileId = "";
      local.draftProfile = createEmptyProfile(ownerId);
      local.inlineRows = [];
      local.rawJsonText = JSON.stringify(local.draftProfile, null, 2);
      local.profileError = "";
      return;
    }
    local.isNewProfile = false;
    local.selectedProfileId = profile.id;
    local.draftProfile = normalizeProfile(profile, ownerId);
    local.inlineRows = pairsObjectToRows(local.draftProfile.context.inline_replacements);
    local.rawJsonText = JSON.stringify(local.draftProfile, null, 2);
    local.profileError = "";
  }

  function loadGlossaryIntoDraft(glossary) {
    const ownerId = store.getState().user?.id || "default";
    if (!glossary) {
      local.isNewGlossary = true;
      local.selectedGlossaryId = "";
      local.draftGlossary = createEmptyGlossary(ownerId);
      local.glossaryError = "";
      return;
    }
    local.isNewGlossary = false;
    local.selectedGlossaryId = glossary.id;
    local.draftGlossary = {
      id: glossary.id,
      name: glossary.name || glossary.display_name || "",
      owner_user_id: glossary.owner_user_id || ownerId,
      visibility: glossary.visibility || "shared",
      termsText: Array.isArray(glossary.terms) ? glossary.terms.join("\n") : "",
    };
    local.glossaryError = "";
  }

  function loadReplacementIntoDraft(dict) {
    const ownerId = store.getState().user?.id || "default";
    if (!dict) {
      local.isNewReplacement = true;
      local.selectedReplacementId = "";
      local.draftReplacement = createEmptyReplacement(ownerId);
      local.replacementError = "";
      return;
    }
    local.isNewReplacement = false;
    local.selectedReplacementId = dict.id;
    local.draftReplacement = {
      id: dict.id,
      name: dict.name || dict.display_name || "",
      owner_user_id: dict.owner_user_id || ownerId,
      visibility: dict.visibility || "shared",
      case_sensitive: Boolean(dict.case_sensitive),
      rows: pairsObjectToRows(dict.rules),
    };
    local.replacementError = "";
  }

  async function ensureConfigResourcesLoaded() {
    const state = store.getState();
    if (
      state.authStatus?.auth_mode === "password" &&
      !state.authStatus?.authenticated
    ) {
      return;
    }
    if (local.hasAttemptedInitialFetch || local.fetchingResources) {
      return;
    }
    const needsProfiles = !Array.isArray(state.profiles) || state.profiles.length === 0;
    const needsGlossaries = !Array.isArray(state.glossaries) || state.glossaries.length === 0;
    const needsReplacements = !Array.isArray(state.replacements) || state.replacements.length === 0;
    if (!needsProfiles && !needsGlossaries && !needsReplacements) {
      local.hasAttemptedInitialFetch = true;
      return;
    }

    local.hasAttemptedInitialFetch = true;
    local.fetchingResources = true;
    try {
      const [profiles, glossaries, replacements] = await Promise.all([
        needsProfiles ? api.getProfiles() : Promise.resolve(state.profiles),
        needsGlossaries ? api.getGlossaries() : Promise.resolve(state.glossaries),
        needsReplacements ? api.getReplacements() : Promise.resolve(state.replacements),
      ]);
      const patch = {};
      if (Array.isArray(profiles) && profiles.length > 0 && needsProfiles) {
        patch.profiles = profiles;
      }
      if (Array.isArray(glossaries) && glossaries.length > 0 && needsGlossaries) {
        patch.glossaries = glossaries;
      }
      if (Array.isArray(replacements) && replacements.length > 0 && needsReplacements) {
        patch.replacements = replacements;
      }
      if (Object.keys(patch).length > 0) {
        store.setState(patch);
      }
    } catch {
      // Errors will be surfaced on explicit user actions or auth handler.
    } finally {
      local.fetchingResources = false;
    }
  }

  function buildProfilePayloadFromForm() {
    const parsedRows = rowsToPairsObject(local.inlineRows);
    if (parsedRows.error === "empty_source") {
      return {
        error: t("errors.VALIDATION_ERROR"),
        payload: null,
      };
    }

    const p = local.draftProfile;
    const id = String(p.id || "").trim();
    const displayName = String(p.display_name || "").trim();
    if (!id || !SLUG_RE.test(id)) {
      return {
        error: `${t("errors.VALIDATION_ERROR")} (id: [a-zA-Z0-9_-]+)`,
        payload: null,
      };
    }
    if (!displayName) {
      return {
        error: `${t("errors.VALIDATION_ERROR")} (${t("advanced.profiles.display_name")})`,
        payload: null,
      };
    }

    if (
      p.diarization.min_speakers !== null &&
      p.diarization.max_speakers !== null &&
      p.diarization.max_speakers < p.diarization.min_speakers
    ) {
      return {
        error: t("errors.VALIDATION_ERROR"),
        payload: null,
      };
    }

    if (Array.isArray(p.hardware.gpus) && p.hardware.gpus.length === 0) {
      return {
        error: `${t("errors.VALIDATION_ERROR")} (${t("advanced.profiles.hardware_gpus")})`,
        payload: null,
      };
    }

    const payload = {
      schema_version: 1,
      id,
      owner_user_id: p.owner_user_id || store.getState().user?.id || "default",
      visibility: p.visibility === "private" ? "private" : "shared",
      display_name: displayName,
      simple_visible: Boolean(p.simple_visible),
      input: { mode: p.input.mode },
      preprocessing: {
        normalize_audio: Boolean(p.preprocessing.normalize_audio),
        generate_mix: p.input.mode === "multitrack" ? Boolean(p.preprocessing.generate_mix) : false,
      },
      asr: {
        backend: p.asr.backend || "faster-whisper",
        model: p.asr.model || "large-v3-turbo",
        language: p.asr.language === "auto" || p.asr.language === "" ? null : p.asr.language,
        compute_type: p.asr.compute_type || "float16",
        beam_size: Math.max(1, Number.parseInt(p.asr.beam_size, 10) || 5),
        vad_filter: Boolean(p.asr.vad_filter),
        word_timestamps: Boolean(p.asr.word_timestamps),
      },
      diarization: {
        enabled: Boolean(p.diarization.enabled),
        backend: p.diarization.backend || "pyannote-3.1",
        min_speakers: p.diarization.min_speakers,
        max_speakers: p.diarization.max_speakers,
      },
      context: {
        initial_prompt: String(p.context.initial_prompt || ""),
        hotwords: Array.isArray(p.context.hotwords) ? [...p.context.hotwords] : [],
        glossary_ids: Array.isArray(p.context.glossary_ids) ? [...p.context.glossary_ids] : [],
        replacement_ids: Array.isArray(p.context.replacement_ids) ? [...p.context.replacement_ids] : [],
        inline_replacements: parsedRows.data,
        apply_replacements: Boolean(p.context.apply_replacements),
      },
      hardware: {
        gpus: Array.isArray(p.hardware.gpus) ? [...p.hardware.gpus] : p.hardware.gpus,
        cpu_fallback: Boolean(p.hardware.cpu_fallback),
        cpu_compute_type: p.hardware.cpu_compute_type || "int8",
        cpu_threads: Math.max(0, Number.parseInt(p.hardware.cpu_threads, 10) || 0),
      },
      output: {
        subtitles: {
          voice_style: p.output.subtitles.voice_style || "v_tag",
          max_cue_seconds: Math.max(0.5, Number(p.output.subtitles.max_cue_seconds) || 6.0),
          max_chars_per_line: Math.max(1, Number.parseInt(p.output.subtitles.max_chars_per_line, 10) || 68),
          max_lines_per_cue: Math.max(1, Number.parseInt(p.output.subtitles.max_lines_per_cue, 10) || 2),
        },
      },
    };
    return { error: null, payload };
  }

  function parseRawJsonOrError() {
    let parsed;
    try {
      parsed = JSON.parse(local.rawJsonText);
    } catch (err) {
      return {
        error: `${t("advanced.profiles.invalid_json")} ${err.message || ""}`.trim(),
        data: null,
      };
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {
        error: t("advanced.profiles.invalid_json"),
        data: null,
      };
    }
    return { error: null, data: parsed };
  }

  async function handleSaveProfile(event) {
    if (event) {
      event.preventDefault();
    }
    if (local.savingProfile) {
      return;
    }
    local.profileError = "";

    let payload;
    if (local.rawJsonMode) {
      const parsed = parseRawJsonOrError();
      if (parsed.error) {
        local.profileError = parsed.error;
        render();
        return;
      }
      payload = parsed.data;
      if (local.isNewProfile) {
        const candidateId = String(payload.id || "").trim();
        if (!candidateId || !SLUG_RE.test(candidateId)) {
          local.profileError = `${t("errors.VALIDATION_ERROR")} (id: [a-zA-Z0-9_-]+)`;
          render();
          return;
        }
      }
    } else {
      const built = buildProfilePayloadFromForm();
      if (built.error) {
        local.profileError = built.error;
        render();
        return;
      }
      payload = built.payload;
    }

    local.savingProfile = true;
    render();
    try {
      const saved = local.isNewProfile
        ? await api.createProfile(payload)
        : await api.updateProfile(local.selectedProfileId, payload);

      const currentProfiles = Array.isArray(store.getState().profiles)
        ? store.getState().profiles
        : [];
      const nextProfiles = local.isNewProfile
        ? [...currentProfiles.filter((item) => item.id !== saved.id), saved]
        : currentProfiles.map((item) => (item.id === saved.id ? saved : item));

      loadProfileIntoDraft(saved);
      store.setState({ profiles: nextProfiles });
      store.addToast({ type: "success", message: t("common.saved") });
    } catch (err) {
      local.profileError = translateError(err);
      store.addToast({ type: "danger", message: local.profileError });
    } finally {
      local.savingProfile = false;
      render();
    }
  }

  async function handleDeleteProfile() {
    if (local.isNewProfile || !local.selectedProfileId || local.savingProfile) {
      return;
    }
    const targetId = local.selectedProfileId;
    if (typeof window !== "undefined" && typeof window.confirm === "function") {
      if (!window.confirm(`${t("common.delete")}: ${targetId}?`)) {
        return;
      }
    }
    local.savingProfile = true;
    local.profileError = "";
    render();
    try {
      await api.deleteProfile(targetId);
      const currentProfiles = Array.isArray(store.getState().profiles)
        ? store.getState().profiles
        : [];
      const nextProfiles = currentProfiles.filter((item) => item.id !== targetId);
      loadProfileIntoDraft(nextProfiles[0] || null);
      store.setState({ profiles: nextProfiles });
      store.addToast({ type: "success", message: t("common.delete") });
    } catch (err) {
      local.profileError = translateError(err);
      store.addToast({ type: "danger", message: local.profileError });
    } finally {
      local.savingProfile = false;
      render();
    }
  }

  function handleDuplicateProfile() {
    const profiles = Array.isArray(store.getState().profiles) ? store.getState().profiles : [];
    const baseSource = local.rawJsonMode
      ? parseRawJsonOrError().data || local.draftProfile
      : buildProfilePayloadFromForm().payload || local.draftProfile;
    const ownerId = store.getState().user?.id || "default";
    const cloned = normalizeProfile(baseSource, ownerId);
    const newId = makeUniqueSlug(cloned.id || "profile", profiles.map((p) => p.id));
    cloned.id = newId;
    cloned.display_name = `${cloned.display_name || newId} (${t("common.duplicate")})`;
    local.isNewProfile = true;
    local.selectedProfileId = "";
    local.draftProfile = cloned;
    local.inlineRows = pairsObjectToRows(cloned.context.inline_replacements);
    local.rawJsonText = JSON.stringify(cloned, null, 2);
    local.profileError = "";
    render();
  }

  function renderGpuControls(targetEl) {
    if (!targetEl) {
      return;
    }
    const state = store.getState();
    const devices = Array.isArray(state.devices) ? state.devices : [];
    const cudaDevices = devices.filter((dev) => dev && dev.kind === "cuda");
    const hw = local.draftProfile.hardware;
    const gpuMode = Array.isArray(hw.gpus) ? "indices" : hw.gpus === "cpu" ? "cpu" : "auto";

    const availableIndices = new Set(
      cudaDevices
        .map((dev) => dev.index)
        .filter((idx) => idx !== null && idx !== undefined)
    );
    if (Array.isArray(hw.gpus)) {
      for (const idx of hw.gpus) {
        availableIndices.add(idx);
      }
    }
    const sortedIndices = Array.from(availableIndices).sort((a, b) => a - b);

    const modeRow = el(
      "div",
      { className: "row" },
      el(
        "label",
        { className: "checkbox-label" },
        el("input", {
          type: "radio",
          name: "adv-profile-gpu-mode",
          value: "auto",
          checked: gpuMode === "auto",
          onChange: () => {
            local.draftProfile.hardware.gpus = "auto";
            renderGpuControls(gpuSectionEl);
          },
        }),
        el("span", { textContent: t("common.auto") })
      ),
      el(
        "label",
        { className: "checkbox-label" },
        el("input", {
          type: "radio",
          name: "adv-profile-gpu-mode",
          value: "cpu",
          checked: gpuMode === "cpu",
          onChange: () => {
            local.draftProfile.hardware.gpus = "cpu";
            renderGpuControls(gpuSectionEl);
          },
        }),
        el("span", { textContent: t("common.cpu") })
      ),
      el(
        "label",
        { className: "checkbox-label" },
        el("input", {
          type: "radio",
          name: "adv-profile-gpu-mode",
          value: "indices",
          checked: gpuMode === "indices",
          onChange: () => {
            const defaults = cudaDevices
              .map((dev) => dev.index)
              .filter((idx) => idx !== null && idx !== undefined);
            local.draftProfile.hardware.gpus = defaults.length > 0 ? defaults : [0];
            renderGpuControls(gpuSectionEl);
          },
        }),
        el("span", { textContent: t("common.cuda") })
      )
    );

    const gpuList = el("div", { className: "chips" });
    if (sortedIndices.length === 0) {
      gpuList.appendChild(
        el("span", { className: "help-text", textContent: `${t("common.cuda")}: ${t("common.none")}` })
      );
    } else {
      for (const idx of sortedIndices) {
        const dev = cudaDevices.find((item) => item.index === idx);
        const labelText = dev
          ? `GPU ${idx}: ${dev.name}${dev.vram_total_mb ? ` (${dev.vram_free_mb ?? "?"}/${dev.vram_total_mb} MB)` : ""}`
          : `GPU ${idx}`;
        const checked = Array.isArray(hw.gpus) && hw.gpus.includes(idx);
        gpuList.appendChild(
          el(
            "label",
            { className: "checkbox-label" },
            el("input", {
              type: "checkbox",
              checked,
              disabled: gpuMode !== "indices",
              onChange: (ev) => {
                const current = Array.isArray(local.draftProfile.hardware.gpus)
                  ? [...local.draftProfile.hardware.gpus]
                  : [];
                if (ev.target.checked) {
                  if (!current.includes(idx)) {
                    current.push(idx);
                  }
                } else if (current.length > 1) {
                  const filtered = current.filter((item) => item !== idx);
                  local.draftProfile.hardware.gpus = filtered;
                  renderGpuControls(gpuSectionEl);
                  return;
                } else {
                  ev.target.checked = true;
                  return;
                }
                local.draftProfile.hardware.gpus = current.sort((a, b) => a - b);
              },
            }),
            el("span", { textContent: labelText })
          )
        );
      }
    }

    targetEl.replaceChildren(modeRow, gpuList);
  }

  function renderProfilesTab(panel) {
    const state = store.getState();
    const profiles = Array.isArray(state.profiles) ? state.profiles : [];
    const glossaries = Array.isArray(state.glossaries) ? state.glossaries : [];
    const replacements = Array.isArray(state.replacements) ? state.replacements : [];
    const locale = state.locale || "es";
    const p = local.draftProfile;

    // Toolbar card
    const selectEl = el(
      "select",
      {
        id: "adv-profile-select",
        className: "select",
        onChange: (ev) => {
          const picked = profiles.find((item) => item.id === ev.target.value);
          if (picked) {
            loadProfileIntoDraft(picked);
            render();
          }
        },
      },
      local.isNewProfile
        ? el("option", { value: "", textContent: `-- ${t("advanced.profiles.new_profile")} --`, selected: true })
        : null,
      profiles.map((item) =>
        el("option", {
          value: item.id,
          textContent: `${localizedProfileName(item, locale)} (${item.id})`,
          selected: !local.isNewProfile && item.id === local.selectedProfileId,
        })
      )
    );

    const toolbarCard = el(
      "div",
      { className: "card" },
      el(
        "div",
        { className: "row-between" },
        el(
          "div",
          { className: "form-group", style: "flex: 1 1 260px;" },
          el("label", { className: "label", for: "adv-profile-select", textContent: t("advanced.profiles.select") }),
          selectEl
        ),
        el(
          "div",
          { className: "row" },
          el("button", {
            type: "button",
            className: "btn btn-secondary btn-sm",
            textContent: t("advanced.profiles.new_profile"),
            onClick: () => {
              loadProfileIntoDraft(null);
              render();
            },
          }),
          el("button", {
            type: "button",
            className: "btn btn-secondary btn-sm",
            textContent: t("common.duplicate"),
            onClick: handleDuplicateProfile,
          }),
          el("button", {
            type: "button",
            className: "btn btn-danger btn-sm",
            disabled: local.isNewProfile || !local.selectedProfileId || local.savingProfile,
            textContent: t("common.delete"),
            onClick: handleDeleteProfile,
          }),
          el(
            "label",
            { className: "checkbox-label" },
            el("input", {
              id: "adv-profile-raw-toggle",
              type: "checkbox",
              checked: local.rawJsonMode,
              onChange: (ev) => {
                if (ev.target.checked) {
                  const built = buildProfilePayloadFromForm();
                  const snapshot = built.payload || {
                    ...local.draftProfile,
                    context: {
                      ...local.draftProfile.context,
                      inline_replacements:
                        rowsToPairsObject(local.inlineRows).data ||
                        local.draftProfile.context.inline_replacements,
                    },
                  };
                  local.rawJsonText = JSON.stringify(snapshot, null, 2);
                  local.profileError = "";
                  local.rawJsonMode = true;
                } else {
                  const parsed = parseRawJsonOrError();
                  if (parsed.error) {
                    local.profileError = parsed.error;
                    ev.target.checked = true;
                    render();
                    return;
                  }
                  const normalized = normalizeProfile(
                    parsed.data,
                    store.getState().user?.id || "default"
                  );
                  local.draftProfile = normalized;
                  local.inlineRows = pairsObjectToRows(normalized.context.inline_replacements);
                  local.profileError = "";
                  local.rawJsonMode = false;
                }
                render();
              },
            }),
            el("span", { textContent: t("advanced.profiles.raw_json_toggle") })
          )
        )
      )
    );

    panel.appendChild(toolbarCard);

    if (local.profileError) {
      panel.appendChild(
        el("div", { className: "alert alert-danger", role: "alert", textContent: local.profileError })
      );
    }

    if (local.rawJsonMode) {
      const rawForm = el(
        "form",
        { className: "card", onSubmit: handleSaveProfile },
        el(
          "div",
          { className: "card-body" },
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "adv-profile-raw-json",
              textContent: t("advanced.profiles.raw_json_toggle"),
            }),
            el("textarea", {
              id: "adv-profile-raw-json",
              className: "textarea mono",
              rows: "22",
              value: local.rawJsonText,
              onInput: (ev) => {
                local.rawJsonText = ev.target.value;
              },
            })
          )
        ),
        el(
          "div",
          { className: "card-footer" },
          el("button", {
            type: "submit",
            className: "btn btn-primary",
            disabled: local.savingProfile,
            textContent: local.savingProfile ? t("common.loading") : t("common.save"),
          })
        )
      );
      panel.appendChild(rawForm);
      return;
    }

    // Form mode
    const form = el("form", { className: "stack", onSubmit: handleSaveProfile });

    // 1. General & Input & Preprocessing
    const generalCard = el(
      "div",
      { className: "card" },
      el(
        "div",
        { className: "card-header" },
        el("h3", { className: "card-title", textContent: t("advanced.profiles.display_name") })
      ),
      el(
        "div",
        { className: "card-body" },
        el(
          "div",
          { className: "form-grid" },
          el(
            "div",
            { className: "form-group" },
            el("label", { className: "label", for: "adv-prof-id", textContent: "ID (slug)" }),
            el("input", {
              id: "adv-prof-id",
              className: "input mono",
              type: "text",
              required: true,
              readOnly: !local.isNewProfile,
              value: p.id,
              placeholder: "mi-perfil-es",
              onInput: (ev) => {
                p.id = ev.target.value;
              },
            })
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "adv-prof-display-name",
              textContent: t("advanced.profiles.display_name"),
            }),
            el("input", {
              id: "adv-prof-display-name",
              className: "input",
              type: "text",
              required: true,
              value: p.display_name,
              onInput: (ev) => {
                p.display_name = ev.target.value;
              },
            })
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "adv-prof-input-mode",
              textContent: t("advanced.profiles.input_mode"),
            }),
            el(
              "select",
              {
                id: "adv-prof-input-mode",
                className: "select",
                onChange: (ev) => {
                  p.input.mode = ev.target.value;
                  if (p.input.mode !== "multitrack") {
                    p.preprocessing.generate_mix = false;
                  }
                  render();
                },
              },
              INPUT_MODES.map((modeOpt) =>
                el("option", {
                  value: modeOpt,
                  textContent: modeOpt,
                  selected: p.input.mode === modeOpt,
                })
              )
            )
          )
        ),
        el(
          "div",
          { className: "row" },
          el(
            "label",
            { className: "checkbox-label" },
            el("input", {
              type: "checkbox",
              checked: p.simple_visible,
              onChange: (ev) => {
                p.simple_visible = ev.target.checked;
              },
            }),
            el("span", { textContent: t("advanced.profiles.simple_visible") })
          ),
          el(
            "label",
            { className: "checkbox-label" },
            el("input", {
              type: "checkbox",
              checked: p.preprocessing.normalize_audio,
              onChange: (ev) => {
                p.preprocessing.normalize_audio = ev.target.checked;
              },
            }),
            el("span", { textContent: t("advanced.profiles.normalize_audio") })
          ),
          p.input.mode === "multitrack"
            ? el(
                "label",
                { className: "checkbox-label" },
                el("input", {
                  type: "checkbox",
                  checked: Boolean(p.preprocessing.generate_mix),
                  onChange: (ev) => {
                    p.preprocessing.generate_mix = ev.target.checked;
                  },
                }),
                el("span", { textContent: t("advanced.profiles.generate_mix") })
              )
            : null
        )
      )
    );
    form.appendChild(generalCard);

    // 2. ASR & Diarization
    const modelOptions = ASR_MODELS.includes(p.asr.model) ? ASR_MODELS : [...ASR_MODELS, p.asr.model];
    const currentLangVal = p.asr.language === null ? "auto" : p.asr.language;
    const langOptions = ASR_LANGUAGES.includes(currentLangVal)
      ? ASR_LANGUAGES
      : [...ASR_LANGUAGES, currentLangVal];
    const computeOptions = COMPUTE_TYPES.includes(p.asr.compute_type)
      ? COMPUTE_TYPES
      : [...COMPUTE_TYPES, p.asr.compute_type];

    const asrDiarCard = el(
      "div",
      { className: "card" },
      el(
        "div",
        { className: "card-header" },
        el("h3", { className: "card-title", textContent: t("advanced.profiles.asr_model") })
      ),
      el(
        "div",
        { className: "card-body" },
        el(
          "div",
          { className: "form-grid" },
          el(
            "div",
            { className: "form-group" },
            el("label", { className: "label", for: "adv-prof-asr-model", textContent: t("advanced.profiles.asr_model") }),
            el(
              "select",
              {
                id: "adv-prof-asr-model",
                className: "select",
                onChange: (ev) => {
                  p.asr.model = ev.target.value;
                },
              },
              modelOptions.map((m) =>
                el("option", { value: m, textContent: m, selected: p.asr.model === m })
              )
            )
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", { className: "label", for: "adv-prof-asr-lang", textContent: t("advanced.profiles.asr_language") }),
            el(
              "select",
              {
                id: "adv-prof-asr-lang",
                className: "select",
                onChange: (ev) => {
                  p.asr.language = ev.target.value === "auto" ? null : ev.target.value;
                },
              },
              langOptions.map((langCode) =>
                el("option", {
                  value: langCode,
                  textContent: langCode === "auto" ? t("common.auto") : langCode,
                  selected: currentLangVal === langCode,
                })
              )
            )
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "adv-prof-compute-type",
              textContent: t("advanced.profiles.compute_type"),
            }),
            el(
              "select",
              {
                id: "adv-prof-compute-type",
                className: "select",
                onChange: (ev) => {
                  p.asr.compute_type = ev.target.value;
                },
              },
              computeOptions.map((ct) =>
                el("option", { value: ct, textContent: ct, selected: p.asr.compute_type === ct })
              )
            )
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", { className: "label", for: "adv-prof-beam-size", textContent: t("advanced.profiles.beam_size") }),
            el("input", {
              id: "adv-prof-beam-size",
              className: "input",
              type: "number",
              min: "1",
              step: "1",
              value: p.asr.beam_size,
              onInput: (ev) => {
                const parsed = Number.parseInt(ev.target.value, 10);
                p.asr.beam_size = Number.isFinite(parsed) && parsed >= 1 ? parsed : 1;
              },
            })
          )
        ),
        el(
          "div",
          { className: "row" },
          el(
            "label",
            { className: "checkbox-label" },
            el("input", {
              type: "checkbox",
              checked: p.asr.vad_filter,
              onChange: (ev) => {
                p.asr.vad_filter = ev.target.checked;
              },
            }),
            el("span", { textContent: t("advanced.profiles.vad_filter") })
          ),
          el(
            "label",
            { className: "checkbox-label" },
            el("input", {
              type: "checkbox",
              checked: p.asr.word_timestamps,
              onChange: (ev) => {
                p.asr.word_timestamps = ev.target.checked;
              },
            }),
            el("span", { textContent: t("advanced.profiles.word_timestamps") })
          ),
          el(
            "label",
            { className: "checkbox-label" },
            el("input", {
              type: "checkbox",
              checked: p.diarization.enabled,
              onChange: (ev) => {
                p.diarization.enabled = ev.target.checked;
              },
            }),
            el("span", { textContent: t("advanced.profiles.diarization_enabled") })
          )
        ),
        el(
          "div",
          { className: "form-grid" },
          el(
            "div",
            { className: "form-group" },
            el("label", { className: "label", for: "adv-prof-min-spk", textContent: t("simple.min_speakers") }),
            el("input", {
              id: "adv-prof-min-spk",
              className: "input",
              type: "number",
              min: "1",
              step: "1",
              placeholder: t("common.optional"),
              value: p.diarization.min_speakers ?? "",
              onInput: (ev) => {
                const val = ev.target.value.trim();
                p.diarization.min_speakers = val ? Math.max(1, Number.parseInt(val, 10) || 1) : null;
              },
            })
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", { className: "label", for: "adv-prof-max-spk", textContent: t("simple.max_speakers") }),
            el("input", {
              id: "adv-prof-max-spk",
              className: "input",
              type: "number",
              min: "1",
              step: "1",
              placeholder: t("common.optional"),
              value: p.diarization.max_speakers ?? "",
              onInput: (ev) => {
                const val = ev.target.value.trim();
                p.diarization.max_speakers = val ? Math.max(1, Number.parseInt(val, 10) || 1) : null;
              },
            })
          )
        )
      )
    );
    form.appendChild(asrDiarCard);

    // 3. Context (initial_prompt, hotwords chips, glossaries, replacements, inline_replacements)
    const promptCounterEl = el("span", { className: "help-text mono" });
    function updatePromptCounter() {
      const approx = estimatePromptTokens(p.context.initial_prompt);
      promptCounterEl.textContent = `~${approx} / ${TOKEN_LIMIT} tokens`;
      promptCounterEl.style.color = approx > TOKEN_LIMIT ? "var(--warning)" : "";
    }
    updatePromptCounter();

    const hotwordsChipsEl = el("div", { className: "chips" });
    function renderHotwordChips() {
      hotwordsChipsEl.replaceChildren(
        ...p.context.hotwords.map((word, idx) =>
          el(
            "span",
            { className: "chip" },
            el("span", { textContent: word }),
            el("button", {
              type: "button",
              className: "chip-remove",
              "aria-label": t("common.delete"),
              textContent: "×",
              onClick: () => {
                p.context.hotwords.splice(idx, 1);
                renderHotwordChips();
              },
            })
          )
        )
      );
    }
    renderHotwordChips();

    const hotwordInput = el("input", {
      id: "adv-prof-hotword-input",
      className: "input",
      type: "text",
      placeholder: t("simple.hotwords_placeholder"),
    });
    function commitHotwordsFromInput() {
      const raw = hotwordInput.value;
      if (!raw.trim()) {
        return;
      }
      const parts = raw
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      for (const term of parts) {
        if (!p.context.hotwords.includes(term)) {
          p.context.hotwords.push(term);
        }
      }
      hotwordInput.value = "";
      renderHotwordChips();
    }
    hotwordInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === ",") {
        ev.preventDefault();
        commitHotwordsFromInput();
      }
    });

    const inlineTbody = el("tbody");
    function renderInlineRows() {
      inlineTbody.replaceChildren(
        ...local.inlineRows.map((row, idx) =>
          el(
            "tr",
            null,
            el(
              "td",
              null,
              el("input", {
                className: "input",
                type: "text",
                placeholder: t("advanced.replacements.source_placeholder"),
                value: row.source,
                onInput: (ev) => {
                  row.source = ev.target.value;
                },
              })
            ),
            el(
              "td",
              null,
              el("input", {
                className: "input",
                type: "text",
                placeholder: t("advanced.replacements.target_placeholder"),
                value: row.target,
                onInput: (ev) => {
                  row.target = ev.target.value;
                },
              })
            ),
            el(
              "td",
              null,
              el("button", {
                type: "button",
                className: "btn btn-ghost btn-sm",
                textContent: t("common.delete"),
                onClick: () => {
                  local.inlineRows.splice(idx, 1);
                  renderInlineRows();
                },
              })
            )
          )
        )
      );
    }
    renderInlineRows();

    const contextCard = el(
      "div",
      { className: "card" },
      el(
        "div",
        { className: "card-header" },
        el("h3", { className: "card-title", textContent: t("advanced.profiles.initial_prompt") })
      ),
      el(
        "div",
        { className: "card-body" },
        el(
          "div",
          { className: "form-group" },
          el(
            "div",
            { className: "row-between" },
            el("label", { className: "label", for: "adv-prof-prompt", textContent: t("advanced.profiles.initial_prompt") }),
            promptCounterEl
          ),
          el("textarea", {
            id: "adv-prof-prompt",
            className: "textarea",
            rows: "3",
            value: p.context.initial_prompt,
            onInput: (ev) => {
              p.context.initial_prompt = ev.target.value;
              updatePromptCounter();
            },
          })
        ),
        el(
          "div",
          { className: "form-group" },
          el("label", { className: "label", for: "adv-prof-hotword-input", textContent: t("advanced.profiles.hotwords") }),
          el(
            "div",
            { className: "chip-input-row" },
            hotwordInput,
            el("button", {
              type: "button",
              className: "btn btn-secondary btn-sm",
              textContent: t("common.create"),
              onClick: commitHotwordsFromInput,
            })
          ),
          hotwordsChipsEl
        ),
        el(
          "div",
          { className: "grid-2" },
          el(
            "div",
            { className: "form-group" },
            el("span", { className: "label", textContent: t("advanced.profiles.glossaries") }),
            glossaries.length === 0
              ? el("span", { className: "help-text", textContent: t("advanced.glossaries.empty") })
              : el(
                  "div",
                  { className: "stack" },
                  glossaries.map((g) =>
                    el(
                      "label",
                      { className: "checkbox-label" },
                      el("input", {
                        type: "checkbox",
                        checked: p.context.glossary_ids.includes(g.id),
                        onChange: (ev) => {
                          if (ev.target.checked) {
                            if (!p.context.glossary_ids.includes(g.id)) {
                              p.context.glossary_ids.push(g.id);
                            }
                          } else {
                            p.context.glossary_ids = p.context.glossary_ids.filter((id) => id !== g.id);
                          }
                        },
                      }),
                      el("span", { textContent: `${g.name || g.id} (${g.id})` })
                    )
                  )
                )
          ),
          el(
            "div",
            { className: "form-group" },
            el("span", { className: "label", textContent: t("advanced.profiles.replacements") }),
            replacements.length === 0
              ? el("span", { className: "help-text", textContent: t("advanced.replacements.empty") })
              : el(
                  "div",
                  { className: "stack" },
                  replacements.map((r) =>
                    el(
                      "label",
                      { className: "checkbox-label" },
                      el("input", {
                        type: "checkbox",
                        checked: p.context.replacement_ids.includes(r.id),
                        onChange: (ev) => {
                          if (ev.target.checked) {
                            if (!p.context.replacement_ids.includes(r.id)) {
                              p.context.replacement_ids.push(r.id);
                            }
                          } else {
                            p.context.replacement_ids = p.context.replacement_ids.filter((id) => id !== r.id);
                          }
                        },
                      }),
                      el("span", { textContent: `${r.name || r.id} (${r.id})` })
                    )
                  )
                )
          )
        ),
        el(
          "div",
          { className: "form-group" },
          el(
            "div",
            { className: "row-between" },
            el("span", { className: "label", textContent: t("advanced.profiles.inline_replacements") }),
            el("button", {
              type: "button",
              className: "btn btn-secondary btn-sm",
              textContent: `+ ${t("advanced.replacements.add_rule")}`,
              onClick: () => {
                local.inlineRows.push({ source: "", target: "" });
                renderInlineRows();
              },
            })
          ),
          el(
            "div",
            { className: "table-wrap" },
            el(
              "table",
              { className: "table" },
              el(
                "thead",
                null,
                el(
                  "tr",
                  null,
                  el("th", { textContent: t("advanced.replacements.source_placeholder") }),
                  el("th", { textContent: t("advanced.replacements.target_placeholder") }),
                  el("th", { textContent: t("common.actions") })
                )
              ),
              inlineTbody
            )
          )
        )
      )
    );
    form.appendChild(contextCard);

    // 4. Hardware
    gpuSectionEl = el("div", { className: "stack" });
    renderGpuControls(gpuSectionEl);

    const hardwareCard = el(
      "div",
      { className: "card" },
      el(
        "div",
        { className: "card-header" },
        el("h3", { className: "card-title", textContent: t("advanced.profiles.hardware_gpus") })
      ),
      el(
        "div",
        { className: "card-body" },
        gpuSectionEl,
        el(
          "div",
          { className: "form-grid" },
          el(
            "label",
            { className: "checkbox-label" },
            el("input", {
              type: "checkbox",
              checked: p.hardware.cpu_fallback,
              onChange: (ev) => {
                p.hardware.cpu_fallback = ev.target.checked;
              },
            }),
            el("span", { textContent: t("advanced.profiles.hardware_cpu_fallback") })
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "adv-prof-cpu-threads",
              textContent: t("advanced.profiles.hardware_cpu_threads"),
            }),
            el("input", {
              id: "adv-prof-cpu-threads",
              className: "input",
              type: "number",
              min: "0",
              step: "1",
              value: p.hardware.cpu_threads,
              onInput: (ev) => {
                const parsed = Number.parseInt(ev.target.value, 10);
                p.hardware.cpu_threads = Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
              },
            })
          )
        )
      )
    );
    form.appendChild(hardwareCard);

    // 5. Output (Subtitles)
    const outputCard = el(
      "div",
      { className: "card" },
      el(
        "div",
        { className: "card-header" },
        el("h3", { className: "card-title", textContent: t("advanced.profiles.voice_style") })
      ),
      el(
        "div",
        { className: "card-body" },
        el(
          "div",
          { className: "form-grid" },
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "adv-prof-voice-style",
              textContent: t("advanced.profiles.voice_style"),
            }),
            el(
              "select",
              {
                id: "adv-prof-voice-style",
                className: "select",
                onChange: (ev) => {
                  p.output.subtitles.voice_style = ev.target.value;
                },
              },
              VOICE_STYLES.map((styleOpt) =>
                el("option", {
                  value: styleOpt,
                  textContent: styleOpt,
                  selected: p.output.subtitles.voice_style === styleOpt,
                })
              )
            )
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "adv-prof-max-cue",
              textContent: t("advanced.profiles.max_cue_seconds"),
            }),
            el("input", {
              id: "adv-prof-max-cue",
              className: "input",
              type: "number",
              min: "0.5",
              step: "0.5",
              value: p.output.subtitles.max_cue_seconds,
              onInput: (ev) => {
                const parsed = Number.parseFloat(ev.target.value);
                p.output.subtitles.max_cue_seconds = Number.isFinite(parsed) && parsed > 0 ? parsed : 6.0;
              },
            })
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "adv-prof-max-chars",
              textContent: t("advanced.profiles.max_chars_per_line"),
            }),
            el("input", {
              id: "adv-prof-max-chars",
              className: "input",
              type: "number",
              min: "1",
              step: "1",
              value: p.output.subtitles.max_chars_per_line,
              onInput: (ev) => {
                const parsed = Number.parseInt(ev.target.value, 10);
                p.output.subtitles.max_chars_per_line = Number.isFinite(parsed) && parsed >= 1 ? parsed : 68;
              },
            })
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "adv-prof-max-lines",
              textContent: t("advanced.profiles.max_lines_per_cue"),
            }),
            el("input", {
              id: "adv-prof-max-lines",
              className: "input",
              type: "number",
              min: "1",
              step: "1",
              value: p.output.subtitles.max_lines_per_cue,
              onInput: (ev) => {
                const parsed = Number.parseInt(ev.target.value, 10);
                p.output.subtitles.max_lines_per_cue = Number.isFinite(parsed) && parsed >= 1 ? parsed : 2;
              },
            })
          )
        )
      ),
      el(
        "div",
        { className: "card-footer" },
        el("button", {
          type: "submit",
          className: "btn btn-primary",
          disabled: local.savingProfile,
          textContent: local.savingProfile ? t("common.loading") : t("common.save"),
        })
      )
    );
    form.appendChild(outputCard);

    panel.appendChild(form);
  }

  async function handleImportGlossaryFile(file) {
    if (!file || local.importingGlossary) {
      return;
    }
    local.importingGlossary = true;
    local.glossaryError = "";
    render();
    try {
      const formData = new FormData();
      formData.append("file", file);
      if (local.isNewGlossary && local.draftGlossary.id.trim()) {
        formData.append("glossary_id", local.draftGlossary.id.trim());
      }
      if (local.isNewGlossary && local.draftGlossary.name.trim()) {
        formData.append("name", local.draftGlossary.name.trim());
      }
      const imported = await api.importGlossary(formData);
      const current = Array.isArray(store.getState().glossaries) ? store.getState().glossaries : [];
      const exists = current.some((item) => item.id === imported.id);
      const next = exists
        ? current.map((item) => (item.id === imported.id ? imported : item))
        : [...current, imported];
      loadGlossaryIntoDraft(imported);
      store.setState({ glossaries: next });
      store.addToast({ type: "success", message: t("common.saved") });
    } catch (err) {
      local.glossaryError = translateError(err);
      store.addToast({ type: "danger", message: local.glossaryError });
    } finally {
      local.importingGlossary = false;
      render();
    }
  }

  async function handleSaveGlossary(event) {
    event.preventDefault();
    if (local.savingGlossary) {
      return;
    }
    local.glossaryError = "";
    const g = local.draftGlossary;
    const id = String(g.id || "").trim();
    const name = String(g.name || "").trim();
    if (!id || !SLUG_RE.test(id)) {
      local.glossaryError = `${t("errors.VALIDATION_ERROR")} (id: [a-zA-Z0-9_-]+)`;
      render();
      return;
    }
    if (!name) {
      local.glossaryError = `${t("errors.VALIDATION_ERROR")} (${t("advanced.glossaries.name")})`;
      render();
      return;
    }

    const terms = String(g.termsText || "")
      .split(/\r?\n/)
      .map((item) => item.trim())
      .filter(Boolean);

    const payload = {
      id,
      name,
      owner_user_id: g.owner_user_id || store.getState().user?.id || "default",
      visibility: g.visibility === "private" ? "private" : "shared",
      terms,
    };

    local.savingGlossary = true;
    render();
    try {
      const saved = local.isNewGlossary
        ? await api.createGlossary(payload)
        : await api.updateGlossary(local.selectedGlossaryId, payload);

      const current = Array.isArray(store.getState().glossaries) ? store.getState().glossaries : [];
      const next = local.isNewGlossary
        ? [...current.filter((item) => item.id !== saved.id), saved]
        : current.map((item) => (item.id === saved.id ? saved : item));

      loadGlossaryIntoDraft(saved);
      store.setState({ glossaries: next });
      store.addToast({ type: "success", message: t("common.saved") });
    } catch (err) {
      local.glossaryError = translateError(err);
      store.addToast({ type: "danger", message: local.glossaryError });
    } finally {
      local.savingGlossary = false;
      render();
    }
  }

  async function handleDeleteGlossary() {
    if (local.isNewGlossary || !local.selectedGlossaryId || local.savingGlossary) {
      return;
    }
    const targetId = local.selectedGlossaryId;
    if (typeof window !== "undefined" && typeof window.confirm === "function") {
      if (!window.confirm(`${t("common.delete")}: ${targetId}?`)) {
        return;
      }
    }
    local.savingGlossary = true;
    local.glossaryError = "";
    render();
    try {
      await api.deleteGlossary(targetId);
      const current = Array.isArray(store.getState().glossaries) ? store.getState().glossaries : [];
      const next = current.filter((item) => item.id !== targetId);
      loadGlossaryIntoDraft(next[0] || null);
      store.setState({ glossaries: next });
      store.addToast({ type: "success", message: t("common.delete") });
    } catch (err) {
      local.glossaryError = translateError(err);
      store.addToast({ type: "danger", message: local.glossaryError });
    } finally {
      local.savingGlossary = false;
      render();
    }
  }

  function renderGlossariesTab(panel) {
    const glossaries = Array.isArray(store.getState().glossaries) ? store.getState().glossaries : [];
    const g = local.draftGlossary;

    const importInput = el("input", {
      type: "file",
      className: "sr-only",
      accept: ".txt,.csv,text/plain,text/csv",
      onChange: (ev) => {
        const file = ev.target.files?.[0];
        ev.target.value = "";
        if (file) {
          void handleImportGlossaryFile(file);
        }
      },
    });

    const listCard = el(
      "div",
      { className: "card" },
      el(
        "div",
        { className: "row-between" },
        el(
          "div",
          { className: "chips" },
          glossaries.length === 0
            ? el("span", { className: "muted", textContent: t("advanced.glossaries.empty") })
            : glossaries.map((item) =>
                el("button", {
                  type: "button",
                  className: `btn btn-sm ${!local.isNewGlossary && item.id === local.selectedGlossaryId ? "btn-primary" : "btn-secondary"}`,
                  textContent: `${item.name || item.id} (${Array.isArray(item.terms) ? item.terms.length : 0})`,
                  onClick: () => {
                    loadGlossaryIntoDraft(item);
                    render();
                  },
                })
              )
        ),
        el(
          "div",
          { className: "row" },
          importInput,
          el("button", {
            type: "button",
            className: "btn btn-secondary btn-sm",
            disabled: local.importingGlossary,
            textContent: local.importingGlossary
              ? t("common.loading")
              : t("advanced.glossaries.import_button"),
            onClick: () => importInput.click(),
          }),
          el("button", {
            type: "button",
            className: "btn btn-secondary btn-sm",
            textContent: t("common.create"),
            onClick: () => {
              loadGlossaryIntoDraft(null);
              render();
            },
          })
        )
      )
    );
    panel.appendChild(listCard);

    if (local.glossaryError) {
      panel.appendChild(
        el("div", { className: "alert alert-danger", role: "alert", textContent: local.glossaryError })
      );
    }

    const editorCard = el(
      "form",
      { className: "card", onSubmit: handleSaveGlossary },
      el(
        "div",
        { className: "card-body" },
        el(
          "div",
          { className: "form-grid" },
          el(
            "div",
            { className: "form-group" },
            el("label", { className: "label", for: "adv-gloss-id", textContent: "ID (slug)" }),
            el("input", {
              id: "adv-gloss-id",
              className: "input mono",
              type: "text",
              required: true,
              readOnly: !local.isNewGlossary,
              value: g.id,
              placeholder: "glosario-tecnico",
              onInput: (ev) => {
                g.id = ev.target.value;
              },
            })
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", { className: "label", for: "adv-gloss-name", textContent: t("advanced.glossaries.name") }),
            el("input", {
              id: "adv-gloss-name",
              className: "input",
              type: "text",
              required: true,
              value: g.name,
              onInput: (ev) => {
                g.name = ev.target.value;
              },
            })
          )
        ),
        el(
          "div",
          { className: "form-group" },
          el("label", { className: "label", for: "adv-gloss-terms", textContent: t("advanced.glossaries.terms") }),
          el("textarea", {
            id: "adv-gloss-terms",
            className: "textarea mono",
            rows: "10",
            value: g.termsText,
            onInput: (ev) => {
              g.termsText = ev.target.value;
            },
          })
        )
      ),
      el(
        "div",
        { className: "card-footer" },
        !local.isNewGlossary && local.selectedGlossaryId
          ? el("button", {
              type: "button",
              className: "btn btn-danger",
              disabled: local.savingGlossary,
              textContent: t("common.delete"),
              onClick: handleDeleteGlossary,
            })
          : null,
        el("button", {
          type: "submit",
          className: "btn btn-primary",
          disabled: local.savingGlossary,
          textContent: local.savingGlossary ? t("common.loading") : t("common.save"),
        })
      )
    );
    panel.appendChild(editorCard);
  }

  async function handleSaveReplacement(event) {
    event.preventDefault();
    if (local.savingReplacement) {
      return;
    }
    local.replacementError = "";
    const r = local.draftReplacement;
    const id = String(r.id || "").trim();
    const name = String(r.name || "").trim();
    if (!id || !SLUG_RE.test(id)) {
      local.replacementError = `${t("errors.VALIDATION_ERROR")} (id: [a-zA-Z0-9_-]+)`;
      render();
      return;
    }
    if (!name) {
      local.replacementError = `${t("errors.VALIDATION_ERROR")} (${t("advanced.replacements.name")})`;
      render();
      return;
    }

    const parsedRules = rowsToPairsObject(r.rows);
    if (parsedRules.error === "empty_source") {
      local.replacementError = t("errors.VALIDATION_ERROR");
      render();
      return;
    }

    const payload = {
      id,
      name,
      owner_user_id: r.owner_user_id || store.getState().user?.id || "default",
      visibility: r.visibility === "private" ? "private" : "shared",
      case_sensitive: Boolean(r.case_sensitive),
      rules: parsedRules.data,
    };

    local.savingReplacement = true;
    render();
    try {
      const saved = local.isNewReplacement
        ? await api.createReplacement(payload)
        : await api.updateReplacement(local.selectedReplacementId, payload);

      const current = Array.isArray(store.getState().replacements)
        ? store.getState().replacements
        : [];
      const next = local.isNewReplacement
        ? [...current.filter((item) => item.id !== saved.id), saved]
        : current.map((item) => (item.id === saved.id ? saved : item));

      loadReplacementIntoDraft(saved);
      store.setState({ replacements: next });
      store.addToast({ type: "success", message: t("common.saved") });
    } catch (err) {
      local.replacementError = translateError(err);
      store.addToast({ type: "danger", message: local.replacementError });
    } finally {
      local.savingReplacement = false;
      render();
    }
  }

  async function handleDeleteReplacement() {
    if (local.isNewReplacement || !local.selectedReplacementId || local.savingReplacement) {
      return;
    }
    const targetId = local.selectedReplacementId;
    if (typeof window !== "undefined" && typeof window.confirm === "function") {
      if (!window.confirm(`${t("common.delete")}: ${targetId}?`)) {
        return;
      }
    }
    local.savingReplacement = true;
    local.replacementError = "";
    render();
    try {
      await api.deleteReplacement(targetId);
      const current = Array.isArray(store.getState().replacements)
        ? store.getState().replacements
        : [];
      const next = current.filter((item) => item.id !== targetId);
      loadReplacementIntoDraft(next[0] || null);
      store.setState({ replacements: next });
      store.addToast({ type: "success", message: t("common.delete") });
    } catch (err) {
      local.replacementError = translateError(err);
      store.addToast({ type: "danger", message: local.replacementError });
    } finally {
      local.savingReplacement = false;
      render();
    }
  }

  function renderReplacementsTab(panel) {
    const replacements = Array.isArray(store.getState().replacements)
      ? store.getState().replacements
      : [];
    const r = local.draftReplacement;

    const listCard = el(
      "div",
      { className: "card" },
      el(
        "div",
        { className: "row-between" },
        el(
          "div",
          { className: "chips" },
          replacements.length === 0
            ? el("span", { className: "muted", textContent: t("advanced.replacements.empty") })
            : replacements.map((item) =>
                el("button", {
                  type: "button",
                  className: `btn btn-sm ${!local.isNewReplacement && item.id === local.selectedReplacementId ? "btn-primary" : "btn-secondary"}`,
                  textContent: `${item.name || item.id} (${Object.keys(item.rules || {}).length})`,
                  onClick: () => {
                    loadReplacementIntoDraft(item);
                    render();
                  },
                })
              )
        ),
        el("button", {
          type: "button",
          className: "btn btn-secondary btn-sm",
          textContent: t("common.create"),
          onClick: () => {
            loadReplacementIntoDraft(null);
            render();
          },
        })
      )
    );
    panel.appendChild(listCard);

    if (local.replacementError) {
      panel.appendChild(
        el("div", { className: "alert alert-danger", role: "alert", textContent: local.replacementError })
      );
    }

    const rulesTbody = el("tbody");
    function renderRulesRows() {
      rulesTbody.replaceChildren(
        ...r.rows.map((row, idx) =>
          el(
            "tr",
            null,
            el(
              "td",
              null,
              el("input", {
                className: "input",
                type: "text",
                placeholder: t("advanced.replacements.source_placeholder"),
                value: row.source,
                onInput: (ev) => {
                  row.source = ev.target.value;
                },
              })
            ),
            el(
              "td",
              null,
              el("input", {
                className: "input",
                type: "text",
                placeholder: t("advanced.replacements.target_placeholder"),
                value: row.target,
                onInput: (ev) => {
                  row.target = ev.target.value;
                },
              })
            ),
            el(
              "td",
              null,
              el("button", {
                type: "button",
                className: "btn btn-ghost btn-sm",
                textContent: t("common.delete"),
                onClick: () => {
                  r.rows.splice(idx, 1);
                  renderRulesRows();
                },
              })
            )
          )
        )
      );
    }
    renderRulesRows();

    const editorCard = el(
      "form",
      { className: "card", onSubmit: handleSaveReplacement },
      el(
        "div",
        { className: "card-body" },
        el(
          "div",
          { className: "form-grid" },
          el(
            "div",
            { className: "form-group" },
            el("label", { className: "label", for: "adv-repl-id", textContent: "ID (slug)" }),
            el("input", {
              id: "adv-repl-id",
              className: "input mono",
              type: "text",
              required: true,
              readOnly: !local.isNewReplacement,
              value: r.id,
              placeholder: "reemplazos-es",
              onInput: (ev) => {
                r.id = ev.target.value;
              },
            })
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", { className: "label", for: "adv-repl-name", textContent: t("advanced.replacements.name") }),
            el("input", {
              id: "adv-repl-name",
              className: "input",
              type: "text",
              required: true,
              value: r.name,
              onInput: (ev) => {
                r.name = ev.target.value;
              },
            })
          )
        ),
        el(
          "label",
          { className: "checkbox-label" },
          el("input", {
            type: "checkbox",
            checked: r.case_sensitive,
            onChange: (ev) => {
              r.case_sensitive = ev.target.checked;
            },
          }),
          el("span", { textContent: t("advanced.replacements.case_sensitive") })
        ),
        el(
          "div",
          { className: "form-group" },
          el(
            "div",
            { className: "row-between" },
            el("span", { className: "label", textContent: t("advanced.tabs.replacements") }),
            el("button", {
              type: "button",
              className: "btn btn-secondary btn-sm",
              textContent: `+ ${t("advanced.replacements.add_rule")}`,
              onClick: () => {
                r.rows.push({ source: "", target: "" });
                renderRulesRows();
              },
            })
          ),
          el(
            "div",
            { className: "table-wrap" },
            el(
              "table",
              { className: "table" },
              el(
                "thead",
                null,
                el(
                  "tr",
                  null,
                  el("th", { textContent: t("advanced.replacements.source_placeholder") }),
                  el("th", { textContent: t("advanced.replacements.target_placeholder") }),
                  el("th", { textContent: t("common.actions") })
                )
              ),
              rulesTbody
            )
          )
        )
      ),
      el(
        "div",
        { className: "card-footer" },
        !local.isNewReplacement && local.selectedReplacementId
          ? el("button", {
              type: "button",
              className: "btn btn-danger",
              disabled: local.savingReplacement,
              textContent: t("common.delete"),
              onClick: handleDeleteReplacement,
            })
          : null,
        el("button", {
          type: "submit",
          className: "btn btn-primary",
          disabled: local.savingReplacement,
          textContent: local.savingReplacement ? t("common.loading") : t("common.save"),
        })
      )
    );
    panel.appendChild(editorCard);
  }

  function render() {
    gpuSectionEl = null;

    const header = el(
      "div",
      { className: "section-header" },
      el("h2", { className: "section-title", textContent: t("advanced.title") })
    );

    const subTabs = el(
      "div",
      { className: "sub-tabs", role: "tablist" },
      [
        ["profiles", t("advanced.tabs.profiles")],
        ["glossaries", t("advanced.tabs.glossaries")],
        ["replacements", t("advanced.tabs.replacements")],
      ].map(([tabId, label]) =>
        el("button", {
          type: "button",
          role: "tab",
          className: `sub-tab ${local.subTab === tabId ? "is-active" : ""}`,
          "aria-selected": local.subTab === tabId ? "true" : "false",
          textContent: label,
          onClick: () => {
            if (local.subTab !== tabId) {
              local.subTab = tabId;
              render();
            }
          },
        })
      )
    );

    const panel = el("div", { className: "stack", role: "tabpanel" });
    if (local.subTab === "profiles") {
      renderProfilesTab(panel);
    } else if (local.subTab === "glossaries") {
      renderGlossariesTab(panel);
    } else {
      renderReplacementsTab(panel);
    }

    container.replaceChildren(header, subTabs, panel);
  }

  const unsubscribe = store.subscribe((state, prevState) => {
    if (state.activeTab === "profiles" && prevState.activeTab !== "profiles") {
      void ensureConfigResourcesLoaded();
    }

    if (prevState.locale !== state.locale) {
      render();
      return;
    }

    let shouldRender = false;

    if (prevState.profiles !== state.profiles) {
      const profiles = Array.isArray(state.profiles) ? state.profiles : [];
      if (local.isNewProfile && !local.draftProfile.id && !local.draftProfile.display_name && profiles.length > 0) {
        loadProfileIntoDraft(profiles[0]);
      } else if (!local.isNewProfile && local.selectedProfileId) {
        const current = profiles.find((item) => item.id === local.selectedProfileId);
        if (!current) {
          loadProfileIntoDraft(profiles[0] || null);
        }
      }
      if (local.subTab === "profiles") {
        shouldRender = true;
      }
    }

    if (prevState.glossaries !== state.glossaries) {
      const glossaries = Array.isArray(state.glossaries) ? state.glossaries : [];
      if (local.isNewGlossary && !local.draftGlossary.id && !local.draftGlossary.name && glossaries.length > 0) {
        loadGlossaryIntoDraft(glossaries[0]);
      } else if (!local.isNewGlossary && local.selectedGlossaryId) {
        const current = glossaries.find((item) => item.id === local.selectedGlossaryId);
        if (!current) {
          loadGlossaryIntoDraft(glossaries[0] || null);
        }
      }
      if (local.subTab === "glossaries" || local.subTab === "profiles") {
        shouldRender = true;
      }
    }

    if (prevState.replacements !== state.replacements) {
      const replacements = Array.isArray(state.replacements) ? state.replacements : [];
      if (local.isNewReplacement && !local.draftReplacement.id && !local.draftReplacement.name && replacements.length > 0) {
        loadReplacementIntoDraft(replacements[0]);
      } else if (!local.isNewReplacement && local.selectedReplacementId) {
        const current = replacements.find((item) => item.id === local.selectedReplacementId);
        if (!current) {
          loadReplacementIntoDraft(replacements[0] || null);
        }
      }
      if (local.subTab === "replacements" || local.subTab === "profiles") {
        shouldRender = true;
      }
    }

    if (shouldRender) {
      render();
      return;
    }

    if (prevState.devices !== state.devices && local.subTab === "profiles" && !local.rawJsonMode && gpuSectionEl) {
      renderGpuControls(gpuSectionEl);
    }
  });

  render();
  if (store.getState().activeTab === "profiles") {
    void ensureConfigResourcesLoaded();
  }

  const cleanup = () => {
    unsubscribe();
    containerCleanups.delete(container);
  };
  containerCleanups.set(container, cleanup);
  return cleanup;
}

export default { mount };
