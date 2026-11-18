/**
 * Simple View: iOS-safe file upload, 4 clear modes, multitrack editor,
 * quick hotwords chips, and live multi-GPU / active queue monitor.
 */

import { api } from "../api.js";
import { i18n } from "../i18n.js";
import { store } from "../store.js";
import { inspectLocalZipFile, supportsLocalZipInspection } from "../zip_inspect.js";

const ZIP_SUPPORTED = supportsLocalZipInspection();
const ACCEPTED_EXTENSIONS = ZIP_SUPPORTED
  ? ".m4a,.mp3,.wav,.flac,.ogg,.opus,.aac,.mp4,.mkv,.webm,.mov,.zip"
  : ".m4a,.mp3,.wav,.flac,.ogg,.opus,.aac,.mp4,.mkv,.webm,.mov";
const ACCEPTED_EXTENSION_SET = new Set(
  ACCEPTED_EXTENSIONS.split(",")
    .map((ext) => ext.trim().toLowerCase())
    .filter(Boolean)
);

const ACTIVE_STATUSES = new Set([
  "queued",
  "preprocessing",
  "downloading_model",
  "loading_model",
  "transcribing",
  "diarizing",
  "merging",
  "exporting",
  "cancelling",
]);

const TERMINAL_STATUSES = new Set([
  "completed",
  "completed_partial",
  "cancelled",
  "failed",
]);

const MULTI_MODE_IDS = ["multitrack", "batch"];

const formState = {
  files: [],
  resolvedEntries: [],
  uiMode: "single",
  diarizeEnabled: null,
  minSpeakers: "",
  maxSpeakers: "",
  profileId: "",
  title: "",
  generateMix: null,
  hotwords: [],
  hotwordDraft: "",
  multitrackRows: [],
  inspectKey: "",
  inspectStatus: "idle",
  inspectError: "",
  selectionWarning: "",
  isSubmitting: false,
  submitError: "",
  cancellingIds: new Set(),
};

let rootContainer = null;
let composerHost = null;
let monitorHost = null;
let unsubscribeStore = null;

const QUEUE_TTL_MS = 120_000;
const dismissedGroupKeys = new Set();
const collapsedGroupKeys = new Set();

let monitorRefs = null;
let ttlTimerId = null;

function stripExtension(filename) {
  const raw = String(filename || "");
  const dot = raw.lastIndexOf(".");
  return dot > 0 ? raw.slice(0, dot) : raw;
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  if (value < 1024 * 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDurationSeconds(totalSeconds) {
  const secs = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(secs / 3600);
  const mins = Math.floor((secs % 3600) / 60);
  const rem = secs % 60;
  const mm = String(mins).padStart(2, "0");
  const ss = String(rem).padStart(2, "0");
  if (hours > 0) {
    return `${hours}:${mm}:${ss}`;
  }
  return `${mm}:${ss}`;
}

function computeEta(processedMs, totalMs, rtfx) {
  const processed = Number(processedMs || 0);
  const total = Number(totalMs || 0);
  const speed = Number(rtfx || 0);
  if (total <= 0 || processed < 0 || total <= processed || speed <= 0) {
    return "";
  }
  const remainingAudioSec = (total - processed) / 1000;
  return formatDurationSeconds(remainingAudioSec / speed);
}

function getSimpleProfiles() {
  const all = Array.isArray(store.getState().profiles) ? store.getState().profiles : [];
  const visible = all.filter((profile) => profile && profile.simple_visible !== false);
  return visible.length > 0 ? visible : all;
}

function resolveDefaultProfileId(profiles) {
  if (!profiles.length) {
    return "";
  }
  if (formState.profileId && profiles.some((p) => p.id === formState.profileId)) {
    return formState.profileId;
  }
  const preferred = store.getState().user?.preferences?.default_profile_id;
  if (preferred && profiles.some((p) => p.id === preferred)) {
    return preferred;
  }
  return profiles[0].id;
}

function getProfileLabel(profile) {
  return profile?.display_name || profile?.id || "";
}

function resolveGenerateMix(profiles, profileId) {
  if (formState.generateMix !== null) {
    return Boolean(formState.generateMix);
  }
  const active = profiles.find((p) => p.id === profileId);
  return Boolean(active?.preprocessing?.generate_mix);
}

function resolveDiarizeEnabled(profiles, profileId) {
  if (formState.diarizeEnabled !== null) {
    return Boolean(formState.diarizeEnabled);
  }
  const active = profiles.find((p) => p.id === profileId);
  return Boolean(active?.diarization?.enabled);
}

function fileIdentity(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

function getActiveEntries() {
  return formState.resolvedEntries.filter((entry) => entry.included !== false);
}

function getExcludedZipTracks() {
  return formState.resolvedEntries.filter(
    (entry) => entry.sourceType === "zip" && entry.included === false
  );
}

function syncModeFromActiveEntries() {
  const activeCount = getActiveEntries().length;
  if (activeCount === 1) {
    formState.uiMode = "single";
  } else if (
    activeCount >= 2 &&
    formState.uiMode !== "multitrack" &&
    formState.uiMode !== "batch"
  ) {
    formState.uiMode = "multitrack";
  }
}

function setEntryIncluded(entryId, included) {
  let changed = false;
  for (const entry of formState.resolvedEntries) {
    if (entry.id === entryId && entry.included !== included) {
      entry.included = included;
      changed = true;
    }
  }
  for (const row of formState.multitrackRows) {
    if (row.id === entryId && row.included !== included) {
      row.included = included;
      changed = true;
    }
  }
  if (!changed) {
    return;
  }
  formState.submitError = "";
  syncModeFromActiveEntries();
  renderComposer();
}

async function analyzeSelectedFiles() {
  const files = formState.files;
  if (!files.length) {
    formState.resolvedEntries = [];
    formState.multitrackRows = [];
    formState.inspectKey = "";
    formState.inspectStatus = "idle";
    formState.inspectError = "";
    formState.uiMode = "single";
    return;
  }

  const key = files.map(fileIdentity).join("|");
  if (formState.inspectKey === key && formState.inspectStatus === "done") {
    return;
  }

  formState.inspectKey = key;
  const hasZip = files.some((f) => String(f.name || "").toLowerCase().endsWith(".zip"));
  if (hasZip && !ZIP_SUPPORTED) {
    formState.inspectStatus = "error";
    formState.inspectError = i18n.t("errors.INVALID_ARCHIVE");
    formState.resolvedEntries = [];
    formState.multitrackRows = [];
    renderComposer();
    return;
  }

  if (hasZip) {
    formState.inspectStatus = "loading";
    formState.inspectError = "";
    renderComposer();
  }

  try {
    const existingById = new Map(
      formState.multitrackRows.map((row) => [row.id, row])
    );
    const resolved = [];
    for (let fileIdx = 0; fileIdx < files.length; fileIdx++) {
      const file = files[fileIdx];
      if (String(file.name || "").toLowerCase().endsWith(".zip")) {
        const { entries } = await inspectLocalZipFile(file);
        if (formState.inspectKey !== key) {
          return;
        }
        for (const entry of entries) {
          const id = `zip:${fileIdx}:${entry.path}`;
          const prev = existingById.get(id);
          resolved.push({
            id,
            sourceType: "zip",
            zipPath: entry.path,
            filename: entry.filename,
            speaker_name: prev?.speaker_name ?? stripExtension(entry.filename),
            offset_ms: prev?.offset_ms ?? 0,
            included: prev ? prev.included !== false : true,
          });
        }
      } else {
        const id = `file:${fileIdx}:${file.name}`;
        const prev = existingById.get(id);
        resolved.push({
          id,
          sourceType: "file",
          zipPath: "",
          filename: file.name,
          speaker_name: prev?.speaker_name ?? stripExtension(file.name),
          offset_ms: prev?.offset_ms ?? 0,
          included: prev ? prev.included !== false : true,
        });
      }
    }

    if (formState.inspectKey !== key) {
      return;
    }

    if (resolved.length === 0) {
      formState.inspectStatus = "error";
      formState.inspectError = i18n.t("errors.INVALID_ARCHIVE");
      formState.resolvedEntries = [];
      formState.multitrackRows = [];
      renderComposer();
      return;
    }

    formState.resolvedEntries = resolved;
    formState.inspectStatus = "done";
    formState.inspectError = "";
    formState.multitrackRows = resolved.map((entry) => ({ ...entry }));
    syncModeFromActiveEntries();
  } catch (err) {
    if (formState.inspectKey !== key) {
      return;
    }
    formState.inspectStatus = "error";
    formState.inspectError = err?.message || i18n.t("errors.INVALID_ARCHIVE");
    formState.resolvedEntries = [];
    formState.multitrackRows = [];
  }
  renderComposer();
}

function getFileExtension(filename) {
  const name = String(filename || "").trim();
  const dotIdx = name.lastIndexOf(".");
  if (dotIdx <= 0 || dotIdx === name.length - 1) {
    return "";
  }
  return name.slice(dotIdx).toLowerCase();
}

function formatRejectedNames(names, maxVisible = 3) {
  if (names.length <= maxVisible) {
    return names.join(", ");
  }
  const visible = names.slice(0, maxVisible).join(", ");
  return `${visible} (+${names.length - maxVisible})`;
}

function buildRejectionMessage(rejectedFolders, rejectedFiles) {
  const hasFolders = rejectedFolders.length > 0;
  const hasInvalidFiles = rejectedFiles.length > 0;
  if (!hasFolders && !hasInvalidFiles) {
    return "";
  }
  if (hasFolders && !hasInvalidFiles) {
    return i18n.t("errors.FOLDER_NOT_ALLOWED");
  }
  const filesSummary = formatRejectedNames(rejectedFiles);
  if (hasFolders && hasInvalidFiles) {
    return i18n.t("errors.FOLDER_AND_UNSUPPORTED_FILES", { files: filesSummary });
  }
  return i18n.t("errors.UNSUPPORTED_FILES", { files: filesSummary });
}

function extractDroppedFiles(dataTransfer) {
  const items = dataTransfer?.items ? Array.from(dataTransfer.items) : [];
  const candidateFiles = [];
  const rejectedFolders = [];

  if (items.length > 0 && items.some((item) => typeof item.webkitGetAsEntry === "function")) {
    for (const item of items) {
      if (item.kind && item.kind !== "file") {
        continue;
      }
      const entry = typeof item.webkitGetAsEntry === "function" ? item.webkitGetAsEntry() : null;
      if (entry?.isDirectory) {
        rejectedFolders.push(entry.name || "folder");
        continue;
      }
      const file = typeof item.getAsFile === "function" ? item.getAsFile() : null;
      if (file) {
        candidateFiles.push(file);
      }
    }
  } else {
    candidateFiles.push(...Array.from(dataTransfer?.files || []));
  }

  return { candidateFiles, rejectedFolders };
}

function handleSelectedFiles(fileList, rejectedFolders = []) {
  const candidates = Array.from(fileList || []);
  const validFiles = [];
  const rejectedFiles = [];

  for (const file of candidates) {
    const ext = getFileExtension(file?.name);
    if (ACCEPTED_EXTENSION_SET.has(ext)) {
      validFiles.push(file);
    } else {
      rejectedFiles.push(file?.name || "?");
    }
  }

  const rejectionMessage = buildRejectionMessage(rejectedFolders, rejectedFiles);

  if (validFiles.length === 0 && rejectionMessage) {
    formState.files = [];
    formState.resolvedEntries = [];
    formState.multitrackRows = [];
    formState.uiMode = "single";
    formState.submitError = "";
    formState.selectionWarning = "";
    formState.inspectKey = "";
    formState.inspectStatus = "error";
    formState.inspectError = rejectionMessage;
    renderComposer();
    return;
  }

  formState.files = validFiles;
  formState.submitError = "";
  formState.selectionWarning = rejectionMessage;
  formState.inspectKey = "";
  formState.inspectStatus = "idle";
  formState.inspectError = "";
  void analyzeSelectedFiles().then(() => renderComposer());
}

function addHotwordsFromRaw(rawText) {
  const parts = String(rawText || "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  if (!parts.length) {
    return false;
  }
  let changed = false;
  for (const term of parts) {
    if (!formState.hotwords.includes(term)) {
      formState.hotwords.push(term);
      changed = true;
    }
  }
  return changed;
}

function upsertReturnedJobs(response) {
  const incoming = [];
  if (Array.isArray(response?.jobs)) {
    incoming.push(...response.jobs);
  } else if (response?.job && response.job.id) {
    incoming.push(response.job);
  } else if (response?.id) {
    incoming.push(response);
  }
  if (!incoming.length) {
    return;
  }

  store.setState((current) => {
    const existing = Array.isArray(current.jobs) ? current.jobs.slice() : [];
    for (const item of incoming) {
      if (!item || !item.id) {
        continue;
      }
      const idx = existing.findIndex((job) => job.id === item.id);
      if (idx === -1) {
        existing.unshift(item);
      } else {
        existing[idx] = { ...existing[idx], ...item };
      }
    }
    return { jobs: existing };
  });
}

async function handleSubmit() {
  if (formState.isSubmitting) {
    return;
  }
  formState.submitError = "";

  if (formState.hotwordDraft.trim()) {
    addHotwordsFromRaw(formState.hotwordDraft);
    formState.hotwordDraft = "";
  }

  if (!formState.files.length) {
    formState.submitError = i18n.t("simple.selected_files", { count: 0 });
    renderComposer();
    return;
  }

  if (formState.inspectStatus === "loading") {
    return;
  }
  if (formState.inspectError || formState.resolvedEntries.length === 0) {
    formState.submitError = formState.inspectError || i18n.t("errors.INVALID_ARCHIVE");
    renderComposer();
    return;
  }

  const activeEntries = getActiveEntries();
  if (activeEntries.length === 0) {
    formState.submitError = i18n.t("simple.no_active_tracks");
    renderComposer();
    return;
  }

  const profiles = getSimpleProfiles();
  const profileId = resolveDefaultProfileId(profiles);
  formState.profileId = profileId;

  const totalTracks = activeEntries.length;
  const backendMode =
    totalTracks === 1
      ? "single"
      : formState.uiMode === "batch"
        ? "batch"
        : "multitrack";

  const overrides = {};
  if (backendMode === "single") {
    const diarizeOn = resolveDiarizeEnabled(profiles, profileId);
    overrides.diarize = diarizeOn;
    if (diarizeOn) {
      const minVal = formState.minSpeakers !== "" ? Number(formState.minSpeakers) : null;
      const maxVal = formState.maxSpeakers !== "" ? Number(formState.maxSpeakers) : null;
      if (minVal !== null && (!Number.isInteger(minVal) || minVal < 1)) {
        formState.submitError = i18n.t("errors.VALIDATION_ERROR");
        renderComposer();
        return;
      }
      if (maxVal !== null && (!Number.isInteger(maxVal) || maxVal < 1)) {
        formState.submitError = i18n.t("errors.VALIDATION_ERROR");
        renderComposer();
        return;
      }
      if (minVal !== null && maxVal !== null && maxVal < minVal) {
        formState.submitError = i18n.t("errors.VALIDATION_ERROR");
        renderComposer();
        return;
      }
      if (minVal !== null) {
        overrides.min_speakers = minVal;
      }
      if (maxVal !== null) {
        overrides.max_speakers = maxVal;
      }
    }
  }

  if (formState.hotwords.length > 0) {
    overrides.extra_hotwords = [...formState.hotwords];
  }

  const activeRows = formState.multitrackRows.filter((row) => row.included !== false);
  if (backendMode === "multitrack" && activeRows.length > 0) {
    const speakerAliases = {};
    const trackOffsets = {};
    activeRows.forEach((row, idx) => {
      const speaker = String(row.speaker_name || "").trim();
      const offset = Math.max(0, Math.round(Number(row.offset_ms || 0)));
      if (speaker) {
        if (row.filename) {
          speakerAliases[row.filename] = speaker;
        }
        speakerAliases[String(idx)] = speaker;
      }
      if (row.filename) {
        trackOffsets[row.filename] = offset;
      }
      trackOffsets[String(idx)] = offset;
    });
    if (Object.keys(speakerAliases).length > 0) {
      overrides.speaker_aliases = speakerAliases;
    }
    if (Object.keys(trackOffsets).length > 0) {
      overrides.track_offsets = trackOffsets;
    }
  }

  const excludedZipPaths = getExcludedZipTracks()
    .map((entry) => entry.zipPath)
    .filter(Boolean);
  if (excludedZipPaths.length > 0) {
    overrides.excluded_tracks = excludedZipPaths;
  }

  if (backendMode === "multitrack") {
    overrides.preprocessing = {
      ...(overrides.preprocessing || {}),
      generate_mix: resolveGenerateMix(profiles, profileId),
    };
  }

  const fd = new FormData();
  for (const file of formState.files) {
    fd.append("files", file);
  }
  fd.append("mode", backendMode);
  if (profileId) {
    fd.append("profile_id", profileId);
  }
  if (formState.title.trim()) {
    fd.append("title", formState.title.trim());
  }
  if (Object.keys(overrides).length > 0) {
    fd.append("overrides_json", JSON.stringify(overrides));
  }

  formState.isSubmitting = true;
  renderComposer();

  try {
    const result = await api.createJob(fd);
    upsertReturnedJobs(result);

    formState.files = [];
    formState.resolvedEntries = [];
    formState.title = "";
    formState.generateMix = null;
    formState.diarizeEnabled = null;
    formState.hotwords = [];
    formState.hotwordDraft = "";
    formState.multitrackRows = [];
    formState.inspectKey = "";
    formState.inspectStatus = "idle";
    formState.inspectError = "";
    formState.selectionWarning = "";
    formState.submitError = "";

    store.addToast({
      type: "success",
      title: i18n.t("simple.submit"),
      message: i18n.t("stages.queued"),
    });

    monitorHost?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    const baseMsg = i18n.translateError(err);
    const detail = typeof err?.detail === "string" ? err.detail.trim() : "";
    const msg = detail && !baseMsg.includes(detail) ? `${baseMsg} (${detail})` : baseMsg;
    formState.submitError = msg;
    store.addToast({
      type: "danger",
      title: i18n.t("common.error"),
      message: msg,
    });
  } finally {
    formState.isSubmitting = false;
    renderComposer();
    renderMonitor();
  }
}

async function handleCancelJob(jobId) {
  if (!jobId || formState.cancellingIds.has(jobId)) {
    return;
  }
  formState.cancellingIds.add(jobId);
  renderMonitor();
  try {
    const updated = await api.cancelJob(jobId);
    upsertReturnedJobs(updated);
  } catch (err) {
    store.addToast({
      type: "danger",
      title: i18n.t("common.error"),
      message: i18n.translateError(err),
    });
  } finally {
    formState.cancellingIds.delete(jobId);
    renderMonitor();
  }
}

function createStageBadge(status) {
  const badge = document.createElement("span");
  const normalized = String(status || "queued");
  let variant = "badge-info";
  if (normalized === "completed") {
    variant = "badge-success";
  } else if (normalized === "completed_partial" || normalized === "cancelling") {
    variant = "badge-warning";
  } else if (normalized === "failed") {
    variant = "badge-danger";
  } else if (normalized === "cancelled" || normalized === "queued") {
    variant = "";
  }
  badge.className = variant ? `badge ${variant}` : "badge";
  const key = `stages.${normalized}`;
  const translated = i18n.t(key);
  badge.textContent = translated === key ? normalized : translated;
  return badge;
}

function buildDropzoneBlock() {
  const block = document.createElement("div");
  block.className = "stack";

  const dropLabel = document.createElement("label");
  dropLabel.className = "dropzone";
  dropLabel.setAttribute("for", "simple-file-input");

  const titleSpan = document.createElement("span");
  titleSpan.className = "dropzone-title";
  titleSpan.textContent = i18n.t("simple.dropzone_title");

  const hintSpan = document.createElement("span");
  hintSpan.className = "dropzone-hint";
  hintSpan.textContent = i18n.t("simple.dropzone_hint");

  const browseBtn = document.createElement("span");
  browseBtn.className = "btn btn-secondary btn-sm";
  browseBtn.textContent = i18n.t("simple.select_files");

  const fileInput = document.createElement("input");
  fileInput.id = "simple-file-input";
  fileInput.className = "sr-only";
  fileInput.type = "file";
  fileInput.multiple = true;
  fileInput.accept = ACCEPTED_EXTENSIONS;

  fileInput.addEventListener("change", (event) => {
    const selected = event.target.files ? Array.from(event.target.files) : [];
    fileInput.value = "";
    if (selected.length > 0) {
      handleSelectedFiles(selected);
    }
  });

  dropLabel.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dropLabel.classList.add("is-dragover");
  });
  dropLabel.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropLabel.classList.add("is-dragover");
  });
  dropLabel.addEventListener("dragleave", (event) => {
    if (!dropLabel.contains(event.relatedTarget)) {
      dropLabel.classList.remove("is-dragover");
    }
  });
  dropLabel.addEventListener("drop", (event) => {
    event.preventDefault();
    dropLabel.classList.remove("is-dragover");
    const { candidateFiles, rejectedFolders } = extractDroppedFiles(event.dataTransfer);
    if (candidateFiles.length > 0 || rejectedFolders.length > 0) {
      handleSelectedFiles(candidateFiles, rejectedFolders);
    }
  });

  dropLabel.appendChild(titleSpan);
  dropLabel.appendChild(hintSpan);
  dropLabel.appendChild(browseBtn);
  dropLabel.appendChild(fileInput);
  block.appendChild(dropLabel);

  if (formState.files.length > 0) {
    const summary = document.createElement("div");
    summary.className = "help-text";
    summary.textContent = i18n.t("simple.selected_files", {
      count: formState.files.length,
    });
    block.appendChild(summary);

    const list = document.createElement("div");
    list.className = "file-list";

    formState.files.forEach((file, index) => {
      const item = document.createElement("div");
      item.className = "file-item";

      const info = document.createElement("div");
      info.className = "row";
      info.style.gap = "0.5rem";

      const nameEl = document.createElement("span");
      nameEl.className = "mono";
      nameEl.textContent = file.name;

      const sizeEl = document.createElement("span");
      sizeEl.className = "muted";
      sizeEl.textContent = formatBytes(file.size);

      info.appendChild(nameEl);
      info.appendChild(sizeEl);

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "btn btn-ghost btn-sm";
      removeBtn.textContent = "×";
      removeBtn.setAttribute("aria-label", i18n.t("common.delete"));
      removeBtn.addEventListener("click", () => {
        const remaining = formState.files.filter((_, idx) => idx !== index);
        handleSelectedFiles(remaining);
      });

      item.appendChild(info);
      item.appendChild(removeBtn);
      list.appendChild(item);
    });

    block.appendChild(list);
  }

  if (formState.inspectStatus === "loading") {
    const loadingEl = document.createElement("div");
    loadingEl.className = "help-text";
    loadingEl.textContent = i18n.t("common.loading");
    block.appendChild(loadingEl);
  } else if (formState.inspectStatus === "error" && formState.inspectError) {
    const errEl = document.createElement("div");
    errEl.className = "form-error";
    errEl.textContent = formState.inspectError;
    block.appendChild(errEl);
  } else if (formState.selectionWarning) {
    const warnEl = document.createElement("div");
    warnEl.className = "alert alert-warning";
    warnEl.textContent = formState.selectionWarning;
    block.appendChild(warnEl);
  }

  return block;
}

function buildModeSelectorBlock() {
  const activeCount = getActiveEntries().length;
  if (!formState.files.length || activeCount === 0) {
    return null;
  }

  const profiles = getSimpleProfiles();
  const activeProfileId = resolveDefaultProfileId(profiles);

  if (activeCount === 1) {
    const block = document.createElement("div");
    block.className = "form-group";

    const diarizeOn = resolveDiarizeEnabled(profiles, activeProfileId);
    const toggleLabel = document.createElement("label");
    toggleLabel.className = "checkbox-label";

    const toggleInput = document.createElement("input");
    toggleInput.id = "simple-diarize-toggle";
    toggleInput.type = "checkbox";
    toggleInput.checked = diarizeOn;
    toggleInput.addEventListener("change", (event) => {
      formState.diarizeEnabled = Boolean(event.target.checked);
      renderComposer();
    });

    const toggleText = document.createElement("span");
    toggleText.textContent = i18n.t("simple.modes.diarize.title");

    toggleLabel.appendChild(toggleInput);
    toggleLabel.appendChild(toggleText);
    block.appendChild(toggleLabel);

    if (diarizeOn) {
      const speakerGrid = document.createElement("div");
      speakerGrid.className = "form-grid";
      speakerGrid.style.marginTop = "0.5rem";

      const minGroup = document.createElement("div");
      minGroup.className = "form-group";
      const minLabel = document.createElement("label");
      minLabel.className = "label";
      minLabel.setAttribute("for", "simple-min-speakers");
      minLabel.textContent = `${i18n.t("simple.min_speakers")} (${i18n.t("common.optional")})`;
      const minInput = document.createElement("input");
      minInput.id = "simple-min-speakers";
      minInput.className = "input";
      minInput.type = "number";
      minInput.min = "1";
      minInput.max = "32";
      minInput.value = formState.minSpeakers;
      minInput.addEventListener("input", (event) => {
        formState.minSpeakers = event.target.value;
      });
      minGroup.appendChild(minLabel);
      minGroup.appendChild(minInput);

      const maxGroup = document.createElement("div");
      maxGroup.className = "form-group";
      const maxLabel = document.createElement("label");
      maxLabel.className = "label";
      maxLabel.setAttribute("for", "simple-max-speakers");
      maxLabel.textContent = `${i18n.t("simple.max_speakers")} (${i18n.t("common.optional")})`;
      const maxInput = document.createElement("input");
      maxInput.id = "simple-max-speakers";
      maxInput.className = "input";
      maxInput.type = "number";
      maxInput.min = "1";
      maxInput.max = "32";
      maxInput.value = formState.maxSpeakers;
      maxInput.addEventListener("input", (event) => {
        formState.maxSpeakers = event.target.value;
      });
      maxGroup.appendChild(maxLabel);
      maxGroup.appendChild(maxInput);

      speakerGrid.appendChild(minGroup);
      speakerGrid.appendChild(maxGroup);
      block.appendChild(speakerGrid);
    }

    return block;
  }

  const block = document.createElement("div");
  block.className = "form-group";

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = i18n.t("simple.mode_label");
  block.appendChild(label);

  const grid = document.createElement("div");
  grid.className = "mode-grid";

  for (const modeId of MULTI_MODE_IDS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className =
      formState.uiMode === modeId ? "mode-card is-selected" : "mode-card";
    btn.setAttribute("aria-pressed", formState.uiMode === modeId ? "true" : "false");

    const title = document.createElement("span");
    title.className = "mode-title";
    title.textContent = i18n.t(`simple.modes.${modeId}.title`);

    const desc = document.createElement("span");
    desc.className = "mode-desc";
    desc.textContent = i18n.t(`simple.modes.${modeId}.description`);

    btn.appendChild(title);
    btn.appendChild(desc);

    btn.addEventListener("click", () => {
      if (formState.uiMode === modeId) {
        return;
      }
      formState.uiMode = modeId;
      formState.submitError = "";
      renderComposer();
    });

    grid.appendChild(btn);
  }

  block.appendChild(grid);
  return block;
}

function buildMultitrackTableBlock() {
  if (formState.uiMode !== "multitrack" || !formState.files.length) {
    return null;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "stack";

  if (formState.inspectStatus === "loading") {
    const loadingAlert = document.createElement("div");
    loadingAlert.className = "alert alert-info";
    loadingAlert.textContent = i18n.t("common.loading");
    wrapper.appendChild(loadingAlert);
    return wrapper;
  }

  if (formState.inspectStatus === "error" && formState.inspectError) {
    const errAlert = document.createElement("div");
    errAlert.className = "alert alert-danger";
    errAlert.textContent = formState.inspectError;
    wrapper.appendChild(errAlert);
    return wrapper;
  }

  const activeRows = formState.multitrackRows.filter((row) => row.included !== false);
  if (!activeRows.length) {
    return null;
  }

  const tableWrap = document.createElement("div");
  tableWrap.className = "table-wrap";

  const table = document.createElement("table");
  table.className = "table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const key of [
    "simple.multitrack_table.filename",
    "simple.multitrack_table.speaker_name",
    "simple.multitrack_table.offset_ms",
  ]) {
    const th = document.createElement("th");
    th.textContent = i18n.t(key);
    headRow.appendChild(th);
  }
  const actionTh = document.createElement("th");
  actionTh.setAttribute("aria-label", i18n.t("simple.discard_track"));
  headRow.appendChild(actionTh);
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  activeRows.forEach((row) => {
    const tr = document.createElement("tr");

    const fileCell = document.createElement("td");
    fileCell.className = "mono";
    fileCell.textContent = row.filename;

    const speakerCell = document.createElement("td");
    const speakerInput = document.createElement("input");
    speakerInput.type = "text";
    speakerInput.className = "input";
    speakerInput.value = row.speaker_name;
    speakerInput.setAttribute(
      "aria-label",
      `${i18n.t("simple.multitrack_table.speaker_name")} (${row.filename})`
    );
    speakerInput.addEventListener("input", (event) => {
      row.speaker_name = event.target.value;
    });
    speakerCell.appendChild(speakerInput);

    const offsetCell = document.createElement("td");
    const offsetInput = document.createElement("input");
    offsetInput.type = "number";
    offsetInput.min = "0";
    offsetInput.step = "100";
    offsetInput.className = "input";
    offsetInput.value = String(row.offset_ms ?? 0);
    offsetInput.setAttribute(
      "aria-label",
      `${i18n.t("simple.multitrack_table.offset_ms")} (${row.filename})`
    );
    offsetInput.addEventListener("input", (event) => {
      const parsed = Number(event.target.value);
      row.offset_ms =
        Number.isFinite(parsed) && parsed >= 0 ? Math.round(parsed) : 0;
    });
    offsetCell.appendChild(offsetInput);

    const actionCell = document.createElement("td");
    if (row.sourceType === "zip") {
      const discardBtn = document.createElement("button");
      discardBtn.type = "button";
      discardBtn.className = "btn btn-ghost btn-sm";
      discardBtn.textContent = "×";
      discardBtn.title = i18n.t("simple.discard_track");
      discardBtn.setAttribute(
        "aria-label",
        `${i18n.t("simple.discard_track")}: ${row.filename}`
      );
      discardBtn.addEventListener("click", () => {
        setEntryIncluded(row.id, false);
      });
      actionCell.appendChild(discardBtn);
    }

    tr.appendChild(fileCell);
    tr.appendChild(speakerCell);
    tr.appendChild(offsetCell);
    tr.appendChild(actionCell);
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  tableWrap.appendChild(table);
  wrapper.appendChild(tableWrap);
  return wrapper;
}

function buildZipTracksBlock() {
  if (!formState.files.length || formState.inspectStatus !== "done") {
    return null;
  }

  const zipEntries = formState.resolvedEntries.filter(
    (entry) => entry.sourceType === "zip"
  );
  if (!zipEntries.length) {
    return null;
  }

  const activeZipTracks = zipEntries.filter((entry) => entry.included !== false);
  const excludedZipTracks = zipEntries.filter((entry) => entry.included === false);
  const showActiveZipList =
    formState.uiMode !== "multitrack" &&
    (formState.uiMode === "batch" || excludedZipTracks.length > 0) &&
    activeZipTracks.length > 0;

  if (!showActiveZipList && excludedZipTracks.length === 0) {
    return null;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "stack";

  if (showActiveZipList) {
    const group = document.createElement("div");
    group.className = "form-group";

    const label = document.createElement("span");
    label.className = "label";
    label.textContent = i18n.t("simple.zip_tracks_label");
    group.appendChild(label);

    const list = document.createElement("div");
    list.className = "file-list";

    for (const entry of activeZipTracks) {
      const item = document.createElement("div");
      item.className = "file-item";

      const nameEl = document.createElement("span");
      nameEl.className = "mono";
      nameEl.textContent = entry.zipPath || entry.filename;

      const discardBtn = document.createElement("button");
      discardBtn.type = "button";
      discardBtn.className = "btn btn-ghost btn-sm";
      discardBtn.textContent = "×";
      discardBtn.title = i18n.t("simple.discard_track");
      discardBtn.setAttribute(
        "aria-label",
        `${i18n.t("simple.discard_track")}: ${entry.filename}`
      );
      discardBtn.addEventListener("click", () => {
        setEntryIncluded(entry.id, false);
      });

      item.appendChild(nameEl);
      item.appendChild(discardBtn);
      list.appendChild(item);
    }

    group.appendChild(list);
    wrapper.appendChild(group);
  }

  if (excludedZipTracks.length > 0) {
    const group = document.createElement("div");
    group.className = "form-group";

    const label = document.createElement("span");
    label.className = "label";
    label.textContent = i18n.t("simple.excluded_tracks_label");
    group.appendChild(label);

    const list = document.createElement("div");
    list.className = "file-list";

    for (const entry of excludedZipTracks) {
      const item = document.createElement("div");
      item.className = "file-item";

      const nameEl = document.createElement("span");
      nameEl.className = "mono muted";
      nameEl.textContent = entry.zipPath || entry.filename;

      const restoreBtn = document.createElement("button");
      restoreBtn.type = "button";
      restoreBtn.className = "btn btn-ghost btn-sm";
      restoreBtn.textContent = i18n.t("simple.restore_track");
      restoreBtn.addEventListener("click", () => {
        setEntryIncluded(entry.id, true);
      });

      item.appendChild(nameEl);
      item.appendChild(restoreBtn);
      list.appendChild(item);
    }

    group.appendChild(list);
    wrapper.appendChild(group);
  }

  return wrapper;
}

function buildProfileAndTitleBlock() {
  const grid = document.createElement("div");
  grid.className = "form-grid";

  const profiles = getSimpleProfiles();
  const activeProfileId = resolveDefaultProfileId(profiles);
  formState.profileId = activeProfileId;

  const profileGroup = document.createElement("div");
  profileGroup.className = "form-group";

  const profileLabel = document.createElement("label");
  profileLabel.className = "label";
  profileLabel.setAttribute("for", "simple-profile-select");
  profileLabel.textContent = i18n.t("simple.profile_label");

  const profileSelect = document.createElement("select");
  profileSelect.id = "simple-profile-select";
  profileSelect.className = "select";

  for (const profile of profiles) {
    const opt = document.createElement("option");
    opt.value = profile.id;
    opt.textContent = getProfileLabel(profile);
    if (profile.id === activeProfileId) {
      opt.selected = true;
    }
    profileSelect.appendChild(opt);
  }

  profileSelect.addEventListener("change", (event) => {
    formState.profileId = event.target.value;
    formState.generateMix = null;
    formState.diarizeEnabled = null;
    if (formState.uiMode === "multitrack" || getActiveEntries().length === 1) {
      renderComposer();
    }
  });

  profileGroup.appendChild(profileLabel);
  profileGroup.appendChild(profileSelect);

  const titleGroup = document.createElement("div");
  titleGroup.className = "form-group";

  const titleLabel = document.createElement("label");
  titleLabel.className = "label";
  titleLabel.setAttribute("for", "simple-title-input");
  titleLabel.textContent = i18n.t("simple.title_label");

  const titleInput = document.createElement("input");
  titleInput.id = "simple-title-input";
  titleInput.className = "input";
  titleInput.type = "text";
  titleInput.value = formState.title;
  titleInput.placeholder = i18n.t("simple.title_placeholder");
  titleInput.addEventListener("input", (event) => {
    formState.title = event.target.value;
  });

  titleGroup.appendChild(titleLabel);
  titleGroup.appendChild(titleInput);

  grid.appendChild(profileGroup);
  grid.appendChild(titleGroup);

  if (formState.uiMode === "multitrack") {
    const mixLabel = document.createElement("label");
    mixLabel.className = "checkbox-label";
    mixLabel.style.gridColumn = "1 / -1";

    const mixInput = document.createElement("input");
    mixInput.id = "simple-generate-mix";
    mixInput.type = "checkbox";
    mixInput.checked = resolveGenerateMix(profiles, activeProfileId);
    mixInput.addEventListener("change", (event) => {
      formState.generateMix = Boolean(event.target.checked);
    });

    const mixText = document.createElement("span");
    mixText.textContent = i18n.t("simple.generate_mix");

    mixLabel.appendChild(mixInput);
    mixLabel.appendChild(mixText);
    grid.appendChild(mixLabel);
  }

  return grid;
}

function buildHotwordsBlock() {
  const group = document.createElement("div");
  group.className = "form-group";

  const label = document.createElement("label");
  label.className = "label";
  label.setAttribute("for", "simple-hotwords-input");
  label.textContent = i18n.t("simple.hotwords_label");
  group.appendChild(label);

  const input = document.createElement("input");
  input.id = "simple-hotwords-input";
  input.className = "input";
  input.type = "text";
  input.value = formState.hotwordDraft;
  input.placeholder = i18n.t("simple.hotwords_placeholder");

  const chipsContainer = document.createElement("div");
  chipsContainer.className = "chips";

  const refreshChips = () => {
    chipsContainer.textContent = "";
    formState.hotwords.forEach((word, index) => {
      const chip = document.createElement("span");
      chip.className = "chip";

      const wordSpan = document.createElement("span");
      wordSpan.textContent = word;

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "chip-remove";
      removeBtn.textContent = "×";
      removeBtn.setAttribute("aria-label", `${i18n.t("common.delete")}: ${word}`);
      removeBtn.addEventListener("click", () => {
        formState.hotwords = formState.hotwords.filter((_, idx) => idx !== index);
        refreshChips();
      });

      chip.appendChild(wordSpan);
      chip.appendChild(removeBtn);
      chipsContainer.appendChild(chip);
    });
  };

  input.addEventListener("input", (event) => {
    const val = event.target.value;
    if (val.includes(",")) {
      const segments = val.split(",");
      const remainder = segments.pop() || "";
      addHotwordsFromRaw(segments.join(","));
      formState.hotwordDraft = remainder;
      event.target.value = remainder;
      refreshChips();
    } else {
      formState.hotwordDraft = val;
    }
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      if (addHotwordsFromRaw(input.value)) {
        formState.hotwordDraft = "";
        input.value = "";
        refreshChips();
      }
    } else if (
      event.key === "Backspace" &&
      input.value === "" &&
      formState.hotwords.length > 0
    ) {
      formState.hotwords.pop();
      refreshChips();
    }
  });

  refreshChips();
  group.appendChild(input);
  group.appendChild(chipsContainer);
  return group;
}

function renderComposer() {
  if (!composerHost) {
    return;
  }

  composerHost.textContent = "";

  const header = document.createElement("div");
  header.className = "section-header";
  const headerText = document.createElement("div");
  const title = document.createElement("h2");
  title.className = "section-title";
  title.textContent = i18n.t("simple.title");
  const subtitle = document.createElement("p");
  subtitle.className = "section-subtitle";
  subtitle.textContent = i18n.t("simple.subtitle");
  headerText.appendChild(title);
  headerText.appendChild(subtitle);
  header.appendChild(headerText);
  composerHost.appendChild(header);

  const card = document.createElement("div");
  card.className = "card";

  const body = document.createElement("div");
  body.className = "card-body";
  body.appendChild(buildDropzoneBlock());
  const modeBlock = buildModeSelectorBlock();
  if (modeBlock) {
    body.appendChild(modeBlock);
  }

  const multitrackBlock = buildMultitrackTableBlock();
  if (multitrackBlock) {
    body.appendChild(multitrackBlock);
  }

  const zipTracksBlock = buildZipTracksBlock();
  if (zipTracksBlock) {
    body.appendChild(zipTracksBlock);
  }

  const noActiveTracks =
    formState.files.length > 0 &&
    formState.inspectStatus === "done" &&
    getActiveEntries().length === 0;
  if (noActiveTracks) {
    const emptyAlert = document.createElement("div");
    emptyAlert.className = "form-error";
    emptyAlert.setAttribute("role", "alert");
    emptyAlert.textContent = i18n.t("simple.no_active_tracks");
    body.appendChild(emptyAlert);
  }

  body.appendChild(buildProfileAndTitleBlock());
  body.appendChild(buildHotwordsBlock());

  if (formState.submitError) {
    const errEl = document.createElement("div");
    errEl.className = "form-error";
    errEl.setAttribute("role", "alert");
    errEl.textContent = formState.submitError;
    body.appendChild(errEl);
  }

  const footer = document.createElement("div");
  footer.className = "card-footer";

  const submitBtn = document.createElement("button");
  submitBtn.type = "button";
  submitBtn.id = "simple-submit-btn";
  submitBtn.className = "btn btn-primary";
  submitBtn.disabled =
    formState.isSubmitting ||
    formState.inspectStatus === "loading" ||
    formState.files.length === 0 ||
    noActiveTracks;
  submitBtn.textContent = formState.isSubmitting
    ? i18n.t("simple.submitting")
    : i18n.t("simple.submit");
  submitBtn.addEventListener("click", () => {
    void handleSubmit();
  });

  footer.appendChild(submitBtn);

  card.appendChild(body);
  card.appendChild(footer);
  composerHost.appendChild(card);
}

function formatCountdownMMSS(ms) {
  const totalSec = Math.max(0, Math.ceil(ms / 1000));
  const m = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const s = String(totalSec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function patchStageBadge(badgeEl, stageCode) {
  const code = stageCode || "queued";
  const nextClass = `badge ${STAGE_BADGE_CLASS[code] || "badge-info"}`;
  const nextText = i18n.t(`stages.${code}`);
  if (badgeEl.className !== nextClass) {
    badgeEl.className = nextClass;
  }
  if (badgeEl.textContent !== nextText) {
    badgeEl.textContent = nextText;
  }
}

function resolveDeviceTrackInfo(device, activeJob) {
  const tracks = Array.isArray(activeJob?.tracks) ? activeJob.tracks : [];
  if (!tracks.length) {
    return null;
  }
  let idx = -1;
  if (device.track_id) {
    idx = tracks.findIndex((t) => t.id === device.track_id);
  }
  if (idx === -1 && tracks.length === 1) {
    idx = 0;
  }
  if (idx === -1) {
    return null;
  }
  const track = tracks[idx];
  const label = track.display_name || track.original_filename || track.id;
  return {
    indexLabel: `${idx + 1}/${tracks.length}`,
    name: label,
  };
}

function createDeviceCardRefs(deviceId) {
  const card = document.createElement("div");
  card.className = "card";
  card.dataset.deviceId = deviceId;

  const header = document.createElement("div");
  header.className = "card-header";

  const left = document.createElement("div");
  left.style.minWidth = "0";
  const title = document.createElement("h3");
  title.className = "card-title mono text-truncate";
  const subtitle = document.createElement("p");
  subtitle.className = "card-subtitle text-truncate";
  left.appendChild(title);
  left.appendChild(subtitle);

  const kindBadge = document.createElement("span");
  header.appendChild(left);
  header.appendChild(kindBadge);
  card.appendChild(header);

  const body = document.createElement("div");
  body.className = "card-body";

  const busyBox = document.createElement("div");
  busyBox.className = "stack";
  busyBox.style.gap = "0.55rem";

  const jobBlock = document.createElement("div");
  jobBlock.className = "device-job-block";

  const jobTitleRow = document.createElement("div");
  jobTitleRow.className = "device-job-title-row";
  const nameSpan = document.createElement("strong");
  nameSpan.className = "device-job-name text-truncate";
  const stageBadge = document.createElement("span");
  jobTitleRow.appendChild(nameSpan);
  jobTitleRow.appendChild(stageBadge);

  const trackRow = document.createElement("div");
  trackRow.className = "device-track-row";
  const trackPill = document.createElement("span");
  trackPill.className = "track-pill";
  const trackNameEl = document.createElement("span");
  trackNameEl.className = "device-track-name text-truncate";
  trackRow.appendChild(trackPill);
  trackRow.appendChild(trackNameEl);

  jobBlock.appendChild(jobTitleRow);
  jobBlock.appendChild(trackRow);
  busyBox.appendChild(jobBlock);

  const bar = document.createElement("div");
  bar.className = "progress-bar";
  const fill = document.createElement("div");
  fill.className = "progress-fill is-active";
  bar.appendChild(fill);
  busyBox.appendChild(bar);

  const meta = document.createElement("div");
  meta.className = "progress-meta";
  const pctEl = document.createElement("span");
  const statsEl = document.createElement("span");
  meta.appendChild(pctEl);
  meta.appendChild(statsEl);
  busyBox.appendChild(meta);

  const actions = document.createElement("div");
  actions.className = "row";
  actions.style.justifyContent = "flex-end";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "btn btn-danger btn-sm";
  cancelBtn.textContent = i18n.t("common.cancel");
  const refs = {
    card,
    title,
    subtitle,
    kindBadge,
    busyBox,
    idleBox: null,
    idleText: null,
    nameSpan,
    stageBadge,
    trackRow,
    trackPill,
    trackNameEl,
    fill,
    pctEl,
    statsEl,
    actions,
    cancelBtn,
    currentJobId: null,
  };
  cancelBtn.addEventListener("click", () => {
    if (refs.currentJobId) {
      void handleCancelJob(refs.currentJobId);
    }
  });
  actions.appendChild(cancelBtn);
  busyBox.appendChild(actions);

  const idleBox = document.createElement("div");
  const idleText = document.createElement("div");
  idleText.className = "help-text";
  idleBox.appendChild(idleText);
  refs.idleBox = idleBox;
  refs.idleText = idleText;

  body.appendChild(busyBox);
  body.appendChild(idleBox);
  card.appendChild(body);
  return refs;
}

function patchDeviceCardRefs(refs, device, jobsById) {
  const devTitle = device.id || (device.kind === "cuda" ? "cuda" : "cpu");
  if (refs.title.textContent !== devTitle) {
    refs.title.textContent = devTitle;
  }
  const devSub =
    device.name || (device.kind === "cuda" ? i18n.t("common.cuda") : i18n.t("common.cpu"));
  if (refs.subtitle.textContent !== devSub) {
    refs.subtitle.textContent = devSub;
  }
  const kindClass = device.kind === "cuda" ? "badge badge-gpu" : "badge badge-cpu";
  const kindText = device.kind === "cuda" ? i18n.t("common.cuda") : i18n.t("common.cpu");
  if (refs.kindBadge.className !== kindClass) {
    refs.kindBadge.className = kindClass;
  }
  if (refs.kindBadge.textContent !== kindText) {
    refs.kindBadge.textContent = kindText;
  }

  const isBusy = Boolean(device.active_job_id);
  refs.busyBox.hidden = !isBusy;
  refs.idleBox.hidden = isBusy;

  if (!isBusy) {
    refs.currentJobId = null;
    refs.fill.style.width = "0%";
    const nextIdle =
      device.vram_total_mb && device.vram_free_mb !== null && device.vram_free_mb !== undefined
        ? i18n.t("settings.system.vram_usage", {
            free: device.vram_free_mb,
            total: device.vram_total_mb,
          })
        : i18n.t("simple.queue_empty");
    if (refs.idleText.textContent !== nextIdle) {
      refs.idleText.textContent = nextIdle;
    }
    return;
  }

  const activeJob = jobsById.get(device.active_job_id) || null;
  refs.currentJobId = device.active_job_id;
  const jobTitle = activeJob?.title || device.active_job_id;
  if (refs.nameSpan.textContent !== jobTitle) {
    refs.nameSpan.textContent = jobTitle;
    refs.nameSpan.title = jobTitle;
  }

  const stageCode = device.stage || activeJob?.stage || activeJob?.status || "transcribing";
  patchStageBadge(refs.stageBadge, stageCode);

  const trackInfo = resolveDeviceTrackInfo(device, activeJob);
  if (trackInfo) {
    refs.trackRow.hidden = false;
    const pillText = `Track ${trackInfo.indexLabel}`;
    if (refs.trackPill.textContent !== pillText) {
      refs.trackPill.textContent = pillText;
    }
    if (refs.trackNameEl.textContent !== trackInfo.name) {
      refs.trackNameEl.textContent = trackInfo.name;
      refs.trackNameEl.title = trackInfo.name;
    }
  } else {
    refs.trackRow.hidden = true;
  }

  const trackProgress =
    device.track_id && activeJob?.track_progress
      ? activeJob.track_progress[device.track_id]
      : null;
  const fallbackProgress = activeJob?.progress || {};
  const processed = device.processed_ms ?? trackProgress?.processed_ms ?? fallbackProgress.processed_ms ?? 0;
  const total = device.total_ms ?? trackProgress?.total_ms ?? fallbackProgress.total_ms ?? 0;
  const percent =
    typeof device.percent === "number" && device.percent > 0
      ? device.percent
      : typeof trackProgress?.percent === "number"
        ? trackProgress.percent
        : typeof fallbackProgress.percent === "number"
          ? fallbackProgress.percent
          : total > 0
            ? Math.min(100, Math.round((processed / total) * 100))
            : 0;
  const clampedPct = Math.max(0, Math.min(100, percent));

  refs.fill.style.width = `${clampedPct}%`;
  refs.fill.className = "progress-fill is-active";

  const pctStr = `${clampedPct}%`;
  if (refs.pctEl.textContent !== pctStr) {
    refs.pctEl.textContent = pctStr;
  }

  const rtfx = device.rtfx ?? trackProgress?.rtfx ?? fallbackProgress.rtfx ?? null;
  const eta = computeEta(processed, total, rtfx);
  const statsParts = [];
  if (rtfx !== null && rtfx !== undefined && Number(rtfx) > 0) {
    statsParts.push(`RTFx ${Number(rtfx).toFixed(1)}x`);
  }
  if (eta) {
    statsParts.push(`ETA ${eta}`);
  }
  const statsStr = statsParts.join(" · ");
  if (refs.statsEl.textContent !== statsStr) {
    refs.statsEl.textContent = statsStr;
  }

  refs.actions.hidden = stageCode === "cancelling";
  refs.cancelBtn.disabled = formState.cancellingIds.has(device.active_job_id);
  const cancelLabel = i18n.t("common.cancel");
  if (refs.cancelBtn.textContent !== cancelLabel) {
    refs.cancelBtn.textContent = cancelLabel;
  }
}

function getJobFinishedMs(job) {
  if (typeof job.finished_at_ms === "number" && job.finished_at_ms > 0) {
    return job.finished_at_ms;
  }
  const ts = job.completed_at || job.updated_at;
  if (!ts) {
    return 0;
  }
  const parsed = Date.parse(ts);
  return Number.isFinite(parsed) ? parsed : 0;
}

function buildQueueGroups(jobs, nowMs) {
  const groups = [];
  const batchBuckets = new Map();

  for (const job of jobs) {
    if (job.mode === "batch" && job.batch_id) {
      const key = `batch:${job.batch_id}`;
      let list = batchBuckets.get(key);
      if (!list) {
        list = [];
        batchBuckets.set(key, list);
        groups.push({ key, kind: "batch", jobs: list });
      }
      list.push(job);
    } else {
      groups.push({ key: `job:${job.id}`, kind: job.mode || "single", jobs: [job] });
    }
  }

  const visible = [];
  for (const g of groups) {
    if (dismissedGroupKeys.has(g.key)) {
      continue;
    }

    const hasActive = g.jobs.some((j) => ACTIVE_STATUSES.has(j.status));
    const allTerminal = g.jobs.every((j) => TERMINAL_STATUSES.has(j.status));

    let finishedAtMs = 0;
    if (allTerminal) {
      finishedAtMs = g.jobs.reduce((max, j) => Math.max(max, getJobFinishedMs(j)), 0);
      if (!finishedAtMs || nowMs - finishedAtMs >= QUEUE_TTL_MS) {
        continue;
      }
    }

    const childTracks = [];
    if (g.kind === "batch" && g.jobs.length > 1) {
      g.jobs.forEach((job, idx) => {
        const isTerminal = TERMINAL_STATUSES.has(job.status);
        const isQueued = job.status === "queued";
        const progress = job.progress || {};
        const processed = Number(progress.processed_ms || 0);
        const total = Number(progress.total_ms || 0);
        let pct = 0;
        if (job.status === "completed" || job.status === "completed_partial") {
          pct = 100;
        } else if (!isQueued && !isTerminal) {
          pct =
            typeof progress.percent === "number"
              ? progress.percent
              : total > 0
                ? Math.min(100, Math.round((processed / total) * 100))
                : 0;
        }
        const statusCode = isQueued ? "queued" : job.stage || job.status || "queued";
        childTracks.push({
          key: `j:${job.id}`,
          indexLabel: `${idx + 1}/${g.jobs.length}`,
          name: job.title || job.id,
          deviceId: isQueued ? null : progress.device_id || job.device_id || null,
          statusCode,
          percent: Math.max(0, Math.min(100, pct)),
          rtfx: progress.rtfx ?? null,
          eta: !isQueued && !isTerminal ? computeEta(processed, total, progress.rtfx ?? null) : null,
          isTerminal,
          isSuccess: job.status === "completed" || job.status === "completed_partial",
          isFailed: job.status === "failed" || job.status === "cancelled",
        });
      });
    } else {
      const job = g.jobs[0];
      const tracks = Array.isArray(job.tracks) ? job.tracks : [];
      const trackProgressMap = job.track_progress || {};
      if (tracks.length > 1) {
        tracks.forEach((track, idx) => {
          const tp = trackProgressMap[track.id] || null;
          const trackDone =
            track.status === "completed" ||
            tp?.stage === "completed" ||
            job.status === "completed" ||
            job.status === "completed_partial";
          const isRunning =
            !trackDone &&
            tp &&
            (tp.stage === "transcribing" || tp.device_id || Number(tp.percent || 0) > 0);
          const pct = trackDone ? 100 : isRunning ? Number(tp.percent || 0) : 0;
          const statusCode = trackDone ? "completed" : isRunning ? tp.stage || "transcribing" : "queued";
          childTracks.push({
            key: `t:${track.id}`,
            indexLabel: `${idx + 1}/${tracks.length}`,
            name: track.display_name || track.original_filename || track.id,
            deviceId: isRunning || trackDone ? tp?.device_id || null : null,
            statusCode,
            percent: Math.max(0, Math.min(100, pct)),
            rtfx: isRunning ? tp?.rtfx ?? null : null,
            eta: isRunning ? computeEta(tp?.processed_ms || 0, tp?.total_ms || 0, tp?.rtfx ?? null) : null,
            isTerminal: trackDone,
            isSuccess: trackDone,
            isFailed: false,
          });
        });
      } else {
        const isTerminal = TERMINAL_STATUSES.has(job.status);
        const isQueued = job.status === "queued";
        const progress = job.progress || {};
        const processed = Number(progress.processed_ms || 0);
        const total = Number(progress.total_ms || 0);
        const pct =
          job.status === "completed" || job.status === "completed_partial"
            ? 100
            : isQueued || isTerminal
              ? 0
              : typeof progress.percent === "number"
                ? progress.percent
                : total > 0
                  ? Math.min(100, Math.round((processed / total) * 100))
                  : 0;
        childTracks.push({
          key: `j:${job.id}`,
          indexLabel: "1/1",
          name: job.title || job.id,
          deviceId: isQueued ? null : progress.device_id || job.device_id || null,
          statusCode: isQueued ? "queued" : job.stage || job.status || "queued",
          percent: Math.max(0, Math.min(100, pct)),
          rtfx: progress.rtfx ?? null,
          eta: !isQueued && !isTerminal ? computeEta(processed, total, progress.rtfx ?? null) : null,
          isTerminal,
          isSuccess: job.status === "completed" || job.status === "completed_partial",
          isFailed: job.status === "failed" || job.status === "cancelled",
        });
      }
    }

    const globalPct = childTracks.length
      ? Math.round(childTracks.reduce((acc, t) => acc + t.percent, 0) / childTracks.length)
      : 0;

    const primaryJob = g.jobs[0];
    const groupStatus = hasActive
      ? g.jobs.find((j) => ACTIVE_STATUSES.has(j.status) && j.status !== "queued")?.stage ||
        g.jobs.find((j) => ACTIVE_STATUSES.has(j.status))?.status ||
        "queued"
      : primaryJob.status;

    const groupTitle =
      g.kind === "batch" && g.jobs.length > 1
        ? `${primaryJob.title || "Batch"} (${g.jobs.length} files)`
        : primaryJob.title || primaryJob.id;

    visible.push({
      key: g.key,
      kind: g.kind,
      jobs: g.jobs,
      primaryJob,
      title: groupTitle,
      mode: primaryJob.mode || "single",
      statusCode: groupStatus,
      hasActive,
      allTerminal,
      finishedAtMs,
      globalPct,
      childTracks,
    });
  }

  return visible.slice(0, 10);
}

function createTrackRowRefs(trackKey) {
  const row = document.createElement("div");
  row.className = "queue-track-row";
  row.dataset.trackKey = trackKey;

  const nameCol = document.createElement("div");
  nameCol.className = "queue-track-name-col";
  const indexEl = document.createElement("span");
  indexEl.className = "track-pill";
  const filenameEl = document.createElement("span");
  filenameEl.className = "mono text-truncate";
  filenameEl.style.fontSize = "0.78rem";
  nameCol.appendChild(indexEl);
  nameCol.appendChild(filenameEl);

  const deviceCol = document.createElement("div");
  const deviceBadge = document.createElement("span");
  deviceBadge.className = "badge mono";
  deviceCol.appendChild(deviceBadge);

  const progCol = document.createElement("div");
  progCol.className = "queue-track-progress-col";
  const bar = document.createElement("div");
  bar.className = "progress-bar sm";
  const fill = document.createElement("div");
  fill.className = "progress-fill";
  bar.appendChild(fill);
  const pctEl = document.createElement("span");
  pctEl.className = "queue-track-pct";
  progCol.appendChild(bar);
  progCol.appendChild(pctEl);

  const metricsEl = document.createElement("div");
  metricsEl.className = "queue-track-metrics";

  const statusCol = document.createElement("div");
  const statusBadge = document.createElement("span");
  statusCol.appendChild(statusBadge);

  row.appendChild(nameCol);
  row.appendChild(deviceCol);
  row.appendChild(progCol);
  row.appendChild(metricsEl);
  row.appendChild(statusCol);

  return {
    row,
    nameCol,
    indexEl,
    filenameEl,
    deviceBadge,
    fill,
    pctEl,
    metricsEl,
    statusBadge,
  };
}

function patchTrackRowRefs(refs, track) {
  if (refs.indexEl.textContent !== track.indexLabel) {
    refs.indexEl.textContent = track.indexLabel;
  }
  if (refs.filenameEl.textContent !== track.name) {
    refs.filenameEl.textContent = track.name;
    refs.nameCol.title = track.name;
  }

  const devText = track.deviceId || "—";
  const devClass = track.deviceId ? "badge badge-info mono" : "badge mono";
  if (refs.deviceBadge.className !== devClass) {
    refs.deviceBadge.className = devClass;
  }
  if (refs.deviceBadge.textContent !== devText) {
    refs.deviceBadge.textContent = devText;
  }

  const fillClass = track.isSuccess
    ? "progress-fill is-success"
    : track.isFailed
      ? "progress-fill is-danger"
      : track.statusCode === "queued"
        ? "progress-fill is-queued"
        : "progress-fill is-active";
  if (refs.fill.className !== fillClass) {
    refs.fill.className = fillClass;
  }
  refs.fill.style.width = `${track.percent}%`;

  const pctStr = `${track.percent}%`;
  if (refs.pctEl.textContent !== pctStr) {
    refs.pctEl.textContent = pctStr;
  }

  let metricsStr = i18n.t(`stages.${track.statusCode}`);
  if (track.statusCode === "transcribing") {
    const parts = [];
    if (track.rtfx !== null && track.rtfx !== undefined && Number(track.rtfx) > 0) {
      parts.push(`${Number(track.rtfx).toFixed(1)}x`);
    }
    if (track.eta) {
      parts.push(`ETA ${track.eta}`);
    }
    metricsStr = parts.length ? parts.join(" · ") : metricsStr;
  }
  if (refs.metricsEl.textContent !== metricsStr) {
    refs.metricsEl.textContent = metricsStr;
  }

  patchStageBadge(refs.statusBadge, track.statusCode);
}

function createQueueGroupRefs(groupKey) {
  const card = document.createElement("div");
  card.className = "queue-group-card";
  card.dataset.groupKey = groupKey;

  const topRow = document.createElement("div");
  topRow.className = "row-between";

  const titleWrap = document.createElement("div");
  titleWrap.className = "row";
  titleWrap.style.gap = "0.5rem";
  titleWrap.style.minWidth = "0";
  titleWrap.style.flex = "1";

  const titleEl = document.createElement("strong");
  titleEl.className = "mono text-truncate";
  const modeBadge = document.createElement("span");
  modeBadge.className = "badge";
  const stageBadge = document.createElement("span");
  const ttlBadge = document.createElement("span");
  ttlBadge.className = "ttl-pill";
  ttlBadge.hidden = true;

  titleWrap.appendChild(titleEl);
  titleWrap.appendChild(modeBadge);
  titleWrap.appendChild(stageBadge);
  titleWrap.appendChild(ttlBadge);

  const headerActions = document.createElement("div");
  headerActions.className = "row";
  headerActions.style.gap = "0.4rem";

  const toggleBtn = document.createElement("button");
  toggleBtn.type = "button";
  toggleBtn.className = "btn btn-sm";
  toggleBtn.hidden = true;

  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "btn btn-danger btn-sm";
  cancelBtn.textContent = i18n.t("common.cancel");

  const openBtn = document.createElement("button");
  openBtn.type = "button";
  openBtn.className = "btn btn-primary btn-sm";
  openBtn.textContent = i18n.t("simple.open_viewer");

  const dismissBtn = document.createElement("button");
  dismissBtn.type = "button";
  dismissBtn.className = "btn btn-sm";
  dismissBtn.textContent = "Dismiss";

  headerActions.appendChild(toggleBtn);
  headerActions.appendChild(cancelBtn);
  headerActions.appendChild(openBtn);
  headerActions.appendChild(dismissBtn);

  topRow.appendChild(titleWrap);
  topRow.appendChild(headerActions);
  card.appendChild(topRow);

  const bar = document.createElement("div");
  bar.className = "progress-bar";
  const fill = document.createElement("div");
  fill.className = "progress-fill";
  bar.appendChild(fill);
  card.appendChild(bar);

  const meta = document.createElement("div");
  meta.className = "progress-meta";
  const summaryEl = document.createElement("span");
  const rightMetaEl = document.createElement("span");
  meta.appendChild(summaryEl);
  meta.appendChild(rightMetaEl);
  card.appendChild(meta);

  const tracksContainer = document.createElement("div");
  tracksContainer.className = "queue-tracks-container";
  tracksContainer.hidden = true;
  card.appendChild(tracksContainer);

  const refs = {
    card,
    titleEl,
    modeBadge,
    stageBadge,
    ttlBadge,
    toggleBtn,
    cancelBtn,
    openBtn,
    dismissBtn,
    fill,
    summaryEl,
    rightMetaEl,
    tracksContainer,
    trackRowsByKey: new Map(),
    currentGroup: null,
  };

  toggleBtn.addEventListener("click", () => {
    if (collapsedGroupKeys.has(groupKey)) {
      collapsedGroupKeys.delete(groupKey);
    } else {
      collapsedGroupKeys.add(groupKey);
    }
    if (refs.currentGroup) {
      patchQueueGroupRefs(refs, refs.currentGroup, Date.now());
    }
  });

  cancelBtn.addEventListener("click", () => {
    const g = refs.currentGroup;
    if (!g) return;
    for (const j of g.jobs) {
      if (ACTIVE_STATUSES.has(j.status) && j.status !== "cancelling") {
        void handleCancelJob(j.id);
      }
    }
  });

  openBtn.addEventListener("click", () => {
    const g = refs.currentGroup;
    if (!g?.primaryJob) return;
    store.setState({ selectedJobId: g.primaryJob.id, activeTab: "viewer" });
  });

  dismissBtn.addEventListener("click", () => {
    dismissedGroupKeys.add(groupKey);
    renderMonitor();
  });

  return refs;
}

function patchQueueGroupRefs(refs, group, nowMs) {
  refs.currentGroup = group;

  if (refs.titleEl.textContent !== group.title) {
    refs.titleEl.textContent = group.title;
    refs.titleEl.title = group.title;
  }

  const modeKey =
    group.mode === "single" && group.primaryJob.effective_config?.diarization?.enabled
      ? "simple.modes.diarize.title"
      : `simple.modes.${group.mode}.title`;
  const modeLabel = i18n.t(modeKey) === modeKey ? group.mode : i18n.t(modeKey);
  if (refs.modeBadge.textContent !== modeLabel) {
    refs.modeBadge.textContent = modeLabel;
  }

  patchStageBadge(refs.stageBadge, group.statusCode);

  if (group.allTerminal && group.finishedAtMs > 0) {
    const remMs = Math.max(0, QUEUE_TTL_MS - (nowMs - group.finishedAtMs));
    refs.ttlBadge.hidden = false;
    refs.ttlBadge.textContent = formatCountdownMMSS(remMs);
    refs.card.classList.add("is-expiring");
  } else {
    refs.ttlBadge.hidden = true;
    refs.card.classList.remove("is-expiring");
  }

  const hasMultiTracks = group.childTracks.length > 1;
  const isCollapsed = collapsedGroupKeys.has(group.key);
  refs.toggleBtn.hidden = !hasMultiTracks;
  if (hasMultiTracks) {
    const toggleLabel = isCollapsed ? `▸ ${group.childTracks.length}` : "▾ Tracks";
    if (refs.toggleBtn.textContent !== toggleLabel) {
      refs.toggleBtn.textContent = toggleLabel;
    }
  }

  const canCancel = group.jobs.some((j) => ACTIVE_STATUSES.has(j.status) && j.status !== "cancelling");
  refs.cancelBtn.hidden = !canCancel;
  refs.cancelBtn.disabled = group.jobs.some((j) => formState.cancellingIds.has(j.id));

  const canOpen =
    group.jobs.length === 1 &&
    (group.primaryJob.status === "completed" || group.primaryJob.status === "completed_partial");
  refs.openBtn.hidden = !canOpen;
  refs.dismissBtn.hidden = !group.allTerminal;

  const isSuccessGroup =
    group.allTerminal &&
    group.jobs.every((j) => j.status === "completed" || j.status === "completed_partial");
  const isFailedGroup =
    group.allTerminal &&
    group.jobs.some((j) => j.status === "failed" || j.status === "cancelled");
  const hasRunning = group.childTracks.some((t) => t.statusCode === "transcribing");

  const fillClass = isSuccessGroup
    ? "progress-fill is-success"
    : isFailedGroup
      ? "progress-fill is-danger"
      : hasRunning
        ? "progress-fill is-active"
        : "progress-fill is-queued";
  if (refs.fill.className !== fillClass) {
    refs.fill.className = fillClass;
  }
  refs.fill.style.width = `${group.globalPct}%`;

  const doneCount = group.childTracks.filter((t) => t.isSuccess).length;
  const runCount = group.childTracks.filter((t) => t.statusCode === "transcribing").length;
  const queuedCount = group.childTracks.filter((t) => t.statusCode === "queued").length;

  const summaryParts = [`${group.globalPct}%`];
  if (hasMultiTracks) {
    summaryParts.push(`${doneCount}/${group.childTracks.length} completed`);
    if (runCount > 0) summaryParts.push(`${runCount} running`);
    if (queuedCount > 0) summaryParts.push(`${queuedCount} queued`);
  }
  const summaryStr = summaryParts.join(" · ");
  if (refs.summaryEl.textContent !== summaryStr) {
    refs.summaryEl.textContent = summaryStr;
  }

  refs.tracksContainer.hidden = !hasMultiTracks || isCollapsed;
  if (hasMultiTracks && !isCollapsed) {
    const seenTrackKeys = new Set();
    let prevRowNode = null;
    for (const track of group.childTracks) {
      seenTrackKeys.add(track.key);
      let tRefs = refs.trackRowsByKey.get(track.key);
      if (!tRefs) {
        tRefs = createTrackRowRefs(track.key);
        refs.trackRowsByKey.set(track.key, tRefs);
      }
      patchTrackRowRefs(tRefs, track);
      if (tRefs.row.parentElement !== refs.tracksContainer) {
        refs.tracksContainer.insertBefore(
          tRefs.row,
          prevRowNode ? prevRowNode.nextSibling : refs.tracksContainer.firstChild
        );
      }
      prevRowNode = tRefs.row;
    }
    for (const [k, tRefs] of refs.trackRowsByKey.entries()) {
      if (!seenTrackKeys.has(k)) {
        tRefs.row.remove();
        refs.trackRowsByKey.delete(k);
      }
    }
  }
}

function ensureMonitorShell() {
  if (monitorRefs && monitorRefs.root === monitorHost && monitorRefs.queueCard.isConnected) {
    return monitorRefs;
  }

  monitorHost.textContent = "";

  const devicesGrid = document.createElement("div");
  devicesGrid.className = "devices-grid";
  monitorHost.appendChild(devicesGrid);

  const queueCard = document.createElement("div");
  queueCard.className = "card";

  const queueHeader = document.createElement("div");
  queueHeader.className = "card-header";
  const queueTitle = document.createElement("h3");
  queueTitle.className = "card-title";
  queueTitle.textContent = i18n.t("simple.queue_title");
  queueHeader.appendChild(queueTitle);
  queueCard.appendChild(queueHeader);

  const queueBody = document.createElement("div");
  queueBody.className = "card-body stack";
  queueBody.style.gap = "0.75rem";

  const emptyEl = document.createElement("p");
  emptyEl.className = "muted";
  emptyEl.style.margin = "0";
  emptyEl.textContent = i18n.t("simple.queue_empty");
  queueBody.appendChild(emptyEl);

  queueCard.appendChild(queueBody);
  monitorHost.appendChild(queueCard);

  monitorRefs = {
    root: monitorHost,
    devicesGrid,
    deviceCardsById: new Map(),
    queueCard,
    queueTitle,
    queueBody,
    emptyEl,
    queueGroupsByKey: new Map(),
  };
  return monitorRefs;
}

function syncTtlTimer(hasExpiringGroups) {
  if (hasExpiringGroups && ttlTimerId === null) {
    ttlTimerId = window.setInterval(() => {
      if (!rootContainer || !rootContainer.isConnected) {
        window.clearInterval(ttlTimerId);
        ttlTimerId = null;
        return;
      }
      renderMonitor();
    }, 1000);
  } else if (!hasExpiringGroups && ttlTimerId !== null) {
    window.clearInterval(ttlTimerId);
    ttlTimerId = null;
  }
}

function renderMonitor() {
  if (!monitorHost) {
    return;
  }

  const refs = ensureMonitorShell();
  const state = store.getState();
  const devices = Array.isArray(state.devices) ? state.devices : [];
  const jobs = Array.isArray(state.jobs) ? state.jobs : [];
  const jobsById = new Map(jobs.map((job) => [job.id, job]));

  refs.devicesGrid.hidden = devices.length === 0;
  refs.devicesGrid.classList.toggle("has-many", devices.length >= 3);

  const seenDeviceIds = new Set();
  for (const device of devices) {
    const devId = device.id || (device.kind === "cuda" ? "cuda" : "cpu");
    seenDeviceIds.add(devId);
    let dRefs = refs.deviceCardsById.get(devId);
    if (!dRefs) {
      dRefs = createDeviceCardRefs(devId);
      refs.deviceCardsById.set(devId, dRefs);
      refs.devicesGrid.appendChild(dRefs.card);
    }
    patchDeviceCardRefs(dRefs, device, jobsById);
  }
  for (const [devId, dRefs] of refs.deviceCardsById.entries()) {
    if (!seenDeviceIds.has(devId)) {
      dRefs.card.remove();
      refs.deviceCardsById.delete(devId);
    }
  }

  const nowMs = Date.now();
  const visibleGroups = buildQueueGroups(jobs, nowMs);

  const queueTitleText = i18n.t("simple.queue_title");
  if (refs.queueTitle.textContent !== queueTitleText) {
    refs.queueTitle.textContent = queueTitleText;
  }
  const emptyText = i18n.t("simple.queue_empty");
  if (refs.emptyEl.textContent !== emptyText) {
    refs.emptyEl.textContent = emptyText;
  }
  refs.emptyEl.hidden = visibleGroups.length > 0;

  const seenGroupKeys = new Set();
  let prevGroupNode = refs.emptyEl;
  let hasExpiring = false;

  for (const group of visibleGroups) {
    if (group.allTerminal) {
      hasExpiring = true;
    }
    seenGroupKeys.add(group.key);
    let gRefs = refs.queueGroupsByKey.get(group.key);
    if (!gRefs) {
      gRefs = createQueueGroupRefs(group.key);
      refs.queueGroupsByKey.set(group.key, gRefs);
    }
    patchQueueGroupRefs(gRefs, group, nowMs);
    if (gRefs.card.parentElement !== refs.queueBody) {
      refs.queueBody.insertBefore(gRefs.card, prevGroupNode.nextSibling);
    }
    prevGroupNode = gRefs.card;
  }

  for (const [key, gRefs] of refs.queueGroupsByKey.entries()) {
    if (!seenGroupKeys.has(key)) {
      gRefs.card.remove();
      refs.queueGroupsByKey.delete(key);
    }
  }

  syncTtlTimer(hasExpiring);
}

export function render() {
  renderComposer();
  renderMonitor();
}

export function mount(container) {
  if (!container) {
    return;
  }

  if (unsubscribeStore) {
    unsubscribeStore();
    unsubscribeStore = null;
  }
  if (ttlTimerId !== null) {
    window.clearInterval(ttlTimerId);
    ttlTimerId = null;
  }
  monitorRefs = null;

  rootContainer = container;
  rootContainer.textContent = "";

  composerHost = document.createElement("div");
  composerHost.className = "stack";
  composerHost.id = "simple-composer-root";

  monitorHost = document.createElement("div");
  monitorHost.className = "stack";
  monitorHost.id = "simple-monitor-root";

  rootContainer.appendChild(composerHost);
  rootContainer.appendChild(monitorHost);

  render();

  unsubscribeStore = store.subscribe((state, prevState) => {
    if (!rootContainer || !rootContainer.isConnected) {
      return;
    }
    if (state.locale !== prevState.locale) {
      render();
      return;
    }
    if (state.profiles !== prevState.profiles) {
      renderComposer();
    }
    if (state.jobs !== prevState.jobs || state.devices !== prevState.devices) {
      renderMonitor();
    }
  });
}

export default { mount, render };
