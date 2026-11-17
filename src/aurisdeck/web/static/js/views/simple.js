/**
 * Simple View: iOS-safe file upload, 4 clear modes, multitrack editor,
 * quick hotwords chips, and live multi-GPU / active queue monitor.
 */

import { api } from "../api.js";
import { i18n } from "../i18n.js";
import { store } from "../store.js";

const ACCEPTED_EXTENSIONS =
  ".m4a,.mp3,.wav,.flac,.ogg,.opus,.aac,.mp4,.mkv,.webm,.mov,.zip";

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

const MODE_IDS = ["single", "diarize", "multitrack", "batch"];

const formState = {
  files: [],
  uiMode: "single",
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
  isSubmitting: false,
  submitError: "",
  cancellingIds: new Set(),
};

let rootContainer = null;
let composerHost = null;
let monitorHost = null;
let unsubscribeStore = null;

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

function isSingleZipSelection(files) {
  return (
    files.length === 1 &&
    String(files[0]?.name || "").toLowerCase().endsWith(".zip")
  );
}

function fileIdentity(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

async function syncMultitrackRows() {
  if (formState.uiMode !== "multitrack") {
    return;
  }

  const files = formState.files;
  if (!files.length) {
    formState.multitrackRows = [];
    formState.inspectKey = "";
    formState.inspectStatus = "idle";
    formState.inspectError = "";
    return;
  }

  if (isSingleZipSelection(files)) {
    const zipFile = files[0];
    const key = fileIdentity(zipFile);
    if (formState.inspectKey === key && formState.inspectStatus !== "idle") {
      return;
    }

    formState.inspectKey = key;
    formState.inspectStatus = "loading";
    formState.inspectError = "";
    formState.multitrackRows = [];
    renderComposer();

    try {
      const fd = new FormData();
      fd.append("file", zipFile);
      const tracks = await api.inspectZip(fd);
      if (formState.inspectKey !== key) {
        return;
      }
      const list = Array.isArray(tracks) ? tracks : [];
      formState.multitrackRows = list.map((item) => ({
        filename: String(item.filename || ""),
        speaker_name: String(item.speaker_name || stripExtension(item.filename || "")),
        offset_ms: Math.max(0, Number(item.offset_ms || 0)),
      }));
      formState.inspectStatus = "done";
    } catch (err) {
      if (formState.inspectKey !== key) {
        return;
      }
      formState.inspectStatus = "error";
      formState.inspectError = i18n.translateError(err);
      formState.multitrackRows = [];
    }
    renderComposer();
    return;
  }

  formState.inspectKey = "";
  formState.inspectStatus = "done";
  formState.inspectError = "";

  const existingByName = new Map(
    formState.multitrackRows.map((row) => [row.filename, row])
  );
  formState.multitrackRows = files.map((file) => {
    const prev = existingByName.get(file.name);
    if (prev) {
      return { ...prev };
    }
    return {
      filename: file.name,
      speaker_name: stripExtension(file.name),
      offset_ms: 0,
    };
  });
}

function handleSelectedFiles(fileList) {
  const nextFiles = Array.from(fileList || []);
  formState.files = nextFiles;
  formState.submitError = "";

  if (
    isSingleZipSelection(nextFiles) &&
    (formState.uiMode === "single" || formState.uiMode === "diarize")
  ) {
    formState.uiMode = "multitrack";
  }

  if (formState.uiMode === "multitrack") {
    void syncMultitrackRows().then(() => renderComposer());
  } else {
    renderComposer();
  }
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

  const backendMode = formState.uiMode === "diarize" ? "single" : formState.uiMode;
  if (backendMode === "single" && formState.files.length !== 1) {
    formState.submitError = i18n.t("errors.VALIDATION_ERROR");
    renderComposer();
    return;
  }

  const overrides = {};
  if (formState.uiMode === "single") {
    overrides.diarize = false;
  } else if (formState.uiMode === "diarize") {
    overrides.diarize = true;
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

  if (formState.hotwords.length > 0) {
    overrides.extra_hotwords = [...formState.hotwords];
  }

  if (formState.uiMode === "multitrack" && formState.multitrackRows.length > 0) {
    const speakerAliases = {};
    const trackOffsets = {};
    formState.multitrackRows.forEach((row, idx) => {
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

  const profiles = getSimpleProfiles();
  const profileId = resolveDefaultProfileId(profiles);
  formState.profileId = profileId;

  if (formState.uiMode === "multitrack") {
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
    formState.title = "";
    formState.hotwords = [];
    formState.hotwordDraft = "";
    formState.multitrackRows = [];
    formState.inspectKey = "";
    formState.inspectStatus = "idle";
    formState.inspectError = "";
    formState.submitError = "";

    store.addToast({
      type: "success",
      title: i18n.t("simple.submit"),
      message: i18n.t("stages.queued"),
    });

    monitorHost?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    const msg = i18n.translateError(err);
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
    const selected = event.target.files;
    if (selected && selected.length > 0) {
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
    const dropped = event.dataTransfer?.files;
    if (dropped && dropped.length > 0) {
      handleSelectedFiles(dropped);
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
        formState.files = formState.files.filter((_, idx) => idx !== index);
        if (formState.uiMode === "multitrack") {
          void syncMultitrackRows().then(() => renderComposer());
        } else {
          renderComposer();
        }
      });

      item.appendChild(info);
      item.appendChild(removeBtn);
      list.appendChild(item);
    });

    block.appendChild(list);
  }

  return block;
}

function buildModeSelectorBlock() {
  const block = document.createElement("div");
  block.className = "form-group";

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = i18n.t("simple.mode_label");
  block.appendChild(label);

  const grid = document.createElement("div");
  grid.className = "mode-grid";

  for (const modeId of MODE_IDS) {
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
      if (modeId === "multitrack") {
        void syncMultitrackRows().then(() => renderComposer());
      } else {
        renderComposer();
      }
    });

    grid.appendChild(btn);
  }

  block.appendChild(grid);

  if (formState.uiMode === "diarize") {
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

  if (!formState.multitrackRows.length) {
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
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  formState.multitrackRows.forEach((row, index) => {
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
      formState.multitrackRows[index].speaker_name = event.target.value;
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
      formState.multitrackRows[index].offset_ms =
        Number.isFinite(parsed) && parsed >= 0 ? Math.round(parsed) : 0;
    });
    offsetCell.appendChild(offsetInput);

    tr.appendChild(fileCell);
    tr.appendChild(speakerCell);
    tr.appendChild(offsetCell);
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  tableWrap.appendChild(table);
  wrapper.appendChild(tableWrap);
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
    if (formState.uiMode === "multitrack") {
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
  body.appendChild(buildModeSelectorBlock());

  const multitrackBlock = buildMultitrackTableBlock();
  if (multitrackBlock) {
    body.appendChild(multitrackBlock);
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
    formState.files.length === 0;
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

function buildDeviceCard(device, jobsById) {
  const card = document.createElement("div");
  card.className = "card";

  const header = document.createElement("div");
  header.className = "card-header";

  const left = document.createElement("div");
  const title = document.createElement("h3");
  title.className = "card-title mono";
  title.textContent = device.id || (device.kind === "cuda" ? "cuda" : "cpu");

  const subtitle = document.createElement("p");
  subtitle.className = "card-subtitle";
  subtitle.textContent =
    device.name || (device.kind === "cuda" ? i18n.t("common.cuda") : i18n.t("common.cpu"));

  left.appendChild(title);
  left.appendChild(subtitle);

  const kindBadge = document.createElement("span");
  kindBadge.className =
    device.kind === "cuda" ? "badge badge-gpu" : "badge badge-cpu";
  kindBadge.textContent =
    device.kind === "cuda" ? i18n.t("common.cuda") : i18n.t("common.cpu");

  header.appendChild(left);
  header.appendChild(kindBadge);
  card.appendChild(header);

  const body = document.createElement("div");
  body.className = "card-body";

  const activeJob = device.active_job_id ? jobsById.get(device.active_job_id) : null;
  if (device.active_job_id) {
    const jobTitle = document.createElement("div");
    jobTitle.className = "row-between";

    const nameSpan = document.createElement("strong");
    nameSpan.textContent = activeJob?.title || device.active_job_id;
    jobTitle.appendChild(nameSpan);

    const stageCode = device.stage || activeJob?.stage || activeJob?.status || "transcribing";
    jobTitle.appendChild(createStageBadge(stageCode));
    body.appendChild(jobTitle);

    const progressData = activeJob?.progress || {};
    const processed =
      device.processed_ms ?? progressData.processed_ms ?? 0;
    const total = device.total_ms ?? progressData.total_ms ?? 0;
    const percent =
      typeof device.percent === "number" && device.percent > 0
        ? device.percent
        : typeof progressData.percent === "number"
          ? progressData.percent
          : total > 0
            ? Math.min(100, Math.round((processed / total) * 100))
            : 0;
    const rtfx = device.rtfx ?? progressData.rtfx ?? null;
    const eta = computeEta(processed, total, rtfx);

    const bar = document.createElement("div");
    bar.className = "progress-bar";
    const fill = document.createElement("div");
    fill.className = "progress-fill";
    fill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    bar.appendChild(fill);
    body.appendChild(bar);

    const meta = document.createElement("div");
    meta.className = "progress-meta";

    const pctEl = document.createElement("span");
    pctEl.textContent = `${percent}%`;
    meta.appendChild(pctEl);

    const statsParts = [];
    if (rtfx !== null && rtfx !== undefined && Number(rtfx) > 0) {
      statsParts.push(`RTFx ${Number(rtfx).toFixed(1)}x`);
    }
    if (eta) {
      statsParts.push(`ETA ${eta}`);
    }
    const statsEl = document.createElement("span");
    statsEl.textContent = statsParts.join(" · ");
    meta.appendChild(statsEl);
    body.appendChild(meta);

    if (stageCode !== "cancelling") {
      const actions = document.createElement("div");
      actions.className = "row";
      actions.style.justifyContent = "flex-end";

      const cancelBtn = document.createElement("button");
      cancelBtn.type = "button";
      cancelBtn.className = "btn btn-danger btn-sm";
      cancelBtn.disabled = formState.cancellingIds.has(device.active_job_id);
      cancelBtn.textContent = i18n.t("common.cancel");
      cancelBtn.addEventListener("click", () => {
        void handleCancelJob(device.active_job_id);
      });
      actions.appendChild(cancelBtn);
      body.appendChild(actions);
    }
  } else {
    const idleText = document.createElement("div");
    idleText.className = "help-text";
    if (device.vram_total_mb && device.vram_free_mb !== null && device.vram_free_mb !== undefined) {
      idleText.textContent = i18n.t("settings.system.vram_usage", {
        free: device.vram_free_mb,
        total: device.vram_total_mb,
      });
    } else {
      idleText.textContent = i18n.t("simple.queue_empty");
    }
    body.appendChild(idleText);
  }

  card.appendChild(body);
  return card;
}

function buildJobQueueRow(job) {
  const item = document.createElement("div");
  item.className = "card";
  item.style.padding = "0.9rem 1rem";

  const topRow = document.createElement("div");
  topRow.className = "row-between";

  const titleWrap = document.createElement("div");
  titleWrap.className = "row";
  titleWrap.style.gap = "0.5rem";

  const titleEl = document.createElement("strong");
  titleEl.textContent = job.title || job.id;
  titleWrap.appendChild(titleEl);

  if (job.mode) {
    const modeBadge = document.createElement("span");
    modeBadge.className = "badge";
    const modeKey =
      job.mode === "single" && job.effective_config?.diarization?.enabled
        ? "simple.modes.diarize.title"
        : `simple.modes.${job.mode}.title`;
    const modeLabel = i18n.t(modeKey);
    modeBadge.textContent = modeLabel === modeKey ? job.mode : modeLabel;
    titleWrap.appendChild(modeBadge);
  }

  const assignedDevice = job.progress?.device_id || job.device_id;
  if (assignedDevice && ACTIVE_STATUSES.has(job.status)) {
    const devBadge = document.createElement("span");
    devBadge.className = assignedDevice.startsWith("cuda")
      ? "badge badge-gpu mono"
      : "badge badge-cpu mono";
    devBadge.textContent = assignedDevice;
    titleWrap.appendChild(devBadge);
  }

  const stageCode =
    ACTIVE_STATUSES.has(job.status) && job.stage ? job.stage : job.status || "queued";
  const statusBadge = createStageBadge(stageCode);

  topRow.appendChild(titleWrap);
  topRow.appendChild(statusBadge);
  item.appendChild(topRow);

  if (ACTIVE_STATUSES.has(job.status)) {
    const progress = job.progress || {};
    const processed = Number(progress.processed_ms || 0);
    const total = Number(progress.total_ms || 0);
    const percent =
      typeof progress.percent === "number"
        ? progress.percent
        : total > 0
          ? Math.min(100, Math.round((processed / total) * 100))
          : 0;
    const rtfx = progress.rtfx ?? null;
    const eta = computeEta(processed, total, rtfx);

    const progressWrap = document.createElement("div");
    progressWrap.className = "stack";
    progressWrap.style.gap = "0.35rem";
    progressWrap.style.marginTop = "0.65rem";

    const bar = document.createElement("div");
    bar.className = "progress-bar";
    const fill = document.createElement("div");
    fill.className = "progress-fill";
    fill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    bar.appendChild(fill);
    progressWrap.appendChild(bar);

    const meta = document.createElement("div");
    meta.className = "progress-meta";

    const leftMeta = document.createElement("span");
    leftMeta.textContent = `${percent}%`;
    meta.appendChild(leftMeta);

    const rightParts = [];
    if (rtfx !== null && rtfx !== undefined && Number(rtfx) > 0) {
      rightParts.push(`RTFx ${Number(rtfx).toFixed(1)}x`);
    }
    if (eta) {
      rightParts.push(`ETA ${eta}`);
    }
    const rightMeta = document.createElement("span");
    rightMeta.textContent = rightParts.join(" · ");
    meta.appendChild(rightMeta);

    progressWrap.appendChild(meta);
    item.appendChild(progressWrap);
  }

  if (
    (job.status === "failed" || job.status === "completed_partial") &&
    (job.error_code || job.error_detail)
  ) {
    const errText = document.createElement("div");
    errText.className = "help-text";
    errText.style.marginTop = "0.45rem";
    errText.textContent = i18n.translateError({
      error_code: job.error_code,
      error_params: job.error_params,
      detail: job.error_detail,
    });
    item.appendChild(errText);
  }

  const actions = document.createElement("div");
  actions.className = "row";
  actions.style.justifyContent = "flex-end";
  actions.style.marginTop = "0.65rem";

  if (ACTIVE_STATUSES.has(job.status) && job.status !== "cancelling") {
    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "btn btn-danger btn-sm";
    cancelBtn.disabled = formState.cancellingIds.has(job.id);
    cancelBtn.textContent = i18n.t("common.cancel");
    cancelBtn.addEventListener("click", () => {
      void handleCancelJob(job.id);
    });
    actions.appendChild(cancelBtn);
  }

  if (job.status === "completed" || job.status === "completed_partial") {
    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "btn btn-primary btn-sm";
    openBtn.textContent = i18n.t("simple.open_viewer");
    openBtn.addEventListener("click", () => {
      store.setState({ selectedJobId: job.id, activeTab: "viewer" });
    });
    actions.appendChild(openBtn);
  }

  if (actions.childElementCount > 0) {
    item.appendChild(actions);
  }

  return item;
}

function renderMonitor() {
  if (!monitorHost) {
    return;
  }

  monitorHost.textContent = "";

  const state = store.getState();
  const devices = Array.isArray(state.devices) ? state.devices : [];
  const jobs = Array.isArray(state.jobs) ? state.jobs : [];
  const jobsById = new Map(jobs.map((job) => [job.id, job]));

  if (devices.length > 0) {
    const devicesGrid = document.createElement("div");
    devicesGrid.className = devices.length >= 3 ? "grid-3" : "grid-2";
    for (const device of devices) {
      devicesGrid.appendChild(buildDeviceCard(device, jobsById));
    }
    monitorHost.appendChild(devicesGrid);
  }

  const queueCard = document.createElement("div");
  queueCard.className = "card";

  const queueHeader = document.createElement("div");
  queueHeader.className = "card-header";
  const queueTitle = document.createElement("h3");
  queueTitle.className = "card-title";
  queueTitle.textContent = i18n.t("simple.queue_title");
  queueHeader.appendChild(queueTitle);
  queueCard.appendChild(queueHeader);

  const activeJobs = jobs.filter((job) => ACTIVE_STATUSES.has(job.status));
  const recentFinished = jobs
    .filter((job) => TERMINAL_STATUSES.has(job.status))
    .slice(0, 5);
  const visibleJobs = [...activeJobs, ...recentFinished].slice(0, 10);

  const queueBody = document.createElement("div");
  queueBody.className = "card-body";

  if (visibleJobs.length === 0) {
    const emptyEl = document.createElement("p");
    emptyEl.className = "muted";
    emptyEl.style.margin = "0";
    emptyEl.textContent = i18n.t("simple.queue_empty");
    queueBody.appendChild(emptyEl);
  } else {
    for (const job of visibleJobs) {
      queueBody.appendChild(buildJobQueueRow(job));
    }
  }

  queueCard.appendChild(queueBody);
  monitorHost.appendChild(queueCard);
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
