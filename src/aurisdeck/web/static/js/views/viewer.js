/**
 * Transcript Viewer: synchronized media player, segment highlight & seek,
 * speaker renaming, partial/failed recovery actions, and export/VTT preview.
 */

import { api } from "../api.js";
import { i18n } from "../i18n.js";
import { store } from "../store.js";

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

const DEFAULT_SPEEDS = [0.75, 1, 1.25, 1.5, 2];
const EXPORT_FORMATS = ["vtt", "srt", "md", "json"];

const containerCleanups = new WeakMap();

function tl(key, esFallback, enFallback) {
  const translated = i18n.t(key);
  if (translated !== key) {
    return translated;
  }
  const locale = String(store.getState().locale || i18n.getLocale() || "es").toLowerCase();
  return locale.startsWith("en") ? enFallback : esFallback;
}

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
      } else if (key === "selected") {
        node.selected = Boolean(value);
      } else if (key === "disabled") {
        node.disabled = Boolean(value);
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

function formatMs(ms, { vtt = false } = {}) {
  const totalMs = Math.max(0, Math.round(Number(ms || 0)));
  const totalSeconds = Math.floor(totalMs / 1000);
  const remMs = totalMs % 1000;
  const hours = Math.floor(totalSeconds / 3600);
  const mins = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;
  const hh = String(hours).padStart(2, "0");
  const mm = String(mins).padStart(2, "0");
  const ss = String(secs).padStart(2, "0");
  if (vtt) {
    return `${hh}:${mm}:${ss}.${String(remMs).padStart(3, "0")}`;
  }
  if (hours > 0) {
    return `${hours}:${mm}:${ss}`;
  }
  return `${mm}:${ss}`;
}

function formatDate(iso) {
  if (!iso) {
    return "";
  }
  const ms = Date.parse(String(iso));
  if (!Number.isFinite(ms)) {
    return String(iso);
  }
  const locale = store.getState().locale || i18n.getLocale() || "es";
  try {
    return new Intl.DateTimeFormat(locale, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(ms));
  } catch {
    return new Date(ms).toLocaleString();
  }
}

function resolvePlayerDefaults() {
  const state = store.getState();
  const prefs = state.preferences || state.user?.preferences || {};
  const player = prefs?.player && typeof prefs.player === "object" ? prefs.player : {};
  const rawRate = Number(player.playback_rate);
  const playbackRate =
    Number.isFinite(rawRate) && rawRate >= 0.25 && rawRate <= 4.0 ? rawRate : 1;
  const autoScroll =
    player.auto_scroll !== undefined ? Boolean(player.auto_scroll) : true;
  return { playbackRate, autoScroll };
}

function createStageBadge(job) {
  const status = String(job?.status || "queued");
  const code = ACTIVE_STATUSES.has(status) && job?.stage ? String(job.stage) : status;
  let variant = "badge-info";
  if (code === "completed") {
    variant = "badge-success";
  } else if (code === "completed_partial" || code === "cancelling") {
    variant = "badge-warning";
  } else if (code === "failed") {
    variant = "badge-danger";
  } else if (code === "cancelled" || code === "queued") {
    variant = "";
  }
  const key = `stages.${code}`;
  const translated = i18n.t(key);
  return el("span", {
    className: variant ? `badge ${variant}` : "badge",
    textContent: translated === key ? code : translated,
  });
}

function getModeLabel(job) {
  const mode = String(job?.mode || "single");
  const modeKey =
    mode === "single" && job?.effective_config?.diarization?.enabled
      ? "simple.modes.diarize.title"
      : `simple.modes.${mode}.title`;
  const translated = i18n.t(modeKey);
  return translated === modeKey ? mode : translated;
}

function buildSpeakerMap(job) {
  const map = new Map();
  if (Array.isArray(job?.speakers)) {
    for (const sp of job.speakers) {
      if (sp?.speaker_key) {
        map.set(sp.speaker_key, sp.display_name || sp.speaker_key);
      }
    }
  }
  return map;
}

function buildTranscriptPlainText(job) {
  const segments = Array.isArray(job?.segments) ? job.segments : [];
  if (!segments.length) {
    return "";
  }
  const speakerMap = buildSpeakerMap(job);
  return segments
    .map((seg) => {
      const time = formatMs(seg.start_ms);
      const speaker = seg.speaker_key
        ? speakerMap.get(seg.speaker_key) || seg.speaker_key
        : "";
      const text = String(seg.text || "").trim();
      return speaker ? `[${time}] ${speaker}: ${text}` : `[${time}] ${text}`;
    })
    .join("\n");
}

function buildFallbackVtt(job) {
  const segments = Array.isArray(job?.segments) ? job.segments : [];
  if (!segments.length) {
    return "WEBVTT\n";
  }
  const speakerMap = buildSpeakerMap(job);
  const lines = ["WEBVTT", ""];
  for (const seg of segments) {
    const start = formatMs(seg.start_ms, { vtt: true });
    const end = formatMs(Math.max(Number(seg.end_ms || 0), Number(seg.start_ms || 0) + 1), {
      vtt: true,
    });
    const speaker = seg.speaker_key
      ? speakerMap.get(seg.speaker_key) || seg.speaker_key
      : "";
    const text = String(seg.text || "").trim();
    lines.push(`${start} --> ${end}`);
    lines.push(speaker ? `<v ${speaker}>${text}` : text);
    lines.push("");
  }
  return lines.join("\n");
}

async function copyTextToClipboard(text) {
  if (
    typeof navigator !== "undefined" &&
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === "function"
  ) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    const ok = document.execCommand("copy");
    if (!ok) {
      throw new Error("Copy command rejected");
    }
  } finally {
    document.body.removeChild(textarea);
  }
}

function upsertJobInStore(returned) {
  const job = returned?.job && returned.job.id ? returned.job : returned;
  if (!job || !job.id) {
    return;
  }
  store.setState((current) => {
    const list = Array.isArray(current.jobs) ? current.jobs.slice() : [];
    const idx = list.findIndex((item) => item.id === job.id);
    if (idx === -1) {
      list.unshift(job);
    } else {
      list[idx] = { ...list[idx], ...job };
    }
    return { jobs: list };
  });
}

function getStoreJobSnapshot(jobId) {
  const jobs = store.getState().jobs;
  if (!Array.isArray(jobs)) {
    return null;
  }
  return jobs.find((item) => item.id === jobId) || null;
}

/**
 * Mount the Transcript Viewer into `container` (#view-viewer).
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

  const defaults = resolvePlayerDefaults();
  const local = {
    disposed: false,
    jobId: store.getState().selectedJobId || null,
    job: null,
    loading: false,
    loadError: "",
    loadSeq: 0,
    vttSeq: 0,

    selectedTrackId: "",
    playbackRate: defaults.playbackRate,
    autoScroll: defaults.autoScroll,
    audioError: "",
    pendingSeekSec: null,
    activeSegmentIndex: -1,
    segmentElements: [],

    speakerDrafts: new Map(),
    savingSpeakers: false,
    diarizing: false,
    retrying: false,
    reexporting: false,
    retryProfileId: "",
    retryHotwords: "",

    vttPreviewText: "",
    vttPreviewStatus: "idle",
    vttPreviewOpen: false,
  };

  let shellRoot = null;
  let headerHost = null;
  let bannerHost = null;
  let playerHost = null;
  let transcriptHost = null;
  let speakersHost = null;
  let exportHost = null;
  let audioEl = null;
  let prevSnapshotStatus = null;
  let prevSnapshotDiarStatus = null;

  function detachAudioListeners() {
    if (!audioEl) {
      return;
    }
    audioEl.removeEventListener("timeupdate", handleTimeUpdate);
    audioEl.removeEventListener("seeked", handleTimeUpdate);
    audioEl.removeEventListener("loadedmetadata", handleLoadedMetadata);
    audioEl.removeEventListener("error", handleAudioError);
    audioEl = null;
  }

  function findActiveSegmentIndex(currentMs) {
    const segments = Array.isArray(local.job?.segments) ? local.job.segments : [];
    for (let i = 0; i < segments.length; i += 1) {
      const start = Number(segments[i].start_ms || 0);
      const end = Number(segments[i].end_ms || 0);
      if (currentMs >= start && currentMs <= end) {
        return i;
      }
    }
    return -1;
  }

  function syncActiveSegmentHighlight({ scroll = true } = {}) {
    if (!audioEl) {
      return;
    }
    const currentMs = Number(audioEl.currentTime || 0) * 1000;
    const nextIdx = findActiveSegmentIndex(currentMs);
    if (nextIdx === local.activeSegmentIndex) {
      return;
    }
    if (local.activeSegmentIndex >= 0 && local.segmentElements[local.activeSegmentIndex]) {
      local.segmentElements[local.activeSegmentIndex].classList.remove("is-active");
    }
    local.activeSegmentIndex = nextIdx;
    if (nextIdx >= 0 && local.segmentElements[nextIdx]) {
      const activeNode = local.segmentElements[nextIdx];
      activeNode.classList.add("is-active");
      if (scroll && local.autoScroll && typeof activeNode.scrollIntoView === "function") {
        activeNode.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }
  }

  function handleTimeUpdate() {
    syncActiveSegmentHighlight({ scroll: true });
  }

  function handleLoadedMetadata() {
    if (!audioEl) {
      return;
    }
    audioEl.playbackRate = local.playbackRate;
    local.audioError = "";
    if (local.pendingSeekSec !== null) {
      const target = local.pendingSeekSec;
      local.pendingSeekSec = null;
      try {
        audioEl.currentTime = target;
      } catch {
        // Ignore invalid seek offsets.
      }
      audioEl.play().catch(() => {});
    }
  }

  function handleAudioError() {
    local.audioError = i18n.t("errors.MEDIA_INVALID");
    const errEl = playerHost?.querySelector("[data-player-error]");
    if (errEl) {
      errEl.textContent = local.audioError;
      errEl.classList.remove("hidden");
    }
  }

  function seekAndPlay(startMs) {
    if (!audioEl) {
      return;
    }
    const targetSec = Math.max(0, Number(startMs || 0) / 1000);
    if (audioEl.readyState >= 1) {
      try {
        audioEl.currentTime = targetSec;
      } catch {
        // Ignore seek errors on unbuffered streams.
      }
      audioEl.playbackRate = local.playbackRate;
      audioEl.play().catch(() => {});
      syncActiveSegmentHighlight({ scroll: false });
      return;
    }
    local.pendingSeekSec = targetSec;
    audioEl.play().catch(() => {});
  }

  function syncSpeakerDraftsFromJob(job) {
    local.speakerDrafts = new Map();
    if (Array.isArray(job?.speakers)) {
      for (const sp of job.speakers) {
        if (sp?.speaker_key) {
          local.speakerDrafts.set(sp.speaker_key, sp.display_name || sp.speaker_key);
        }
      }
    }
  }

  function syncRetryDefaultsFromJob(job) {
    local.retryProfileId = String(job?.effective_config?.profile_id || "");
    const hw = job?.effective_config?.context?.hotwords;
    local.retryHotwords = Array.isArray(hw) ? hw.join(", ") : "";
  }

  async function refreshVttPreview() {
    const job = local.job;
    if (!job || !job.id) {
      local.vttPreviewText = "";
      local.vttPreviewStatus = "idle";
      return;
    }

    const available = Array.isArray(job.available_formats) ? job.available_formats : [];
    if (!available.includes("vtt")) {
      local.vttPreviewText = buildFallbackVtt(job);
      local.vttPreviewStatus = "ready";
      renderExportBar();
      return;
    }

    const seq = ++local.vttSeq;
    local.vttPreviewStatus = "loading";
    renderExportBar();
    try {
      const response = await fetch(api.jobDownloadUrl(job.id, "vtt"), {
        credentials: "same-origin",
        headers: { Accept: "text/vtt, text/plain;q=0.9, */*;q=0.8" },
      });
      if (local.disposed || seq !== local.vttSeq) {
        return;
      }
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      local.vttPreviewText = await response.text();
      local.vttPreviewStatus = "ready";
    } catch {
      if (local.disposed || seq !== local.vttSeq) {
        return;
      }
      local.vttPreviewText = buildFallbackVtt(job);
      local.vttPreviewStatus = "ready";
    }
    renderExportBar();
  }

  async function loadJob(jobId, { preservePlayer = false } = {}) {
    if (!jobId) {
      local.jobId = null;
      local.job = null;
      local.loading = false;
      local.loadError = "";
      detachAudioListeners();
      renderEmptyOrError();
      return;
    }

    const seq = ++local.loadSeq;
    const switchedJob = local.jobId !== jobId;
    local.jobId = jobId;
    local.loading = true;
    local.loadError = "";
    if (switchedJob || !local.job) {
      local.job = null;
      local.selectedTrackId = "";
      local.audioError = "";
      local.activeSegmentIndex = -1;
      local.vttPreviewText = "";
      local.vttPreviewStatus = "idle";
      const defaultsNow = resolvePlayerDefaults();
      local.playbackRate = defaultsNow.playbackRate;
      local.autoScroll = defaultsNow.autoScroll;
      detachAudioListeners();
      renderEmptyOrError();
    }

    try {
      const detail = await api.getJob(jobId);
      if (local.disposed || seq !== local.loadSeq) {
        return;
      }
      local.job = detail;
      local.loading = false;
      local.loadError = "";
      prevSnapshotStatus = detail.status || null;
      prevSnapshotDiarStatus = detail.diarization_status || null;
      if (
        !local.selectedTrackId ||
        !Array.isArray(detail.tracks) ||
        !detail.tracks.some((t) => t.id === local.selectedTrackId)
      ) {
        local.selectedTrackId = detail.tracks?.[0]?.id || "";
      }
      syncSpeakerDraftsFromJob(detail);
      syncRetryDefaultsFromJob(detail);

      if (preservePlayer && shellRoot && playerHost) {
        renderHeader();
        renderBanner();
        if (!audioEl) {
          renderPlayer();
        }
        renderTranscript();
        renderSpeakers();
        renderExportBar();
      } else {
        renderShell();
      }
      void refreshVttPreview();
    } catch (err) {
      if (local.disposed || seq !== local.loadSeq) {
        return;
      }
      local.loading = false;
      local.loadError = i18n.translateError(err);
      renderEmptyOrError();
    }
  }

  async function handleSaveSpeakers(event) {
    event.preventDefault();
    if (!local.job || local.savingSpeakers) {
      return;
    }

    const aliases = {};
    for (const sp of local.job.speakers || []) {
      const key = sp.speaker_key;
      if (!key) {
        continue;
      }
      const raw = String(local.speakerDrafts.get(key) ?? "").trim();
      if (!raw) {
        store.addToast({
          type: "danger",
          title: i18n.t("common.error"),
          message: i18n.t("errors.VALIDATION_ERROR"),
        });
        return;
      }
      aliases[key] = raw;
    }

    local.savingSpeakers = true;
    renderSpeakers();
    try {
      const res = await api.patchSpeakers(local.job.id, aliases);
      if (local.disposed) {
        return;
      }
      local.job = {
        ...local.job,
        speakers: Array.isArray(res?.speakers) ? res.speakers : local.job.speakers,
        segments: Array.isArray(res?.segments) ? res.segments : local.job.segments,
        available_formats: Array.isArray(res?.available_formats)
          ? res.available_formats
          : local.job.available_formats,
      };
      syncSpeakerDraftsFromJob(local.job);
      upsertJobInStore({
        id: local.job.id,
        speakers: local.job.speakers,
      });
      renderSpeakers();
      renderTranscript();
      renderExportBar();
      void refreshVttPreview();
      store.addToast({
        type: "success",
        title: i18n.t("viewer.speakers_title"),
        message: i18n.t("common.saved"),
      });
    } catch (err) {
      store.addToast({
        type: "danger",
        title: i18n.t("common.error"),
        message: i18n.translateError(err),
      });
    } finally {
      local.savingSpeakers = false;
      if (!local.disposed) {
        renderSpeakers();
      }
    }
  }

  async function handleDiarizeNow() {
    if (!local.job || local.diarizing) {
      return;
    }
    local.diarizing = true;
    renderBanner();
    try {
      const res = await api.diarizeJob(local.job.id);
      upsertJobInStore(res);
      const updatedJob = res?.job || res;
      if (updatedJob && updatedJob.id) {
        local.job = { ...local.job, ...updatedJob };
      }
      renderHeader();
      renderBanner();
      store.addToast({
        type: "info",
        title: i18n.t("viewer.diarize_now"),
        message: i18n.t("stages.diarizing"),
      });
    } catch (err) {
      store.addToast({
        type: "danger",
        title: i18n.t("common.error"),
        message: i18n.translateError(err),
      });
    } finally {
      local.diarizing = false;
      if (!local.disposed) {
        renderBanner();
      }
    }
  }

  async function handleRetry() {
    if (!local.job || local.retrying) {
      return;
    }
    local.retrying = true;
    renderBanner();

    const body = {};
    if (local.retryProfileId) {
      body.profile_id = local.retryProfileId;
    }
    const parsedHotwords = String(local.retryHotwords || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const originalHotwords = Array.isArray(local.job.effective_config?.context?.hotwords)
      ? local.job.effective_config.context.hotwords
      : [];
    const hotwordsChanged =
      parsedHotwords.length !== originalHotwords.length ||
      parsedHotwords.some((w, idx) => w !== originalHotwords[idx]);
    if (hotwordsChanged) {
      body.extra_hotwords = parsedHotwords;
    }

    try {
      const res = await api.retryJob(
        local.job.id,
        Object.keys(body).length > 0 ? body : null
      );
      upsertJobInStore(res);
      const updatedJob = res?.job || res;
      if (updatedJob && updatedJob.id) {
        local.job = { ...local.job, ...updatedJob };
      }
      renderHeader();
      renderBanner();
      store.addToast({
        type: "info",
        title: i18n.t("common.retry"),
        message: i18n.t("stages.queued"),
      });
    } catch (err) {
      store.addToast({
        type: "danger",
        title: i18n.t("common.error"),
        message: i18n.translateError(err),
      });
    } finally {
      local.retrying = false;
      if (!local.disposed) {
        renderBanner();
      }
    }
  }

  async function handleReexport() {
    if (!local.job || local.reexporting) {
      return;
    }
    local.reexporting = true;
    renderExportBar();
    try {
      const res = await api.reexportJob(local.job.id);
      if (local.disposed) {
        return;
      }
      if (Array.isArray(res?.available_formats)) {
        local.job = { ...local.job, available_formats: res.available_formats };
      }
      renderExportBar();
      void refreshVttPreview();
      store.addToast({
        type: "success",
        title: i18n.t("viewer.reexport"),
        message: i18n.t("common.success"),
      });
    } catch (err) {
      store.addToast({
        type: "danger",
        title: i18n.t("common.error"),
        message: i18n.translateError(err),
      });
    } finally {
      local.reexporting = false;
      if (!local.disposed) {
        renderExportBar();
      }
    }
  }

  async function handleCopyText() {
    const text = buildTranscriptPlainText(local.job);
    if (!text) {
      store.addToast({
        type: "warning",
        title: i18n.t("viewer.title"),
        message: i18n.t("viewer.no_segments"),
      });
      return;
    }
    try {
      await copyTextToClipboard(text);
      store.addToast({
        type: "success",
        title: tl("viewer.copy_text", "Copiar texto", "Copy text"),
        message: i18n.t("common.success"),
      });
    } catch (err) {
      store.addToast({
        type: "danger",
        title: i18n.t("common.error"),
        message: i18n.translateError(err),
      });
    }
  }

  function renderHeader() {
    if (!headerHost) {
      return;
    }
    headerHost.textContent = "";
    const job = local.job;
    if (!job) {
      return;
    }
    const storeSnap = getStoreJobSnapshot(job.id);
    const mergedJob = storeSnap ? { ...job, ...storeSnap } : job;
    const lang =
      mergedJob.language_detected ||
      mergedJob.language_requested ||
      i18n.t("common.auto");

    headerHost.appendChild(
      el(
        "div",
        { className: "section-header" },
        el(
          "div",
          { className: "stack", style: "gap: 0.4rem;" },
          el("h2", { className: "section-title", textContent: mergedJob.title || mergedJob.id }),
          el(
            "div",
            { className: "row", style: "gap: 0.45rem;" },
            createStageBadge(mergedJob),
            el("span", { className: "badge", textContent: getModeLabel(mergedJob) }),
            el("span", { className: "badge mono", textContent: lang }),
            mergedJob.created_at
              ? el("span", { className: "help-text", textContent: formatDate(mergedJob.created_at) })
              : null
          )
        ),
        el(
          "div",
          { className: "row" },
          el("button", {
            type: "button",
            className: "btn btn-secondary btn-sm",
            textContent: i18n.t("nav.library"),
            onClick: () => store.setState({ activeTab: "library" }),
          })
        )
      )
    );
  }

  function renderBanner() {
    if (!bannerHost) {
      return;
    }
    bannerHost.textContent = "";
    const job = local.job;
    if (!job) {
      return;
    }

    const storeSnap = getStoreJobSnapshot(job.id);
    const effectiveStatus = storeSnap?.status || job.status;
    const effectiveStage = storeSnap?.stage || job.stage;
    const progress = storeSnap?.progress || null;

    if (ACTIVE_STATUSES.has(effectiveStatus)) {
      const percent = Math.max(0, Math.min(100, Number(progress?.percent || 0)));
      const fill = el("div", { className: "progress-fill" });
      fill.style.width = `${percent}%`;
      bannerHost.appendChild(
        el(
          "div",
          { className: "alert alert-info" },
          el(
            "div",
            { className: "row-between" },
            el("strong", {
              textContent: i18n.t(`stages.${effectiveStage || effectiveStatus}`),
            }),
            el("span", { className: "mono", textContent: `${percent}%` })
          ),
          el("div", { className: "progress-bar" }, fill)
        )
      );
      return;
    }

    if (effectiveStatus === "failed" || effectiveStatus === "cancelled") {
      const hasError = Boolean(job.error_code || job.error_detail);
      const errMsg = hasError
        ? i18n.translateError({
            error_code: job.error_code,
            error_params: job.error_params,
            detail: job.error_detail,
          })
        : i18n.t(`stages.${effectiveStatus}`);

      const profiles = Array.isArray(store.getState().profiles)
        ? store.getState().profiles
        : [];
      const locale = store.getState().locale || i18n.getLocale() || "es";

      const retryOptions = el(
        "details",
        null,
        el("summary", {
          style: "cursor: pointer; font-weight: 600;",
          textContent: tl(
            "viewer.retry_options",
            "Opciones de reintento (perfil / palabras clave)",
            "Retry options (profile / hotwords)"
          ),
        }),
        el(
          "div",
          { className: "form-grid", style: "margin-top: 0.65rem;" },
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              textContent: i18n.t("simple.profile_label"),
            }),
            el(
              "select",
              {
                className: "select",
                onChange: (event) => {
                  local.retryProfileId = event.target.value;
                },
              },
              el("option", {
                value: "",
                selected: local.retryProfileId === "",
                textContent: i18n.t("common.auto"),
              }),
              profiles.map((profile) =>
                el("option", {
                  value: profile.id,
                  selected: local.retryProfileId === profile.id,
                  textContent:
                    profile?.display_name_i18n?.[locale] ||
                    profile?.display_name ||
                    profile.id,
                })
              )
            )
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              textContent: i18n.t("simple.hotwords_label"),
            }),
            el("input", {
              type: "text",
              className: "input",
              value: local.retryHotwords,
              placeholder: i18n.t("simple.hotwords_placeholder"),
              onInput: (event) => {
                local.retryHotwords = event.target.value;
              },
            })
          )
        )
      );

      bannerHost.appendChild(
        el(
          "div",
          {
            className:
              effectiveStatus === "failed" ? "alert alert-danger" : "alert",
          },
          el(
            "div",
            { className: "row-between" },
            el("strong", { textContent: errMsg }),
            el("button", {
              type: "button",
              className: "btn btn-primary btn-sm",
              disabled: local.retrying,
              textContent: i18n.t("common.retry"),
              onClick: () => void handleRetry(),
            })
          ),
          job.error_detail
            ? el(
                "details",
                null,
                el("summary", {
                  style: "cursor: pointer;",
                  textContent: tl("viewer.error_details", "Detalle técnico", "Technical details"),
                }),
                el("pre", {
                  className: "mono",
                  style: "margin: 0.4rem 0 0; white-space: pre-wrap; font-size: 0.8rem;",
                  textContent: job.error_detail,
                })
              )
            : null,
          retryOptions
        )
      );
      return;
    }

    const canPostHocDiarize =
      effectiveStatus === "completed_partial" ||
      (job.mode === "single" && job.diarization_status === "none");

    if (canPostHocDiarize) {
      const hasError = Boolean(job.error_code || job.error_detail);
      const message = hasError
        ? i18n.translateError({
            error_code: job.error_code,
            error_params: job.error_params,
            detail: job.error_detail,
          })
        : i18n.t("simple.modes.diarize.description");

      bannerHost.appendChild(
        el(
          "div",
          {
            className:
              effectiveStatus === "completed_partial"
                ? "alert alert-warning"
                : "alert alert-info",
          },
          el(
            "div",
            { className: "row-between" },
            el("span", { textContent: message }),
            el("button", {
              type: "button",
              className: "btn btn-primary btn-sm",
              disabled: local.diarizing,
              textContent: i18n.t("viewer.diarize_now"),
              onClick: () => void handleDiarizeNow(),
            })
          ),
          job.error_detail
            ? el(
                "details",
                null,
                el("summary", {
                  style: "cursor: pointer;",
                  textContent: tl("viewer.error_details", "Detalle técnico", "Technical details"),
                }),
                el("pre", {
                  className: "mono",
                  style: "margin: 0.4rem 0 0; white-space: pre-wrap; font-size: 0.8rem;",
                  textContent: job.error_detail,
                })
              )
            : null
        )
      );
    }
  }

  function renderPlayer() {
    if (!playerHost) {
      return;
    }
    detachAudioListeners();
    playerHost.textContent = "";

    const job = local.job;
    const tracks = Array.isArray(job?.tracks) ? job.tracks : [];
    if (!job || tracks.length === 0 || job.media_retained === false) {
      return;
    }

    if (!tracks.some((t) => t.id === local.selectedTrackId)) {
      local.selectedTrackId = tracks[0].id;
    }

    const speakerMap = buildSpeakerMap(job);

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = api.jobMediaUrl(job.id, local.selectedTrackId);
    audio.playbackRate = local.playbackRate;
    audioEl = audio;

    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("seeked", handleTimeUpdate);
    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("error", handleAudioError);

    const trackSelect =
      tracks.length > 1
        ? el(
            "label",
            { className: "row", style: "gap: 0.4rem;" },
            el("span", {
              className: "help-text",
              textContent: tl("viewer.track_label", "Pista", "Track"),
            }),
            el(
              "select",
              {
                className: "select",
                style: "width: auto;",
                "aria-label": tl("viewer.track_label", "Pista", "Track"),
                onChange: (event) => {
                  const nextId = event.target.value;
                  if (!nextId || nextId === local.selectedTrackId) {
                    return;
                  }
                  const wasPlaying = Boolean(audioEl && !audioEl.paused);
                  const prevTime = Number(audioEl?.currentTime || 0);
                  local.selectedTrackId = nextId;
                  local.pendingSeekSec = prevTime > 0 ? prevTime : null;
                  if (audioEl) {
                    audioEl.src = api.jobMediaUrl(job.id, nextId);
                    audioEl.load();
                    if (wasPlaying) {
                      audioEl.play().catch(() => {});
                    }
                  }
                },
              },
              tracks.map((track) => {
                const spLabel = track.speaker_key
                  ? speakerMap.get(track.speaker_key) || track.speaker_key
                  : "";
                const label = spLabel
                  ? `${spLabel} — ${track.original_filename}`
                  : track.original_filename;
                return el("option", {
                  value: track.id,
                  selected: track.id === local.selectedTrackId,
                  textContent: label,
                });
              })
            )
          )
        : null;

    const speedOptions = DEFAULT_SPEEDS.includes(local.playbackRate)
      ? DEFAULT_SPEEDS
      : [...DEFAULT_SPEEDS, local.playbackRate].sort((a, b) => a - b);

    const speedSelect = el(
      "label",
      { className: "row", style: "gap: 0.4rem;" },
      el("span", {
        className: "help-text",
        textContent: i18n.t("viewer.playback_rate"),
      }),
      el(
        "select",
        {
          className: "select",
          style: "width: auto;",
          "aria-label": i18n.t("viewer.playback_rate"),
          onChange: (event) => {
            const nextRate = Number(event.target.value);
            if (Number.isFinite(nextRate) && nextRate > 0) {
              local.playbackRate = nextRate;
              if (audioEl) {
                audioEl.playbackRate = nextRate;
              }
            }
          },
        },
        speedOptions.map((rate) =>
          el("option", {
            value: String(rate),
            selected: rate === local.playbackRate,
            textContent: `${rate}x`,
          })
        )
      )
    );

    const autoScrollToggle = el(
      "label",
      { className: "checkbox-label" },
      el("input", {
        type: "checkbox",
        checked: local.autoScroll,
        onChange: (event) => {
          local.autoScroll = Boolean(event.target.checked);
        },
      }),
      el("span", { textContent: i18n.t("viewer.auto_scroll") })
    );

    const errorNote = el("div", {
      className: local.audioError ? "form-error" : "form-error hidden",
      "data-player-error": "true",
      textContent: local.audioError,
    });

    playerHost.appendChild(
      el(
        "div",
        { className: "stack", style: "gap: 0.45rem;" },
        el(
          "div",
          { className: "player-bar" },
          audio,
          trackSelect,
          speedSelect,
          autoScrollToggle
        ),
        errorNote
      )
    );
  }

  function renderTranscript() {
    if (!transcriptHost) {
      return;
    }
    transcriptHost.textContent = "";
    local.segmentElements = [];
    local.activeSegmentIndex = -1;

    const job = local.job;
    if (!job) {
      return;
    }

    const segments = Array.isArray(job.segments) ? job.segments : [];
    const speakerMap = buildSpeakerMap(job);

    const header = el(
      "div",
      { className: "card-header" },
      el("h3", { className: "card-title", textContent: i18n.t("viewer.title") }),
      el("span", {
        className: "help-text",
        textContent: i18n.t("common.items_count", { count: segments.length }),
      })
    );

    const body = el("div", { className: "card-body" });

    if (segments.length === 0) {
      body.appendChild(
        el("p", {
          className: "muted",
          style: "margin: 0;",
          textContent: i18n.t("viewer.no_segments"),
        })
      );
      transcriptHost.appendChild(el("div", { className: "card" }, header, body));
      return;
    }

    const list = el("div", { className: "segment-list" });
    segments.forEach((seg, index) => {
      const speakerLabel = seg.speaker_key
        ? speakerMap.get(seg.speaker_key) || seg.speaker_key
        : "";
      const timeNode = el("span", {
        className: "segment-time",
        textContent: formatMs(seg.start_ms),
      });
      const speakerNode = el("span", {
        className: "segment-speaker",
        textContent: speakerLabel,
      });
      const textNode = el("p", {
        className: "segment-text",
        textContent: String(seg.text || ""),
      });

      const item = el(
        "div",
        {
          className: "segment-item",
          role: "button",
          tabindex: "0",
          "data-segment-index": String(index),
          onClick: () => seekAndPlay(seg.start_ms),
          onKeydown: (event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              seekAndPlay(seg.start_ms);
            }
          },
        },
        timeNode,
        speakerNode,
        textNode
      );

      local.segmentElements.push(item);
      list.appendChild(item);
    });

    body.appendChild(list);
    transcriptHost.appendChild(el("div", { className: "card" }, header, body));
    syncActiveSegmentHighlight({ scroll: false });
  }

  function renderSpeakers() {
    if (!speakersHost) {
      return;
    }
    speakersHost.textContent = "";
    const job = local.job;
    const speakers = Array.isArray(job?.speakers) ? job.speakers : [];
    if (!job || speakers.length === 0) {
      return;
    }

    const form = el(
      "form",
      { className: "stack", onSubmit: handleSaveSpeakers },
      speakers.map((sp) => {
        const key = sp.speaker_key;
        const currentVal = local.speakerDrafts.has(key)
          ? local.speakerDrafts.get(key)
          : sp.display_name || key;
        return el(
          "div",
          { className: "form-group" },
          el(
            "label",
            { className: "label row", style: "gap: 0.4rem;" },
            el("span", { className: "badge mono", textContent: key })
          ),
          el("input", {
            type: "text",
            className: "input",
            value: currentVal,
            placeholder: i18n.t("viewer.speaker_placeholder"),
            disabled: local.savingSpeakers,
            "aria-label": `${i18n.t("viewer.rename_speaker")} (${key})`,
            onInput: (event) => {
              local.speakerDrafts.set(key, event.target.value);
            },
          })
        );
      }),
      el(
        "div",
        { className: "row", style: "justify-content: flex-end;" },
        el("button", {
          type: "submit",
          className: "btn btn-primary btn-sm",
          disabled: local.savingSpeakers,
          textContent: tl(
            "viewer.save_and_regenerate",
            "Guardar nombres y regenerar",
            "Save names and regenerate"
          ),
        })
      )
    );

    speakersHost.appendChild(
      el(
        "div",
        { className: "card" },
        el(
          "div",
          { className: "card-header" },
          el("h3", { className: "card-title", textContent: i18n.t("viewer.speakers_title") })
        ),
        el("div", { className: "card-body" }, form)
      )
    );
  }

  function renderExportBar() {
    if (!exportHost) {
      return;
    }
    exportHost.textContent = "";
    const job = local.job;
    if (!job) {
      return;
    }

    const available = new Set(
      Array.isArray(job.available_formats) ? job.available_formats : []
    );

    const downloadButtons = EXPORT_FORMATS.map((fmt) => {
      const label = i18n.t(`viewer.download_${fmt}`);
      if (available.has(fmt)) {
        return el("a", {
          className: "btn btn-secondary btn-sm",
          href: api.jobDownloadUrl(job.id, fmt),
          download: "",
          textContent: label,
        });
      }
      return el("button", {
        type: "button",
        className: "btn btn-secondary btn-sm",
        disabled: true,
        textContent: label,
      });
    });

    const utilityButtons = el(
      "div",
      { className: "row", style: "gap: 0.45rem;" },
      el("button", {
        type: "button",
        className: "btn btn-secondary btn-sm",
        textContent: tl("viewer.copy_text", "Copiar texto", "Copy text"),
        onClick: () => void handleCopyText(),
      }),
      el("button", {
        type: "button",
        className: "btn btn-ghost btn-sm",
        disabled: local.reexporting,
        textContent: i18n.t("viewer.reexport"),
        onClick: () => void handleReexport(),
      })
    );

    const previewContent =
      local.vttPreviewStatus === "loading"
        ? i18n.t("common.loading")
        : local.vttPreviewText || buildFallbackVtt(job);

    const vttPreviewEl = el(
      "details",
      {
        open: local.vttPreviewOpen,
        onToggle: (event) => {
          local.vttPreviewOpen = Boolean(event.currentTarget.open);
        },
      },
      el("summary", {
        style: "cursor: pointer; font-weight: 600; font-size: 0.88rem;",
        textContent: tl(
          "viewer.vtt_preview",
          "Vista previa WebVTT (.vtt)",
          "WebVTT preview (.vtt)"
        ),
      }),
      el("pre", {
        className: "mono",
        style:
          "margin: 0.6rem 0 0; padding: 0.75rem; border-radius: var(--radius-sm); background: var(--bg); border: 1px solid var(--border); max-height: 240px; overflow: auto; font-size: 0.8rem; white-space: pre-wrap;",
        textContent: previewContent,
      })
    );

    exportHost.appendChild(
      el(
        "div",
        { className: "card" },
        el(
          "div",
          { className: "card-header" },
          el("h3", { className: "card-title", textContent: i18n.t("viewer.downloads_title") })
        ),
        el(
          "div",
          { className: "card-body" },
          el("div", { className: "row", style: "gap: 0.45rem;" }, downloadButtons),
          utilityButtons,
          vttPreviewEl
        )
      )
    );
  }

  function renderEmptyOrError() {
    detachAudioListeners();
    container.textContent = "";
    shellRoot = null;
    headerHost = null;
    bannerHost = null;
    playerHost = null;
    transcriptHost = null;
    speakersHost = null;
    exportHost = null;

    if (local.loading) {
      container.appendChild(
        el(
          "div",
          { className: "card" },
          el("p", { className: "muted", style: "margin: 0;", textContent: i18n.t("common.loading") })
        )
      );
      return;
    }

    if (local.loadError) {
      container.appendChild(
        el(
          "div",
          { className: "card" },
          el("div", { className: "alert alert-danger", textContent: local.loadError }),
          el(
            "div",
            { className: "card-footer" },
            el("button", {
              type: "button",
              className: "btn btn-secondary btn-sm",
              textContent: i18n.t("nav.library"),
              onClick: () => store.setState({ activeTab: "library" }),
            })
          )
        )
      );
      return;
    }

    container.appendChild(
      el(
        "div",
        { className: "card" },
        el(
          "div",
          { className: "card-body" },
          el("p", {
            className: "muted",
            style: "margin: 0;",
            textContent: i18n.t("viewer.no_job_selected"),
          }),
          el(
            "div",
            { className: "row" },
            el("button", {
              type: "button",
              className: "btn btn-primary btn-sm",
              textContent: i18n.t("nav.library"),
              onClick: () => store.setState({ activeTab: "library" }),
            })
          )
        )
      )
    );
  }

  function renderShell() {
    if (!local.job) {
      renderEmptyOrError();
      return;
    }

    detachAudioListeners();
    container.textContent = "";

    headerHost = el("div", { className: "stack" });
    bannerHost = el("div", { className: "stack" });
    playerHost = el("div", { className: "stack" });
    transcriptHost = el("div", { className: "stack" });
    speakersHost = el("div", { className: "stack" });
    exportHost = el("div", { className: "stack" });

    const mainGrid = el(
      "div",
      {
        className: "grid-2",
        style: "align-items: start;",
      },
      transcriptHost,
      el("div", { className: "stack" }, speakersHost, exportHost)
    );

    shellRoot = el(
      "div",
      { className: "stack", id: "viewer-root" },
      headerHost,
      bannerHost,
      playerHost,
      mainGrid
    );

    container.appendChild(shellRoot);

    renderHeader();
    renderBanner();
    renderPlayer();
    renderTranscript();
    renderSpeakers();
    renderExportBar();
  }

  const unsubscribeStore = store.subscribe((state, prevState) => {
    if (local.disposed || !container.isConnected) {
      return;
    }
    if (state.selectedJobId !== prevState.selectedJobId) {
      void loadJob(state.selectedJobId || null);
      return;
    }
    if (state.locale !== prevState.locale) {
      if (local.job) {
        renderHeader();
        renderBanner();
        renderTranscript();
        renderSpeakers();
        renderExportBar();
      } else {
        renderEmptyOrError();
      }
      return;
    }
    if (state.jobs !== prevState.jobs && local.jobId) {
      const snap = getStoreJobSnapshot(local.jobId);
      if (!snap) {
        if (local.job) {
          local.job = null;
          renderEmptyOrError();
        }
        return;
      }
      const nextStatus = snap.status || null;
      const nextDiar = snap.diarization_status || null;
      const transitioned =
        (prevSnapshotStatus !== null && nextStatus !== prevSnapshotStatus) ||
        (prevSnapshotDiarStatus !== null && nextDiar !== prevSnapshotDiarStatus);
      prevSnapshotStatus = nextStatus;
      prevSnapshotDiarStatus = nextDiar;

      if (transitioned && !ACTIVE_STATUSES.has(nextStatus)) {
        void loadJob(local.jobId, { preservePlayer: true });
      } else if (local.job) {
        renderHeader();
        renderBanner();
      }
    }
  });

  const handleJobEvent = (event) => {
    const jobId = event?.detail?.job_id;
    if (!local.disposed && jobId && jobId === local.jobId) {
      void loadJob(local.jobId, { preservePlayer: true });
    }
  };

  if (typeof window !== "undefined") {
    window.addEventListener("aurisdeck:job-completed", handleJobEvent);
    window.addEventListener("aurisdeck:job-error", handleJobEvent);
  }

  const initialSnap = local.jobId ? getStoreJobSnapshot(local.jobId) : null;
  prevSnapshotStatus = initialSnap?.status || null;
  prevSnapshotDiarStatus = initialSnap?.diarization_status || null;

  if (local.jobId) {
    void loadJob(local.jobId);
  } else {
    renderEmptyOrError();
  }

  const cleanup = () => {
    local.disposed = true;
    detachAudioListeners();
    unsubscribeStore();
    if (typeof window !== "undefined") {
      window.removeEventListener("aurisdeck:job-completed", handleJobEvent);
      window.removeEventListener("aurisdeck:job-error", handleJobEvent);
    }
  };

  containerCleanups.set(container, cleanup);
  return cleanup;
}

export default { mount };
