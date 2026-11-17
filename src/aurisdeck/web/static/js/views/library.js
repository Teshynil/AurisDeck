/**
 * Library view: job history sorted by recent date, status & text filters,
 * visual batch grouping with ZIP download, and Open / Retry / Delete actions.
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

const COMPLETED_STATUSES = new Set(["completed", "completed_partial"]);
const VALID_FILTERS = ["all", "active", "completed", "partial", "failed"];

const containerCleanups = new WeakMap();

const viewState = {
  filter: "all",
  query: "",
  busyRetries: new Set(),
  busyDeletes: new Set(),
};

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

function parseTimestamp(iso) {
  if (!iso) {
    return 0;
  }
  const ms = Date.parse(String(iso));
  return Number.isFinite(ms) ? ms : 0;
}

function sortJobsByRecent(jobs) {
  return [...(Array.isArray(jobs) ? jobs : [])].sort((a, b) => {
    const diff = parseTimestamp(b?.created_at) - parseTimestamp(a?.created_at);
    if (diff !== 0) {
      return diff;
    }
    return String(b?.created_at || "").localeCompare(String(a?.created_at || ""));
  });
}

function formatDate(iso) {
  if (!iso) {
    return "—";
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

function matchesStatusFilter(job, filter) {
  const status = String(job?.status || "");
  switch (filter) {
    case "active":
      return ACTIVE_STATUSES.has(status);
    case "completed":
      return status === "completed";
    case "partial":
      return status === "completed_partial";
    case "failed":
      return status === "failed";
    case "all":
    default:
      return true;
  }
}

function matchesSearchQuery(job, rawQuery) {
  const query = String(rawQuery || "").trim().toLowerCase();
  if (!query) {
    return true;
  }
  const parts = [job?.title, job?.id];
  if (Array.isArray(job?.speakers)) {
    for (const sp of job.speakers) {
      parts.push(sp?.display_name, sp?.speaker_key);
    }
  }
  if (Array.isArray(job?.tracks)) {
    for (const track of job.tracks) {
      parts.push(track?.original_filename, track?.speaker_key);
    }
  }
  return parts.some((part) => part && String(part).toLowerCase().includes(query));
}

function countByFilter(jobs) {
  const counts = { all: 0, active: 0, completed: 0, partial: 0, failed: 0 };
  for (const job of jobs) {
    counts.all += 1;
    if (ACTIVE_STATUSES.has(job?.status)) {
      counts.active += 1;
    } else if (job?.status === "completed") {
      counts.completed += 1;
    } else if (job?.status === "completed_partial") {
      counts.partial += 1;
    } else if (job?.status === "failed") {
      counts.failed += 1;
    }
  }
  return counts;
}

function getFilterLabel(filterId) {
  switch (filterId) {
    case "all":
      return i18n.t("library.filter_all");
    case "active":
      return i18n.t("library.filter_active");
    case "completed":
      return i18n.t("library.filter_completed");
    case "partial":
      return i18n.t("stages.completed_partial");
    case "failed":
      return i18n.t("library.filter_failed");
    default:
      return filterId;
  }
}

function resolveBatchFormat(batchJobs) {
  const supported = new Set(["vtt", "srt", "md", "json"]);
  for (const job of batchJobs) {
    const fmts = job?.effective_config?.output?.formats;
    if (Array.isArray(fmts)) {
      if (fmts.includes("vtt")) {
        return "vtt";
      }
      const firstValid = fmts.find((fmt) => supported.has(fmt));
      if (firstValid) {
        return firstValid;
      }
    }
  }
  return "vtt";
}

function mergeServerJobs(serverJobs) {
  store.setState((current) => {
    const existingById = new Map(
      (Array.isArray(current.jobs) ? current.jobs : []).map((item) => [item.id, item])
    );
    const merged = serverJobs.map((job) => {
      const prev = existingById.get(job.id);
      if (prev && ACTIVE_STATUSES.has(job.status) && ACTIVE_STATUSES.has(prev.status)) {
        return {
          ...job,
          ...(prev.progress ? { progress: prev.progress } : {}),
          ...(prev.stage ? { stage: prev.stage } : {}),
        };
      }
      return job;
    });
    return { jobs: merged };
  });
}

function upsertSingleJob(returned) {
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

/**
 * Mount the Library view into `container` (#view-library).
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

  let disposed = false;
  let controlsHost = null;
  let listHost = null;
  let headerTitleEl = null;
  let filterButtons = new Map();
  let searchInputEl = null;

  function openJobInViewer(jobId) {
    store.setState({ selectedJobId: jobId, activeTab: "viewer" });
  }

  async function handleRetry(jobId) {
    if (!jobId || viewState.busyRetries.has(jobId)) {
      return;
    }
    viewState.busyRetries.add(jobId);
    renderList();
    try {
      const updated = await api.retryJob(jobId);
      upsertSingleJob(updated);
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
      viewState.busyRetries.delete(jobId);
      if (!disposed) {
        renderList();
      }
    }
  }

  async function handleDelete(jobId) {
    if (!jobId || viewState.busyDeletes.has(jobId)) {
      return;
    }
    const confirmed =
      typeof window !== "undefined" && typeof window.confirm === "function"
        ? window.confirm(i18n.t("library.delete_confirm"))
        : true;
    if (!confirmed) {
      return;
    }

    viewState.busyDeletes.add(jobId);
    renderList();
    try {
      await api.deleteJob(jobId);
      store.setState((current) => {
        const remaining = (Array.isArray(current.jobs) ? current.jobs : []).filter(
          (item) => item.id !== jobId
        );
        return {
          jobs: remaining,
          ...(current.selectedJobId === jobId ? { selectedJobId: null } : {}),
        };
      });
      store.addToast({
        type: "success",
        title: i18n.t("common.delete"),
        message: i18n.t("common.success"),
      });
    } catch (err) {
      store.addToast({
        type: "danger",
        title: i18n.t("common.error"),
        message: i18n.translateError(err),
      });
    } finally {
      viewState.busyDeletes.delete(jobId);
      if (!disposed) {
        renderList();
      }
    }
  }

  function buildJobCard(job, { inBatch = false } = {}) {
    const isActive = ACTIVE_STATUSES.has(job.status);
    const isBusy = viewState.busyRetries.has(job.id) || viewState.busyDeletes.has(job.id);
    const languageLabel =
      job.language_detected || job.language_requested || i18n.t("common.auto");

    const speakerChips = [];
    if (Array.isArray(job.speakers) && job.speakers.length > 0) {
      for (const sp of job.speakers) {
        const label = sp?.display_name || sp?.speaker_key;
        if (label) {
          speakerChips.push(el("span", { className: "chip", textContent: label }));
        }
      }
    }

    const metaRow = el(
      "div",
      { className: "row", style: "gap: 0.45rem;" },
      createStageBadge(job),
      el("span", { className: "badge", textContent: getModeLabel(job) }),
      el("span", { className: "badge mono", textContent: languageLabel }),
      el("span", { className: "help-text", textContent: formatDate(job.created_at) })
    );

    const progressEl = (() => {
      if (!isActive || !job.progress) {
        return null;
      }
      const percent = Math.max(0, Math.min(100, Number(job.progress.percent || 0)));
      const fill = el("div", { className: "progress-fill" });
      fill.style.width = `${percent}%`;
      return el(
        "div",
        { className: "stack", style: "gap: 0.25rem;" },
        el("div", { className: "progress-bar" }, fill),
        el(
          "div",
          { className: "progress-meta" },
          el("span", { textContent: `${percent}%` }),
          job.progress.device_id
            ? el("span", { className: "mono", textContent: job.progress.device_id })
            : null
        )
      );
    })();

    const errorEl =
      (job.status === "failed" || job.status === "completed_partial") &&
      (job.error_code || job.error_detail)
        ? el("div", {
            className: job.status === "failed" ? "form-error" : "help-text",
            textContent: i18n.translateError({
              error_code: job.error_code,
              error_params: job.error_params,
              detail: job.error_detail,
            }),
          })
        : null;

    const actions = el(
      "div",
      { className: "row", style: "gap: 0.45rem;" },
      el("button", {
        type: "button",
        className: "btn btn-primary btn-sm",
        disabled: isBusy,
        textContent: tl("library.open_action", "Abrir", "Open"),
        onClick: () => openJobInViewer(job.id),
      }),
      !isActive
        ? el("button", {
            type: "button",
            className: "btn btn-secondary btn-sm",
            disabled: isBusy,
            textContent: i18n.t("common.retry"),
            onClick: () => void handleRetry(job.id),
          })
        : null,
      el("button", {
        type: "button",
        className: "btn btn-danger btn-sm",
        disabled: isBusy,
        textContent: i18n.t("common.delete"),
        onClick: () => void handleDelete(job.id),
      })
    );

    const card = el(
      "article",
      {
        className: inBatch ? "file-item" : "card",
        style: inBatch
          ? "flex-direction: column; align-items: stretch; padding: 0.85rem 1rem; background: var(--surface);"
          : "",
      },
      el(
        "div",
        { className: "row-between" },
        el(
          "div",
          { className: "stack", style: "gap: 0.35rem; min-width: 0; flex: 1 1 260px;" },
          el("button", {
            type: "button",
            className: "btn btn-ghost",
            style: "justify-content: flex-start; padding: 0; font-size: 1rem; text-align: left;",
            textContent: job.title || job.id,
            onClick: () => openJobInViewer(job.id),
          }),
          metaRow
        ),
        actions
      ),
      progressEl,
      speakerChips.length > 0 ? el("div", { className: "chips" }, speakerChips) : null,
      errorEl
    );

    return card;
  }

  function buildBatchCard(batchId, batchJobs) {
    const hasDownloadable = batchJobs.some((item) => COMPLETED_STATUSES.has(item.status));
    const batchFmt = resolveBatchFormat(batchJobs);
    const downloadLabel = tl(
      "library.download_batch",
      "Descargar lote (.zip)",
      "Download batch (.zip)"
    );

    const header = el(
      "div",
      { className: "card-header" },
      el(
        "div",
        { className: "row", style: "gap: 0.5rem;" },
        el("span", {
          className: "badge badge-info",
          textContent: i18n.t("simple.modes.batch.title"),
        }),
        el("span", {
          className: "help-text",
          textContent: i18n.t("common.items_count", { count: batchJobs.length }),
        }),
        el("span", { className: "help-text mono", textContent: batchId.slice(0, 8) })
      ),
      hasDownloadable
        ? el("a", {
            className: "btn btn-secondary btn-sm",
            href: api.batchDownloadUrl(batchId, batchFmt),
            download: "",
            textContent: downloadLabel,
          })
        : el("button", {
            type: "button",
            className: "btn btn-secondary btn-sm",
            disabled: true,
            textContent: downloadLabel,
          })
    );

    const body = el(
      "div",
      { className: "card-body" },
      batchJobs.map((job) => buildJobCard(job, { inBatch: true }))
    );

    const groupCard = el("section", { className: "card" }, header, body);
    groupCard.style.borderLeft = "4px solid var(--primary)";
    return groupCard;
  }

  function updateControls() {
    if (!controlsHost) {
      return;
    }
    const allJobs = Array.isArray(store.getState().jobs) ? store.getState().jobs : [];
    const counts = countByFilter(allJobs);

    if (headerTitleEl) {
      headerTitleEl.textContent = i18n.t("library.title");
    }
    for (const [filterId, btn] of filterButtons.entries()) {
      const active = viewState.filter === filterId;
      btn.className = active ? "sub-tab is-active" : "sub-tab";
      btn.setAttribute("aria-selected", active ? "true" : "false");
      btn.textContent = `${getFilterLabel(filterId)} (${counts[filterId] ?? 0})`;
    }
    if (searchInputEl) {
      const placeholder = i18n.t("library.search_placeholder");
      searchInputEl.placeholder = placeholder;
      searchInputEl.setAttribute("aria-label", placeholder);
    }
  }

  function renderControls() {
    if (!controlsHost) {
      return;
    }
    controlsHost.textContent = "";
    filterButtons = new Map();

    headerTitleEl = el("h2", { className: "section-title" });
    const header = el("div", { className: "section-header" }, el("div", null, headerTitleEl));

    const tabs = el(
      "div",
      { className: "sub-tabs", role: "tablist" },
      VALID_FILTERS.map((filterId) => {
        const btn = el("button", {
          type: "button",
          role: "tab",
          onClick: () => {
            if (viewState.filter === filterId) {
              return;
            }
            viewState.filter = filterId;
            updateControls();
            renderList();
          },
        });
        filterButtons.set(filterId, btn);
        return btn;
      })
    );

    searchInputEl = el("input", {
      id: "library-search-input",
      type: "search",
      className: "input",
      value: viewState.query,
      onInput: (event) => {
        viewState.query = event.target.value;
        renderList();
      },
    });

    const toolbar = el(
      "div",
      { className: "card" },
      el("div", { className: "stack" }, tabs, searchInputEl)
    );

    controlsHost.appendChild(header);
    controlsHost.appendChild(toolbar);
    updateControls();
  }

  function renderList() {
    if (!listHost) {
      return;
    }
    listHost.textContent = "";

    const sorted = sortJobsByRecent(store.getState().jobs);
    const filtered = sorted.filter(
      (job) =>
        matchesStatusFilter(job, viewState.filter) &&
        matchesSearchQuery(job, viewState.query)
    );

    if (filtered.length === 0) {
      listHost.appendChild(
        el(
          "div",
          { className: "card" },
          el("p", {
            className: "muted",
            style: "margin: 0;",
            textContent: i18n.t("library.empty"),
          })
        )
      );
      return;
    }

    const renderedBatches = new Set();
    for (const job of filtered) {
      if (job.batch_id) {
        if (renderedBatches.has(job.batch_id)) {
          continue;
        }
        renderedBatches.add(job.batch_id);
        const batchJobs = filtered.filter((item) => item.batch_id === job.batch_id);
        listHost.appendChild(buildBatchCard(job.batch_id, batchJobs));
      } else {
        listHost.appendChild(buildJobCard(job));
      }
    }
  }

  async function refreshJobsFromApi() {
    const auth = store.getState().authStatus;
    if (auth?.auth_mode === "password" && !auth?.authenticated) {
      return;
    }
    try {
      const res = await api.listJobs({ limit: 200 });
      if (!disposed && Array.isArray(res?.jobs)) {
        mergeServerJobs(res.jobs);
      }
    } catch {
      // Initial list from store remains visible if refresh fails transiently.
    }
  }

  container.textContent = "";
  controlsHost = el("div", { className: "stack", id: "library-controls-root" });
  listHost = el("div", { className: "stack", id: "library-list-root" });
  container.appendChild(controlsHost);
  container.appendChild(listHost);

  renderControls();
  renderList();
  void refreshJobsFromApi();

  const unsubscribeStore = store.subscribe((state, prevState) => {
    if (disposed || !container.isConnected) {
      return;
    }
    if (state.locale !== prevState.locale) {
      updateControls();
      renderList();
      return;
    }
    if (state.jobs !== prevState.jobs) {
      updateControls();
      renderList();
    }
  });

  const cleanup = () => {
    disposed = true;
    unsubscribeStore();
  };
  containerCleanups.set(container, cleanup);
  return cleanup;
}

export default { mount };
