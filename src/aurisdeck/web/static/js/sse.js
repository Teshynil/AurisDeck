/**
 * Single multiplexed Server-Sent Events (SSE) client synchronized with store.
 */

import { apiUrl, getDevices, getJob, listJobs } from "./api.js";
import { t, translateError } from "./i18n.js";
import { store } from "./store.js";

let eventSource = null;
let hasConnectedOnce = false;
let resyncInFlight = false;

function parseEnvelope(event) {
  try {
    return JSON.parse(event.data);
  } catch {
    return null;
  }
}

function upsertJob(jobId, patch) {
  if (!jobId) {
    return;
  }
  store.setState((current) => {
    const list = Array.isArray(current.jobs) ? current.jobs : [];
    const idx = list.findIndex((item) => item.id === jobId);
    if (idx === -1) {
      return {
        jobs: [{ id: jobId, ...patch }, ...list],
      };
    }
    const next = list.slice();
    next[idx] = { ...next[idx], ...patch };
    return { jobs: next };
  });
}

function updateDevice(deviceId, patch) {
  if (!deviceId) {
    return;
  }
  store.setState((current) => {
    const list = Array.isArray(current.devices) ? current.devices : [];
    const idx = list.findIndex((item) => item.id === deviceId);
    if (idx === -1) {
      return current;
    }
    const next = list.slice();
    next[idx] = { ...next[idx], ...patch };
    return { devices: next };
  });
}

function clearDeviceJob(jobId) {
  if (!jobId) {
    return;
  }
  store.setState((current) => {
    const list = Array.isArray(current.devices) ? current.devices : [];
    let changed = false;
    const next = list.map((dev) => {
      if (dev.active_job_id === jobId) {
        changed = true;
        return {
          ...dev,
          active_job_id: null,
          track_id: null,
          stage: null,
          state: "idle",
          percent: 0,
          rtfx: null,
          processed_ms: 0,
          total_ms: 0,
        };
      }
      return dev;
    });
    return changed ? { devices: next } : current;
  });
}

function emitWindowEvent(name, detail) {
  if (typeof window !== "undefined" && typeof window.dispatchEvent === "function") {
    window.dispatchEvent(new CustomEvent(name, { detail }));
  }
}

function hydrateJobsFromSnapshot(rawJobs, rawDevices) {
  const devicesByJobTrack = new Map();
  for (const dev of rawDevices) {
    if (dev && dev.active_job_id && dev.track_id) {
      devicesByJobTrack.set(`${dev.active_job_id}:${dev.track_id}`, dev);
    }
  }
  return rawJobs.map((job) => {
    const tracks = Array.isArray(job?.tracks) ? job.tracks : [];
    if (!tracks.length) {
      return job;
    }
    const trackProgress = { ...(job.track_progress || {}) };
    let hasTrackData = false;
    for (const t of tracks) {
      const dev = devicesByJobTrack.get(`${job.id}:${t.id}`);
      if (t.status === "completed" || t.asr_status === "completed") {
        trackProgress[t.id] = {
          ...(trackProgress[t.id] || {}),
          track_id: t.id,
          stage: "completed",
          percent: 100,
        };
        hasTrackData = true;
      } else if (dev || (t.status && t.status !== "queued")) {
        trackProgress[t.id] = {
          ...(trackProgress[t.id] || {}),
          track_id: t.id,
          device_id: dev?.id ?? t.device_id ?? null,
          stage: dev?.stage ?? t.status ?? "transcribing",
          processed_ms: dev?.processed_ms ?? t.processed_ms ?? 0,
          total_ms: dev?.total_ms ?? t.total_ms ?? 0,
          rtfx: dev?.rtfx ?? t.rtfx ?? null,
          percent: typeof dev?.percent === "number" ? dev.percent : Number(t.percent || 0),
        };
        hasTrackData = true;
      }
    }
    return hasTrackData ? { ...job, track_progress: trackProgress } : job;
  });
}

async function resyncState() {
  const { authStatus } = store.getState();
  if (authStatus && authStatus.auth_mode === "password" && !authStatus.authenticated) {
    return;
  }
  if (resyncInFlight) {
    return;
  }
  resyncInFlight = true;
  try {
    const [jobsRes, devicesRes] = await Promise.all([listJobs(), getDevices()]);
    const rawJobs = Array.isArray(jobsRes?.jobs) ? jobsRes.jobs : [];
    const rawDevices = Array.isArray(devicesRes?.devices) ? devicesRes.devices : [];
    store.setState({
      jobs: hydrateJobsFromSnapshot(rawJobs, rawDevices),
      devices: rawDevices,
    });
  } catch (err) {
    console.warn("SSE resync failed:", err);
  } finally {
    resyncInFlight = false;
  }
}

async function hydrateMissingJob(jobId) {
  try {
    const full = await getJob(jobId);
    if (full && full.id) {
      upsertJob(full.id, full);
    }
  } catch {
    // Ignore transient hydration error; list resync will recover.
  }
}

function handleJobCreated(envelope) {
  const jobId = envelope?.job_id;
  const payload = envelope?.payload || {};
  const existed = store.getState().jobs.some((item) => item.id === jobId);
  upsertJob(jobId, {
    status: payload.status || "queued",
    created_at: envelope.ts,
    updated_at: envelope.ts,
    ...payload,
  });
  if (!existed && jobId && !payload.title) {
    void hydrateMissingJob(jobId);
  }
}

function handleJobStage(envelope) {
  const jobId = envelope?.job_id;
  const payload = envelope?.payload || {};
  const nextStatus = payload.status || payload.stage;
  const trackId = payload.track_id ?? null;

  store.setState((current) => {
    const list = Array.isArray(current.jobs) ? current.jobs : [];
    const idx = list.findIndex((item) => item.id === jobId);
    if (idx === -1) {
      return {
        jobs: [
          {
            id: jobId,
            ...(nextStatus ? { status: nextStatus } : {}),
            stage: payload.stage || nextStatus || null,
            device_id: payload.device_id ?? null,
            updated_at: envelope.ts,
          },
          ...list,
        ],
      };
    }
    const existing = list[idx];
    const nextTrackProgress = { ...(existing.track_progress || {}) };
    if (trackId) {
      const prevTrack = nextTrackProgress[trackId] || {};
      const stageValue = payload.stage || nextStatus || prevTrack.stage || null;
      nextTrackProgress[trackId] = {
        ...prevTrack,
        track_id: trackId,
        device_id: payload.device_id ?? prevTrack.device_id ?? null,
        stage: stageValue,
        ...(stageValue === "completed" ? { percent: 100 } : {}),
        updated_at: envelope.ts,
      };
    }
    const next = list.slice();
    next[idx] = {
      ...existing,
      ...(nextStatus ? { status: nextStatus } : {}),
      stage: payload.stage || nextStatus || null,
      device_id: payload.device_id ?? null,
      track_progress: nextTrackProgress,
      updated_at: envelope.ts,
    };
    return { jobs: next };
  });

  if (payload.device_id) {
    updateDevice(payload.device_id, {
      active_job_id: jobId,
      ...(trackId ? { track_id: trackId } : {}),
      stage: payload.stage || nextStatus || null,
      state: "busy",
    });
  }
}

function handleJobProgress(envelope) {
  const jobId = envelope?.job_id;
  const payload = envelope?.payload || {};
  const processed = Number(payload.processed_ms || 0);
  const total = Number(payload.total_ms || 0);
  const percent = total > 0 ? Math.min(100, Math.round((processed / total) * 100)) : 0;
  const trackId = payload.track_id ?? null;
  const trackKey = trackId || "default";

  store.setState((current) => {
    const list = Array.isArray(current.jobs) ? current.jobs : [];
    const idx = list.findIndex((item) => item.id === jobId);
    const trackEntry = {
      device_id: payload.device_id ?? null,
      track_id: trackId,
      processed_ms: processed,
      total_ms: total,
      rtfx: payload.rtfx ?? null,
      percent,
      stage: percent >= 100 ? "completed" : "transcribing",
      updated_at: envelope.ts,
    };

    if (idx === -1) {
      return {
        jobs: [
          {
            id: jobId,
            progress: trackEntry,
            track_progress: { [trackKey]: trackEntry },
            updated_at: envelope.ts,
          },
          ...list,
        ],
      };
    }

    const existing = list[idx];
    const nextTrackProgress = {
      ...(existing.track_progress || {}),
      [trackKey]: { ...(existing.track_progress?.[trackKey] || {}), ...trackEntry },
    };

    const tracks = Array.isArray(existing.tracks) ? existing.tracks : [];
    let globalPercent = percent;
    if (tracks.length > 1) {
      const sum = tracks.reduce((acc, t) => {
        const tp = nextTrackProgress[t.id];
        if (tp && typeof tp.percent === "number") {
          return acc + tp.percent;
        }
        return acc + (t.status === "completed" ? 100 : 0);
      }, 0);
      globalPercent = Math.round(sum / tracks.length);
    }

    const next = list.slice();
    next[idx] = {
      ...existing,
      track_progress: nextTrackProgress,
      progress: {
        ...trackEntry,
        percent: globalPercent,
      },
      updated_at: envelope.ts,
    };
    return { jobs: next };
  });

  if (payload.device_id) {
    updateDevice(payload.device_id, {
      active_job_id: jobId,
      track_id: trackId,
      state: "busy",
      stage: "transcribing",
      processed_ms: processed,
      total_ms: total,
      rtfx: payload.rtfx ?? null,
      percent,
    });
  }
}

function handleJobCompleted(envelope) {
  const jobId = envelope?.job_id;
  const payload = envelope?.payload || {};
  const status = payload.status || "completed";
  const isPartial = status === "completed_partial" || envelope.type === "job.completed_partial";

  upsertJob(jobId, {
    status,
    diarization_status: payload.diarization_status,
    error_code: payload.error_code ?? null,
    error_params: payload.error_params ?? null,
    error_detail: payload.detail ?? null,
    completed_at: envelope.ts,
    finished_at_ms: Date.now(),
    updated_at: envelope.ts,
    progress: null,
  });
  clearDeviceJob(jobId);

  const message = isPartial
    ? translateError({
        error_code: payload.error_code || "DIARIZATION_INTERRUPTED",
        error_params: payload.error_params || {},
        detail: payload.detail || "",
      })
    : t("stages.completed");

  store.addToast({
    type: isPartial ? "warning" : "success",
    title: t(isPartial ? "stages.completed_partial" : "stages.completed"),
    message,
  });
  emitWindowEvent("aurisdeck:job-completed", envelope);
}

function handleJobError(envelope) {
  const jobId = envelope?.job_id;
  const payload = envelope?.payload || {};

  upsertJob(jobId, {
    status: "failed",
    error_code: payload.error_code || "INTERNAL_ERROR",
    error_params: payload.error_params || {},
    error_detail: payload.detail || "",
    finished_at_ms: Date.now(),
    updated_at: envelope.ts,
    progress: null,
  });
  clearDeviceJob(jobId);

  store.addToast({
    type: "danger",
    title: t("stages.failed"),
    message: translateError(payload),
  });
  emitWindowEvent("aurisdeck:job-error", envelope);
}

function handleJobCancelled(envelope) {
  const jobId = envelope?.job_id;
  upsertJob(jobId, {
    status: "cancelled",
    finished_at_ms: Date.now(),
    updated_at: envelope.ts,
    progress: null,
  });
  clearDeviceJob(jobId);
}

function handleDeviceStatus(envelope) {
  const payload = envelope?.payload || {};
  if (Array.isArray(payload.devices)) {
    store.setState({ devices: payload.devices });
    return;
  }
  if (payload.id) {
    store.setState((current) => {
      const list = Array.isArray(current.devices) ? current.devices : [];
      const idx = list.findIndex((item) => item.id === payload.id);
      if (idx === -1) {
        return { devices: [...list, payload] };
      }
      const next = list.slice();
      next[idx] = { ...next[idx], ...payload };
      return { devices: next };
    });
  }
}

function handleWorkerUnloaded(envelope) {
  const payload = envelope?.payload || {};
  if (payload.device_id) {
    updateDevice(payload.device_id, {
      state: "idle",
      active_job_id: null,
      stage: null,
      loaded_model: null,
    });
  }
}

/**
 * Open the single multiplexed EventSource connection to /api/v1/events.
 */
export function connectSSE() {
  if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
    return eventSource;
  }

  hasConnectedOnce = false;
  const es = new EventSource(apiUrl("/api/v1/events"));
  eventSource = es;

  es.onopen = () => {
    if (hasConnectedOnce) {
      void resyncState();
    } else {
      hasConnectedOnce = true;
    }
  };

  es.addEventListener("job.created", (ev) => {
    const env = parseEnvelope(ev);
    if (env) handleJobCreated(env);
  });

  es.addEventListener("job.stage", (ev) => {
    const env = parseEnvelope(ev);
    if (env) handleJobStage(env);
  });

  es.addEventListener("job.progress", (ev) => {
    const env = parseEnvelope(ev);
    if (env) handleJobProgress(env);
  });

  es.addEventListener("job.completed", (ev) => {
    const env = parseEnvelope(ev);
    if (env) handleJobCompleted(env);
  });

  es.addEventListener("job.completed_partial", (ev) => {
    const env = parseEnvelope(ev);
    if (env) handleJobCompleted(env);
  });

  es.addEventListener("job.error", (ev) => {
    const env = parseEnvelope(ev);
    if (env) handleJobError(env);
  });

  es.addEventListener("job.cancelled", (ev) => {
    const env = parseEnvelope(ev);
    if (env) handleJobCancelled(env);
  });

  es.addEventListener("device.status", (ev) => {
    const env = parseEnvelope(ev);
    if (env) handleDeviceStatus(env);
  });

  es.addEventListener("system.worker_unloaded", (ev) => {
    const env = parseEnvelope(ev);
    if (env) handleWorkerUnloaded(env);
  });

  es.addEventListener("model.pull.progress", (ev) => {
    const env = parseEnvelope(ev);
    const payload = env?.payload;
    if (!payload) return;
    if (payload.status === "completed" || payload.status === "failed") {
      store.setState({ modelPull: null });
    } else {
      store.setState({ modelPull: payload });
    }
  });

  return es;
}

/**
 * Close the active EventSource connection cleanly (e.g. on logout).
 */
export function disconnectSSE() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  hasConnectedOnce = false;
}

export function isSSEConnected() {
  return Boolean(eventSource && eventSource.readyState === EventSource.OPEN);
}

export const sse = {
  connect: connectSSE,
  disconnect: disconnectSSE,
  isConnected: isSSEConnected,
  resync: resyncState,
};

export default sse;
