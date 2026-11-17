/**
 * HTTP API client with BASE_PATH discovery, FormData handling, and structured errors.
 */

const AUTH_REQUIRED_EVENT = "aurisdeck:auth-required";
const authRequiredListeners = new Set();
let customBasePath = null;

/**
 * Structured error thrown by the AurisDeck API client.
 */
export class ApiClientError extends Error {
  constructor({ status = 0, error_code = "INTERNAL_ERROR", error_params = {}, detail = "" }) {
    super(detail || error_code);
    this.name = "ApiClientError";
    this.status = status;
    this.error_code = error_code;
    this.error_params = error_params && typeof error_params === "object" ? error_params : {};
    this.detail = detail || error_code;
  }
}

/**
 * Normalize a base path to either "" or "/prefix" without trailing slash.
 * @param {string} raw
 */
function normalizeBasePath(raw) {
  if (!raw) {
    return "";
  }
  let cleaned = String(raw).trim();
  if (!cleaned || cleaned === "/") {
    return "";
  }
  if (!cleaned.startsWith("/")) {
    cleaned = `/${cleaned}`;
  }
  cleaned = cleaned.replace(/\/+$/, "");
  return cleaned === "/" ? "" : cleaned;
}

/**
 * Detect BASE_PATH from <meta name="base-path"> or window.location.pathname.
 */
export function getBasePath() {
  if (customBasePath !== null) {
    return customBasePath;
  }

  if (typeof document !== "undefined") {
    const meta = document.querySelector('meta[name="base-path"]');
    const metaContent = meta?.getAttribute("content");
    if (metaContent && metaContent.trim() !== "") {
      return normalizeBasePath(metaContent);
    }
  }

  if (typeof window !== "undefined" && window.location) {
    let pathname = window.location.pathname || "/";
    if (pathname.endsWith("/index.html")) {
      pathname = pathname.slice(0, -"/index.html".length);
    }
    return normalizeBasePath(pathname);
  }

  return "";
}

/**
 * Override detected BASE_PATH (useful for testing or explicit mounts).
 * @param {string | null} basePath
 */
export function setBasePath(basePath) {
  customBasePath = basePath === null ? null : normalizeBasePath(basePath);
}

/**
 * Resolve an API or static path against BASE_PATH.
 * @param {string} path
 */
export function apiUrl(path) {
  const base = getBasePath();
  const normalizedPath = String(path || "").startsWith("/")
    ? String(path)
    : `/${String(path || "")}`;
  return `${base}${normalizedPath}`;
}

/**
 * Register a listener invoked when the backend returns 401 AUTH_REQUIRED.
 * @param {(err: ApiClientError) => void} handler
 */
export function onAuthRequired(handler) {
  if (typeof handler !== "function") {
    throw new TypeError("onAuthRequired handler must be a function");
  }
  authRequiredListeners.add(handler);
  return () => {
    authRequiredListeners.delete(handler);
  };
}

function notifyAuthRequired(err) {
  for (const handler of Array.from(authRequiredListeners)) {
    try {
      handler(err);
    } catch (callbackErr) {
      console.error("onAuthRequired listener error:", callbackErr);
    }
  }
  if (typeof window !== "undefined" && typeof window.dispatchEvent === "function") {
    window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT, { detail: err }));
  }
}

function isPlainJsonBody(body) {
  if (body === undefined || body === null) {
    return false;
  }
  if (typeof FormData !== "undefined" && body instanceof FormData) {
    return false;
  }
  if (typeof Blob !== "undefined" && body instanceof Blob) {
    return false;
  }
  if (typeof URLSearchParams !== "undefined" && body instanceof URLSearchParams) {
    return false;
  }
  if (typeof body === "string") {
    return false;
  }
  return typeof body === "object";
}

function buildQuery(params = {}) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  }
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
}

/**
 * Perform a fetch request against the AurisDeck API with structured error parsing.
 * @param {string} path
 * @param {RequestInit & { body?: any }} options
 */
export async function request(path, options = {}) {
  const { body, headers: inputHeaders, ...restOptions } = options;
  const headers = new Headers(inputHeaders || {});

  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }

  let preparedBody = body;
  if (typeof FormData !== "undefined" && body instanceof FormData) {
    headers.delete("Content-Type");
  } else if (isPlainJsonBody(body)) {
    if (!headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    preparedBody = JSON.stringify(body);
  }

  const response = await fetch(apiUrl(path), {
    credentials: "same-origin",
    ...restOptions,
    headers,
    body: preparedBody,
  });

  if (response.status === 204) {
    return null;
  }

  const contentType = (response.headers.get("content-type") || "").toLowerCase();
  let parsed = null;
  if (contentType.includes("application/json")) {
    try {
      parsed = await response.json();
    } catch {
      parsed = null;
    }
  } else {
    const text = await response.text();
    parsed = text ? { detail: text } : null;
  }

  if (!response.ok) {
    const err = new ApiClientError({
      status: response.status,
      error_code: parsed?.error_code || `HTTP_${response.status}`,
      error_params: parsed?.error_params || {},
      detail: parsed?.detail || response.statusText || `HTTP ${response.status}`,
    });
    if (response.status === 401 && err.error_code === "AUTH_REQUIRED") {
      notifyAuthRequired(err);
    }
    throw err;
  }

  return parsed;
}

export const authStatus = () => request("/api/v1/auth/status");
export const login = (password) =>
  request("/api/v1/auth/login", { method: "POST", body: { password: String(password ?? "") } });
export const logout = () => request("/api/v1/auth/logout", { method: "POST" });

export const getMe = () => request("/api/v1/me");
export const patchMe = (patch) =>
  request("/api/v1/me", { method: "PATCH", body: patch });

export const getLocales = () => request("/api/v1/locales");
export const getLocale = (code) =>
  request(`/api/v1/locales/${encodeURIComponent(code)}.json`);

export const getProfiles = () => request("/api/v1/profiles");
export const getProfile = (id) =>
  request(`/api/v1/profiles/${encodeURIComponent(id)}`);
export const createProfile = (profile) =>
  request("/api/v1/profiles", { method: "POST", body: profile });
export const updateProfile = (id, profile) =>
  request(`/api/v1/profiles/${encodeURIComponent(id)}`, { method: "PUT", body: profile });
export const deleteProfile = (id) =>
  request(`/api/v1/profiles/${encodeURIComponent(id)}`, { method: "DELETE" });

export const getGlossaries = () => request("/api/v1/glossaries");
export const createGlossary = (glossary) =>
  request("/api/v1/glossaries", { method: "POST", body: glossary });
export const importGlossary = (formData) =>
  request("/api/v1/glossaries/import", { method: "POST", body: formData });
export const updateGlossary = (id, glossary) =>
  request(`/api/v1/glossaries/${encodeURIComponent(id)}`, { method: "PUT", body: glossary });
export const deleteGlossary = (id) =>
  request(`/api/v1/glossaries/${encodeURIComponent(id)}`, { method: "DELETE" });

export const getReplacements = () => request("/api/v1/replacements");
export const createReplacement = (dict) =>
  request("/api/v1/replacements", { method: "POST", body: dict });
export const updateReplacement = (id, dict) =>
  request(`/api/v1/replacements/${encodeURIComponent(id)}`, { method: "PUT", body: dict });
export const deleteReplacement = (id) =>
  request(`/api/v1/replacements/${encodeURIComponent(id)}`, { method: "DELETE" });

export const getDevices = () => request("/api/v1/system/devices");
export const getAdminSettings = () => request("/api/v1/admin/settings");
export const patchAdminSettings = (patch) =>
  request("/api/v1/admin/settings", { method: "PATCH", body: patch });
export const unloadModels = () =>
  request("/api/v1/system/unload-models", { method: "POST" });

export const listJobs = (params = {}) =>
  request(`/api/v1/jobs${buildQuery(params)}`);
export const getJob = (id) =>
  request(`/api/v1/jobs/${encodeURIComponent(id)}`);
export const createJob = (formData) =>
  request("/api/v1/jobs", { method: "POST", body: formData });
export const inspectZip = (formData) =>
  request("/api/v1/jobs/inspect-zip", { method: "POST", body: formData });
export const cancelJob = (id) =>
  request(`/api/v1/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
export const retryJob = (id, body = null) =>
  request(`/api/v1/jobs/${encodeURIComponent(id)}/retry`, {
    method: "POST",
    ...(body ? { body } : {}),
  });
export const diarizeJob = (id, body = null) =>
  request(`/api/v1/jobs/${encodeURIComponent(id)}/diarize`, {
    method: "POST",
    ...(body ? { body } : {}),
  });
export const patchSpeakers = (id, aliases) =>
  request(`/api/v1/jobs/${encodeURIComponent(id)}/speakers`, {
    method: "PATCH",
    body: { aliases },
  });
export const reexportJob = (id, body = null) =>
  request(`/api/v1/jobs/${encodeURIComponent(id)}/export`, {
    method: "POST",
    ...(body ? { body } : {}),
  });
export const deleteJob = (id) =>
  request(`/api/v1/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });

export const jobMediaUrl = (jobId, trackId) =>
  apiUrl(`/api/v1/jobs/${encodeURIComponent(jobId)}/media/${encodeURIComponent(trackId)}`);
export const jobDownloadUrl = (jobId, fmt) =>
  apiUrl(`/api/v1/jobs/${encodeURIComponent(jobId)}/download/${encodeURIComponent(fmt)}`);
export const batchDownloadUrl = (batchId, fmt = "vtt") =>
  apiUrl(`/api/v1/jobs/batches/${encodeURIComponent(batchId)}/download.zip?fmt=${encodeURIComponent(fmt)}`);

export const api = {
  getBasePath,
  setBasePath,
  apiUrl,
  onAuthRequired,
  request,
  authStatus,
  login,
  logout,
  getMe,
  patchMe,
  getLocales,
  getLocale,
  getProfiles,
  getProfile,
  createProfile,
  updateProfile,
  deleteProfile,
  getGlossaries,
  createGlossary,
  importGlossary,
  updateGlossary,
  deleteGlossary,
  getReplacements,
  createReplacement,
  updateReplacement,
  deleteReplacement,
  getDevices,
  getAdminSettings,
  patchAdminSettings,
  unloadModels,
  listJobs,
  getJob,
  createJob,
  inspectZip,
  cancelJob,
  retryJob,
  diarizeJob,
  patchSpeakers,
  reexportJob,
  deleteJob,
  jobMediaUrl,
  jobDownloadUrl,
  batchDownloadUrl,
};

export default api;
