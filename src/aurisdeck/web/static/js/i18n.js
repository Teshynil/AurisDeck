/**
 * Runtime JSON internationalization engine with fallback, Intl.PluralRules, and safe DOM binding.
 */

import { apiUrl, getBasePath } from "./api.js";
import { store } from "./store.js";

const loadedDictionaries = new Map();
let currentLocale = "es";
let fallbackLocale = "en";
let availableLocales = [];

export const detectBasePath = getBasePath;

function resolvePath(obj, dotKey) {
  if (!obj || typeof obj !== "object" || !dotKey) {
    return undefined;
  }
  const parts = String(dotKey).split(".");
  let cursor = obj;
  for (const part of parts) {
    if (cursor === null || typeof cursor !== "object" || !(part in cursor)) {
      return undefined;
    }
    cursor = cursor[part];
  }
  return cursor;
}

function interpolate(template, params = {}) {
  return String(template).replace(/\{([a-zA-Z0-9_]+)\}/g, (match, key) => {
    if (
      params &&
      Object.prototype.hasOwnProperty.call(params, key) &&
      params[key] !== undefined &&
      params[key] !== null
    ) {
      return String(params[key]);
    }
    return match;
  });
}

function selectPlural(pluralObj, count, localeCode) {
  let category = Number(count) === 1 ? "one" : "other";
  try {
    category = new Intl.PluralRules(localeCode).select(Number(count));
  } catch {
    category = Number(count) === 1 ? "one" : "other";
  }
  if (typeof pluralObj[category] === "string") {
    return pluralObj[category];
  }
  if (typeof pluralObj.other === "string") {
    return pluralObj.other;
  }
  if (typeof pluralObj.one === "string") {
    return pluralObj.one;
  }
  return undefined;
}

async function fetchLocaleJson(code) {
  const response = await fetch(apiUrl(`/api/v1/locales/${encodeURIComponent(code)}.json`), {
    credentials: "same-origin",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Failed to load locale '${code}' (${response.status})`);
  }
  return response.json();
}

/**
 * Load the list of available locales from GET /api/v1/locales.
 */
export async function loadLocales() {
  const response = await fetch(apiUrl("/api/v1/locales"), {
    credentials: "same-origin",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Failed to list locales (${response.status})`);
  }
  const data = await response.json();
  availableLocales = Array.isArray(data?.locales) ? data.locales : [];
  store.setState({ locales: availableLocales });
  return availableLocales;
}

/**
 * Load a locale dictionary (plus fallback 'en'/'es' if different) and update document metadata.
 * @param {string} code
 */
export async function load(code) {
  const requested = String(code || "es").trim() || "es";
  const primaryDict = await fetchLocaleJson(requested);
  const meta = primaryDict?._meta || {};
  const resolvedCode = String(meta.code || requested);
  loadedDictionaries.set(resolvedCode, primaryDict);

  const fallbackCandidate = resolvedCode === "en" ? "es" : "en";
  fallbackLocale = fallbackCandidate;
  if (!loadedDictionaries.has(fallbackCandidate)) {
    try {
      const fbDict = await fetchLocaleJson(fallbackCandidate);
      loadedDictionaries.set(fallbackCandidate, fbDict);
    } catch {
      // Optional fallback locale may not exist in minimal setups.
    }
  }

  currentLocale = resolvedCode;
  if (typeof document !== "undefined" && document.documentElement) {
    document.documentElement.lang = resolvedCode;
    document.documentElement.dir = meta.direction === "rtl" ? "rtl" : "ltr";
  }

  store.setState({ locale: resolvedCode });
  if (typeof document !== "undefined") {
    applyStaticTranslations(document);
  }
  return primaryDict;
}

/**
 * Translate a dot-separated key with optional pluralization and parameter interpolation.
 * Never returns HTML; interpolation produces plain text only.
 * @param {string} key
 * @param {Record<string, any>} params
 */
export function t(key, params = {}) {
  const primaryDict = loadedDictionaries.get(currentLocale);
  const fallbackDict = loadedDictionaries.get(fallbackLocale);

  let value = resolvePath(primaryDict, key);
  let resolvedLocale = currentLocale;

  if (value === undefined && fallbackDict) {
    value = resolvePath(fallbackDict, key);
    resolvedLocale = fallbackLocale;
  }

  if (value === undefined || value === null) {
    return String(key);
  }

  if (typeof value === "object" && !Array.isArray(value) && params.count !== undefined) {
    const selected = selectPlural(value, params.count, resolvedLocale);
    if (selected === undefined) {
      return String(key);
    }
    return interpolate(selected, params);
  }

  if (typeof value !== "string") {
    return String(key);
  }

  return interpolate(value, params);
}

/**
 * Translate a structured API/SSE error object ({ error_code, error_params, detail }).
 * @param {any} errObj
 */
export function translateError(errObj) {
  if (!errObj) {
    return t("errors.INTERNAL_ERROR");
  }
  if (typeof errObj === "string") {
    return errObj;
  }

  const code = errObj.error_code ? String(errObj.error_code) : "";
  if (code) {
    const key = `errors.${code}`;
    const translated = t(key, errObj.error_params || {});
    if (translated !== key) {
      return translated;
    }
  }

  return String(
    errObj.detail || errObj.error_detail || errObj.message || code || errObj
  );
}

/**
 * Apply translations to static DOM elements using textContent and safe attributes only.
 * @param {ParentNode} root
 */
export function applyStaticTranslations(root = document) {
  if (!root || typeof root.querySelectorAll !== "function") {
    return;
  }

  for (const el of root.querySelectorAll("[data-i18n]")) {
    const key = el.getAttribute("data-i18n");
    if (key) {
      el.textContent = t(key);
    }
  }

  for (const el of root.querySelectorAll("[data-i18n-placeholder]")) {
    const key = el.getAttribute("data-i18n-placeholder");
    if (key) {
      el.setAttribute("placeholder", t(key));
    }
  }

  for (const el of root.querySelectorAll("[data-i18n-title]")) {
    const key = el.getAttribute("data-i18n-title");
    if (key) {
      el.setAttribute("title", t(key));
    }
  }

  for (const el of root.querySelectorAll("[data-i18n-aria-label]")) {
    const key = el.getAttribute("data-i18n-aria-label");
    if (key) {
      el.setAttribute("aria-label", t(key));
    }
  }
}

export function getLocale() {
  return currentLocale;
}

export function getLocales() {
  return availableLocales;
}

export const i18n = {
  detectBasePath,
  loadLocales,
  load,
  t,
  translateError,
  applyStaticTranslations,
  getLocale,
  getLocales,
};

export default i18n;
