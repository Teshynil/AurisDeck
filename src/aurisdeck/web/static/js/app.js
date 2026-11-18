/**
 * AurisDeck Web UI entry point and application shell orchestrator.
 */

import { api, onAuthRequired } from "./api.js";
import { i18n } from "./i18n.js";
import { connectSSE, disconnectSSE } from "./sse.js";
import { store } from "./store.js";

const VALID_TABS = new Set(["transcribe", "library", "viewer", "profiles", "settings"]);
const THEME_STORAGE_KEY = "aurisdeck:theme";

const VIEW_LOADERS = {
  transcribe: () => import("./views/transcribe.js").catch(() => null),
  library: () => import("./views/library.js").catch(() => null),
  viewer: () => import("./views/viewer.js").catch(() => null),
  profiles: () => import("./views/profiles.js").catch(() => null),
  settings: () => import("./views/settings.js").catch(() => null),
};

const loadedViewModules = new Map();
let loginWaiters = [];
let isBooted = false;
let sessionLoading = false;

function getSafeStorage(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function setSafeStorage(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore storage errors in restricted contexts.
  }
}

function resolveEffectiveTheme(theme) {
  if (theme === "light" || theme === "dark") {
    return theme;
  }
  if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return "light";
}

function applyTheme(theme) {
  const normalized = theme === "light" || theme === "dark" ? theme : "system";
  if (typeof document !== "undefined" && document.documentElement) {
    if (normalized === "system") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", normalized);
    }
    const iconEl = document.getElementById("theme-toggle-icon");
    if (iconEl) {
      const effective = resolveEffectiveTheme(normalized);
      iconEl.textContent = effective === "dark" ? "☀" : "☾";
    }
  }
  setSafeStorage(THEME_STORAGE_KEY, normalized);
  store.setState({ theme: normalized });
}

function toggleTheme() {
  const currentEffective = resolveEffectiveTheme(store.getState().theme);
  applyTheme(currentEffective === "dark" ? "light" : "dark");
}

function showLoginModal() {
  const modal = document.getElementById("login-modal");
  const errorEl = document.getElementById("login-error");
  const passwordEl = document.getElementById("login-password");
  if (!modal) {
    return;
  }
  if (errorEl) {
    errorEl.textContent = "";
    errorEl.classList.add("hidden");
  }
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
  if (passwordEl) {
    passwordEl.value = "";
    passwordEl.focus();
  }
}

function hideLoginModal() {
  const modal = document.getElementById("login-modal");
  const errorEl = document.getElementById("login-error");
  if (!modal) {
    return;
  }
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
  if (errorEl) {
    errorEl.textContent = "";
    errorEl.classList.add("hidden");
  }
}

function waitForLogin() {
  return new Promise((resolve) => {
    loginWaiters.push(resolve);
  });
}

function resolveLoginWaiters() {
  const waiters = loginWaiters;
  loginWaiters = [];
  for (const resolve of waiters) {
    resolve();
  }
}

function updateLogoutButton(authStatus) {
  const logoutBtn = document.getElementById("logout-btn");
  if (!logoutBtn) {
    return;
  }
  const shouldShow = Boolean(
    authStatus && authStatus.auth_mode === "password" && authStatus.authenticated
  );
  logoutBtn.classList.toggle("hidden", !shouldShow);
}

function updateDevicesBadge(devices) {
  const badge = document.getElementById("header-devices-badge");
  if (!badge) {
    return;
  }

  const list = Array.isArray(devices) ? devices : [];
  const cudaDevices = list.filter((dev) => dev && dev.kind === "cuda");
  const cpuDevices = list.filter((dev) => dev && dev.kind === "cpu");

  badge.textContent = "";
  badge.classList.remove("badge-gpu", "badge-cpu");

  if (cudaDevices.length > 0) {
    badge.classList.add("badge-gpu");
    const label = `${cudaDevices.length} × ${i18n.t("common.cuda")}`;
    badge.textContent = label;
    badge.title = cudaDevices
      .map((dev) => `${dev.id}: ${dev.name || dev.id}`)
      .join(" | ");
    return;
  }

  badge.classList.add("badge-cpu");
  if (cpuDevices.length > 0) {
    badge.textContent = i18n.t("common.cpu");
    badge.title = cpuDevices.map((dev) => dev.name || dev.id).join(" | ");
  } else {
    badge.textContent = i18n.t("common.auto");
    badge.title = i18n.t("common.auto");
  }
}

function renderToasts(toasts) {
  const container = document.getElementById("toast-container");
  if (!container) {
    return;
  }

  container.textContent = "";
  const list = Array.isArray(toasts) ? toasts : [];
  for (const toast of list) {
    const item = document.createElement("div");
    const type = ["info", "success", "warning", "danger"].includes(toast.type)
      ? toast.type
      : "info";
    item.className = `toast toast-${type}`;
    item.setAttribute("role", "status");

    const body = document.createElement("div");
    body.className = "stack";
    body.style.gap = "0.2rem";

    if (toast.title) {
      const titleEl = document.createElement("strong");
      titleEl.textContent = toast.title;
      body.appendChild(titleEl);
    }

    if (toast.message) {
      const msgEl = document.createElement("div");
      msgEl.textContent = toast.message;
      body.appendChild(msgEl);
    }

    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "toast-close";
    closeBtn.textContent = "×";
    closeBtn.setAttribute("aria-label", i18n.t("common.close"));
    closeBtn.addEventListener("click", () => {
      store.removeToast(toast.id);
    });

    item.appendChild(body);
    item.appendChild(closeBtn);
    container.appendChild(item);
  }
}

function getTabFromHash() {
  if (typeof window === "undefined") {
    return null;
  }
  const raw = (window.location.hash || "").replace(/^#/, "").trim().toLowerCase();
  return VALID_TABS.has(raw) ? raw : null;
}

async function loadViewModule(tab) {
  if (loadedViewModules.has(tab)) {
    return loadedViewModules.get(tab);
  }
  const loader = VIEW_LOADERS[tab];
  if (!loader) {
    return null;
  }
  const mod = await loader();
  if (mod) {
    loadedViewModules.set(tab, mod);
  }
  return mod;
}

async function renderTab(tab) {
  const resolvedTab = VALID_TABS.has(tab) ? tab : "transcribe";

  for (const btn of document.querySelectorAll("#main-nav .nav-tab")) {
    const btnTab = btn.getAttribute("data-tab");
    const active = btnTab === resolvedTab;
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  }

  for (const section of document.querySelectorAll("#app-main .view-section")) {
    const viewName = section.getAttribute("data-view");
    section.classList.toggle("hidden", viewName !== resolvedTab);
  }

  if (typeof window !== "undefined" && window.location.hash !== `#${resolvedTab}`) {
    window.history.replaceState(null, "", `#${resolvedTab}`);
  }

  const sectionEl = document.getElementById(`view-${resolvedTab}`);
  if (!sectionEl) {
    return;
  }

  const mod = await loadViewModule(resolvedTab);
  if (mod && typeof mod.mount === "function") {
    await mod.mount(sectionEl);
  }
}

function resolveBootLocaleFallback() {
  if (typeof navigator !== "undefined" && typeof navigator.language === "string") {
    return navigator.language.toLowerCase().startsWith("en") ? "en" : "es";
  }
  return "es";
}

function rethrowAuthError(err) {
  if (err && err.status === 401) {
    throw err;
  }
}

async function loadAuthenticatedSession({ useHashOrPref = false } = {}) {
  if (sessionLoading) {
    return;
  }
  sessionLoading = true;
  try {
    const me = await api.getMe();
    const preferences = me?.preferences && typeof me.preferences === "object" ? me.preferences : {};
    const targetLocale = preferences.locale || "es";

    await i18n.load(targetLocale);
    i18n.applyStaticTranslations(document);

    store.setState({
      user: me,
      preferences,
    });

    const [profilesRes, glossariesRes, replacementsRes, devicesRes, jobsRes] = await Promise.all([
      api.getProfiles().catch((err) => {
        rethrowAuthError(err);
        return [];
      }),
      api.getGlossaries().catch((err) => {
        rethrowAuthError(err);
        return [];
      }),
      api.getReplacements().catch((err) => {
        rethrowAuthError(err);
        return [];
      }),
      api.getDevices().catch((err) => {
        rethrowAuthError(err);
        return { devices: [] };
      }),
      api.listJobs().catch((err) => {
        rethrowAuthError(err);
        return { jobs: [] };
      }),
    ]);

    const profiles = Array.isArray(profilesRes) ? profilesRes : profilesRes?.profiles || [];
    const glossaries = Array.isArray(glossariesRes) ? glossariesRes : glossariesRes?.glossaries || [];
    const replacements = Array.isArray(replacementsRes)
      ? replacementsRes
      : replacementsRes?.replacements || [];
    const devices = Array.isArray(devicesRes?.devices) ? devicesRes.devices : [];
    const jobs = Array.isArray(jobsRes?.jobs) ? jobsRes.jobs : [];

    store.setState({
      profiles,
      glossaries,
      replacements,
      devices,
      jobs,
    });

    updateDevicesBadge(devices);

    const hashTab = getTabFromHash();
    const nextTab = useHashOrPref
      ? hashTab || "transcribe"
      : store.getState().activeTab || "transcribe";

    store.setState({ activeTab: nextTab });
    await renderTab(nextTab);
    connectSSE();
  } finally {
    sessionLoading = false;
  }
}

function bindShellEvents() {
  const themeBtn = document.getElementById("theme-toggle-btn");
  themeBtn?.addEventListener("click", toggleTheme);

  const nav = document.getElementById("main-nav");
  nav?.addEventListener("click", (event) => {
    const btn = event.target?.closest?.(".nav-tab[data-tab]");
    if (!btn) {
      return;
    }
    const tab = btn.getAttribute("data-tab");
    if (tab && VALID_TABS.has(tab)) {
      store.setState({ activeTab: tab });
    }
  });

  const brandLink = document.getElementById("brand-link");
  brandLink?.addEventListener("click", (event) => {
    event.preventDefault();
    store.setState({ activeTab: "transcribe" });
  });

  window.addEventListener("hashchange", () => {
    const hashTab = getTabFromHash();
    if (hashTab && hashTab !== store.getState().activeTab) {
      store.setState({ activeTab: hashTab });
    }
  });

  const loginForm = document.getElementById("login-form");
  loginForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const usernameEl = document.getElementById("login-username");
    const passwordEl = document.getElementById("login-password");
    const submitEl = document.getElementById("login-submit");
    const errorEl = document.getElementById("login-error");
    const username = usernameEl ? usernameEl.value.trim() : "";
    const password = passwordEl ? passwordEl.value : "";

    if (submitEl) {
      submitEl.disabled = true;
    }
    if (errorEl) {
      errorEl.textContent = "";
      errorEl.classList.add("hidden");
    }

    try {
      await api.login(password, username || null);
      hideLoginModal();
      const nextStatus = { auth_mode: "password", authenticated: true };
      store.setState({ authStatus: nextStatus });
      updateLogoutButton(nextStatus);
      resolveLoginWaiters();
      if (isBooted) {
        await loadAuthenticatedSession({ useHashOrPref: false });
      }
    } catch (err) {
      if (errorEl) {
        errorEl.textContent = i18n.translateError(err);
        errorEl.classList.remove("hidden");
      }
    } finally {
      if (submitEl) {
        submitEl.disabled = false;
      }
    }
  });

  const logoutBtn = document.getElementById("logout-btn");
  logoutBtn?.addEventListener("click", async () => {
    try {
      await api.logout();
    } catch {
      // Proceed to clear client session state even if logout request fails.
    }
    disconnectSSE();
    const nextStatus = {
      auth_mode: store.getState().authStatus?.auth_mode || "password",
      authenticated: false,
    };
    store.setState({
      authStatus: nextStatus,
      user: null,
      preferences: {},
      profiles: [],
      glossaries: [],
      replacements: [],
      devices: [],
      jobs: [],
      selectedJobId: null,
    });
    updateLogoutButton(nextStatus);
    updateDevicesBadge([]);
    showLoginModal();
  });

  onAuthRequired(() => {
    disconnectSSE();
    const currentStatus = store.getState().authStatus || {};
    const nextStatus = {
      auth_mode: currentStatus.auth_mode === "none" ? "password" : currentStatus.auth_mode || "password",
      authenticated: false,
    };
    store.setState({ authStatus: nextStatus });
    updateLogoutButton(nextStatus);
    showLoginModal();
  });

  store.subscribe((state, prevState) => {
    if (state.toasts !== prevState.toasts) {
      renderToasts(state.toasts);
    }
    if (state.devices !== prevState.devices || state.locale !== prevState.locale) {
      updateDevicesBadge(state.devices);
    }
    if (state.authStatus !== prevState.authStatus) {
      updateLogoutButton(state.authStatus);
    }
    if (state.activeTab !== prevState.activeTab) {
      void renderTab(state.activeTab);
    } else if (state.locale !== prevState.locale && isBooted) {
      i18n.applyStaticTranslations(document);
      void renderTab(state.activeTab);
    }
  });
}

async function boot() {
  const savedTheme = getSafeStorage(THEME_STORAGE_KEY) || "system";
  applyTheme(savedTheme);
  bindShellEvents();

  try {
    const locales = await i18n.loadLocales().catch(() => []);
    const status = await api.authStatus();
    store.setState({ authStatus: status });
    updateLogoutButton(status);

    if (status?.setup_required) {
      const bootLocale = resolveBootLocaleFallback();
      await i18n.load(bootLocale).catch(() => null);
      i18n.applyStaticTranslations(document);

      const navEl = document.getElementById("main-nav");
      const badgeEl = document.getElementById("header-devices-badge");
      const setupSection = document.getElementById("view-setup");
      const transcribeSection = document.getElementById("view-transcribe");

      navEl?.classList.add("hidden");
      badgeEl?.classList.add("hidden");
      transcribeSection?.classList.add("hidden");
      setupSection?.classList.remove("hidden");

      const { mountSetupWizard } = await import("./views/setup_wizard.js");
      await new Promise((resolve) => {
        mountSetupWizard(setupSection, {
          locales,
          initialLocale: bootLocale,
          onCompleted: async () => {
            setupSection?.classList.add("hidden");
            navEl?.classList.remove("hidden");
            badgeEl?.classList.remove("hidden");
            const refreshed = await api.authStatus().catch(() => ({
              auth_mode: "password",
              authenticated: true,
              setup_required: false,
            }));
            store.setState({ authStatus: refreshed });
            updateLogoutButton(refreshed);
            resolve();
          },
        });
      });
    } else if (status?.auth_mode === "password" && !status?.authenticated) {
      await i18n.load(resolveBootLocaleFallback()).catch(() => null);
      i18n.applyStaticTranslations(document);
      showLoginModal();
      await waitForLogin();
    }

    await loadAuthenticatedSession({ useHashOrPref: true });
    isBooted = true;
  } catch (err) {
    isBooted = true;
    store.addToast({
      type: "danger",
      title: i18n.t("common.error"),
      message: i18n.translateError(err),
    });
  }
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      void boot();
    });
  } else {
    void boot();
  }
}
