/**
 * Settings view: strictly separates Account Settings (/api/v1/me)
 * from System & Hardware Settings (/api/v1/admin/settings, admin-only).
 */

import { api } from "../api.js";
import { buildHfTokenPanel } from "../components/hf_token_panel.js";
import { i18n, t, translateError } from "../i18n.js";
import { store } from "../store.js";

const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3];
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

function extractAccountForm(state) {
  const prefs = state.user?.preferences || state.preferences || {};
  const player = prefs.player && typeof prefs.player === "object" ? prefs.player : {};
  const profiles = Array.isArray(state.profiles) ? state.profiles : [];
  const defaultProfileId =
    typeof prefs.default_profile_id === "string" && prefs.default_profile_id
      ? prefs.default_profile_id
      : profiles[0]?.id || "";
  const rate = Number(player.playback_rate);
  return {
    locale: typeof prefs.locale === "string" && prefs.locale ? prefs.locale : state.locale || "es",
    default_profile_id: defaultProfileId,
    playback_rate: Number.isFinite(rate) && rate >= 0.25 && rate <= 4.0 ? rate : 1.0,
    auto_scroll: player.auto_scroll !== undefined ? Boolean(player.auto_scroll) : true,
  };
}

function localizedProfileName(profile) {
  return profile?.display_name || profile?.id || "";
}

/**
 * Mount the Settings view into `container` (#view-settings).
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
  const local = {
    account: extractAccountForm(initial),
    accountDirty: false,
    savingAccount: false,
    accountError: "",

    system: {
      model_idle_unload_seconds: 300,
      cpu_threads: 0,
      hf_token_configured: false,
      hf_token_masked: null,
      hfTokenInput: "",
      clearHfToken: false,
    },
    systemLoaded: false,
    loadingSystem: false,
    savingSystem: false,
    unloadingModels: false,
    systemError: "",
  };

  let devicesListEl = null;

  function isAuthenticated(state) {
    return !(state.authStatus?.auth_mode === "password" && !state.authStatus?.authenticated);
  }

  async function ensureLocalesLoaded() {
    const state = store.getState();
    if (Array.isArray(state.locales) && state.locales.length > 0) {
      return;
    }
    try {
      await i18n.loadLocales();
    } catch {
      // Locale list fallback handled gracefully in selector.
    }
  }

  async function ensureAdminSettingsLoaded(force = false) {
    const state = store.getState();
    if (state.user?.role !== "admin" || !isAuthenticated(state)) {
      return;
    }
    if ((local.systemLoaded && !force) || local.loadingSystem) {
      return;
    }
    local.loadingSystem = true;
    local.systemError = "";
    try {
      const [settingsPayload, devicesPayload] = await Promise.all([
        api.getAdminSettings(),
        Array.isArray(state.devices) && state.devices.length > 0
          ? Promise.resolve(null)
          : api.getDevices(),
      ]);
      if (settingsPayload && typeof settingsPayload === "object") {
        local.system = {
          model_idle_unload_seconds: Number(settingsPayload.model_idle_unload_seconds ?? 300),
          cpu_threads: Number(settingsPayload.cpu_threads ?? 0),
          hf_token_configured: Boolean(settingsPayload.hf_token_configured),
          hf_token_masked: settingsPayload.hf_token_masked ?? null,
          hfTokenInput: "",
          clearHfToken: false,
        };
        local.systemLoaded = true;
      }
      if (devicesPayload && Array.isArray(devicesPayload.devices)) {
        store.setState({ devices: devicesPayload.devices });
      }
    } catch (err) {
      local.systemError = translateError(err);
    } finally {
      local.loadingSystem = false;
      render();
    }
  }

  function renderDevicesList(targetEl) {
    if (!targetEl) {
      return;
    }
    const devices = Array.isArray(store.getState().devices) ? store.getState().devices : [];
    if (devices.length === 0) {
      targetEl.replaceChildren(
        el("span", { className: "help-text", textContent: t("common.none") })
      );
      return;
    }

    targetEl.replaceChildren(
      ...devices.map((dev) => {
        const isCuda = dev.kind === "cuda";
        const hasVram =
          isCuda &&
          dev.vram_total_mb !== null &&
          dev.vram_total_mb !== undefined &&
          dev.vram_free_mb !== null &&
          dev.vram_free_mb !== undefined;
        const vramText = hasVram
          ? t("settings.system.vram_usage", {
              free: dev.vram_free_mb,
              total: dev.vram_total_mb,
            })
          : "";
        return el(
          "div",
          { className: "file-item" },
          el(
            "div",
            { className: "row" },
            el("span", {
              className: `badge ${isCuda ? "badge-gpu" : "badge-cpu"}`,
              textContent: dev.id || (isCuda ? t("common.cuda") : t("common.cpu")),
            }),
            el("strong", { textContent: dev.name || dev.id || "" })
          ),
          el(
            "div",
            { className: "row" },
            vramText ? el("span", { className: "help-text mono", textContent: vramText }) : null,
            dev.loaded_model
              ? el("span", { className: "badge badge-info mono", textContent: dev.loaded_model })
              : null,
            dev.state
              ? el("span", {
                  className: `badge ${dev.state === "busy" ? "badge-warning" : "badge-success"}`,
                  textContent: dev.state,
                })
              : null
          )
        );
      })
    );
  }

  async function handleSaveAccount(event) {
    event.preventDefault();
    if (local.savingAccount) {
      return;
    }
    local.savingAccount = true;
    local.accountError = "";
    render();

    const newLocale = local.account.locale || "es";
    const patch = {
      locale: newLocale,
      player: {
        playback_rate: Number(local.account.playback_rate) || 1.0,
        auto_scroll: Boolean(local.account.auto_scroll),
      },
    };
    if (local.account.default_profile_id) {
      patch.default_profile_id = local.account.default_profile_id;
    }

    try {
      const updatedUser = await api.patchMe(patch);
      await i18n.load(newLocale);
      store.setState({
        user: updatedUser,
        preferences: updatedUser?.preferences || {},
        locale: newLocale,
      });
      i18n.applyStaticTranslations();
      local.account = extractAccountForm(store.getState());
      local.accountDirty = false;
      store.addToast({ type: "success", message: t("settings.account.saved") });
    } catch (err) {
      local.accountError = translateError(err);
      store.addToast({ type: "danger", message: local.accountError });
    } finally {
      local.savingAccount = false;
      render();
    }
  }

  async function handleSaveSystem(event) {
    event.preventDefault();
    if (local.savingSystem) {
      return;
    }
    local.savingSystem = true;
    local.systemError = "";
    render();

    const payload = {
      model_idle_unload_seconds: Math.max(
        0,
        Number.parseInt(local.system.model_idle_unload_seconds, 10) || 0
      ),
      cpu_threads: Math.max(0, Number.parseInt(local.system.cpu_threads, 10) || 0),
    };
    const trimmedToken = local.system.hfTokenInput.trim();
    if (trimmedToken) {
      payload.hf_token = trimmedToken;
    } else if (local.system.clearHfToken) {
      payload.hf_token = null;
    }

    try {
      const updated = await api.patchAdminSettings(payload);
      if (updated && typeof updated === "object") {
        local.system = {
          model_idle_unload_seconds: Number(updated.model_idle_unload_seconds ?? payload.model_idle_unload_seconds),
          cpu_threads: Number(updated.cpu_threads ?? payload.cpu_threads),
          hf_token_configured: Boolean(updated.hf_token_configured),
          hf_token_masked: updated.hf_token_masked ?? null,
          hfTokenInput: "",
          clearHfToken: false,
        };
      }
      store.addToast({ type: "success", message: t("settings.system.saved") });
    } catch (err) {
      local.systemError = translateError(err);
      store.addToast({ type: "danger", message: local.systemError });
    } finally {
      local.savingSystem = false;
      render();
    }
  }

  async function handleUnloadModels() {
    if (local.unloadingModels) {
      return;
    }
    local.unloadingModels = true;
    local.systemError = "";
    render();
    try {
      await api.unloadModels();
      try {
        const devicesRes = await api.getDevices();
        if (Array.isArray(devicesRes?.devices)) {
          store.setState({ devices: devicesRes.devices });
        }
      } catch {
        // Non-fatal if device refresh lags behind unload broadcast.
      }
      store.addToast({ type: "info", message: t("settings.system.models_unloaded") });
    } catch (err) {
      local.systemError = translateError(err);
      store.addToast({ type: "danger", message: local.systemError });
    } finally {
      local.unloadingModels = false;
      render();
    }
  }

  function renderAccountCard(state) {
    const locales = Array.isArray(state.locales) ? state.locales : [];
    const profiles = Array.isArray(state.profiles) ? state.profiles : [];
    const currentLocale = state.locale || local.account.locale || "es";

    const localeOptions =
      locales.length > 0
        ? locales
        : [{ code: currentLocale, name: currentLocale }];

    const rateOptions = PLAYBACK_RATES.includes(local.account.playback_rate)
      ? PLAYBACK_RATES
      : [...PLAYBACK_RATES, local.account.playback_rate].sort((a, b) => a - b);

    return el(
      "form",
      { id: "account-settings-card", className: "card", onSubmit: handleSaveAccount },
      el(
        "div",
        { className: "card-header" },
        el(
          "div",
          null,
          el("h3", { className: "card-title", textContent: t("settings.account.title") }),
          el("p", { className: "card-subtitle", textContent: t("settings.account.description") })
        )
      ),
      el(
        "div",
        { className: "card-body" },
        local.accountError
          ? el("div", { className: "alert alert-danger", role: "alert", textContent: local.accountError })
          : null,
        el(
          "div",
          { className: "form-grid" },
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "settings-account-locale",
              textContent: t("settings.account.locale"),
            }),
            el(
              "select",
              {
                id: "settings-account-locale",
                className: "select",
                onChange: (ev) => {
                  local.account.locale = ev.target.value;
                  local.accountDirty = true;
                },
              },
              localeOptions.map((loc) =>
                el("option", {
                  value: loc.code,
                  textContent: `${loc.name || loc.code} (${loc.code})`,
                  selected: loc.code === local.account.locale,
                })
              )
            )
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "settings-account-default-profile",
              textContent: t("settings.account.default_profile"),
            }),
            el(
              "select",
              {
                id: "settings-account-default-profile",
                className: "select",
                onChange: (ev) => {
                  local.account.default_profile_id = ev.target.value;
                  local.accountDirty = true;
                },
              },
              profiles.length === 0
                ? el("option", { value: "", textContent: t("common.none"), selected: true })
                : profiles.map((prof) =>
                    el("option", {
                      value: prof.id,
                      textContent: localizedProfileName(prof, currentLocale),
                      selected: prof.id === local.account.default_profile_id,
                    })
                  )
            )
          )
        ),
        el(
          "div",
          { className: "stack" },
          el("h4", { className: "label", style: "margin: 0;", textContent: t("settings.account.player_title") }),
          el(
            "div",
            { className: "form-grid" },
            el(
              "div",
              { className: "form-group" },
              el("label", {
                className: "label",
                for: "settings-account-playback-rate",
                textContent: t("settings.account.playback_rate"),
              }),
              el(
                "select",
                {
                  id: "settings-account-playback-rate",
                  className: "select",
                  onChange: (ev) => {
                    local.account.playback_rate = Number(ev.target.value) || 1.0;
                    local.accountDirty = true;
                  },
                },
                rateOptions.map((rate) =>
                  el("option", {
                    value: String(rate),
                    textContent: `${rate}x`,
                    selected: Number(local.account.playback_rate) === rate,
                  })
                )
              )
            ),
            el(
              "div",
              { className: "form-group", style: "justify-content: flex-end;" },
              el(
                "label",
                { className: "checkbox-label" },
                el("input", {
                  id: "settings-account-auto-scroll",
                  type: "checkbox",
                  checked: local.account.auto_scroll,
                  onChange: (ev) => {
                    local.account.auto_scroll = ev.target.checked;
                    local.accountDirty = true;
                  },
                }),
                el("span", { textContent: t("settings.account.auto_scroll") })
              )
            )
          )
        )
      ),
      el(
        "div",
        { className: "card-footer" },
        el("button", {
          id: "settings-account-save",
          type: "submit",
          className: "btn btn-primary",
          disabled: local.savingAccount,
          textContent: local.savingAccount ? t("common.loading") : t("settings.account.save"),
        })
      )
    );
  }

  function renderSystemCard() {
    devicesListEl = el("div", { id: "settings-devices-list", className: "file-list" });
    renderDevicesList(devicesListEl);

    return el(
      "form",
      { id: "system-settings-card", className: "card", onSubmit: handleSaveSystem },
      el(
        "div",
        { className: "card-header" },
        el(
          "div",
          null,
          el("h3", { className: "card-title", textContent: t("settings.system.title") }),
          el("p", { className: "card-subtitle", textContent: t("settings.system.description") })
        )
      ),
      el(
        "div",
        { className: "card-body" },
        local.systemError
          ? el("div", { className: "alert alert-danger", role: "alert", textContent: local.systemError })
          : null,
        el(
          "div",
          { className: "form-group" },
          el("span", { className: "label", textContent: t("settings.system.devices_title") }),
          devicesListEl
        ),
        el(
          "div",
          { className: "form-grid" },
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "settings-system-idle-unload",
              textContent: t("settings.system.model_idle_unload_seconds"),
            }),
            el("input", {
              id: "settings-system-idle-unload",
              className: "input",
              type: "number",
              min: "0",
              step: "1",
              value: local.system.model_idle_unload_seconds,
              onInput: (ev) => {
                const parsed = Number.parseInt(ev.target.value, 10);
                local.system.model_idle_unload_seconds =
                  Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
              },
            })
          ),
          el(
            "div",
            { className: "form-group" },
            el("label", {
              className: "label",
              for: "settings-system-cpu-threads",
              textContent: t("settings.system.cpu_threads"),
            }),
            el("input", {
              id: "settings-system-cpu-threads",
              className: "input",
              type: "number",
              min: "0",
              step: "1",
              value: local.system.cpu_threads,
              onInput: (ev) => {
                const parsed = Number.parseInt(ev.target.value, 10);
                local.system.cpu_threads = Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
              },
            })
          )
        ),
        buildHfTokenPanel({
          inputId: "settings-system-hf-token",
          value: local.system.hfTokenInput,
          configured: local.system.hf_token_configured,
          maskedToken: local.system.hf_token_masked,
          onInput: (val) => {
            local.system.hfTokenInput = val;
            if (val.trim()) {
              local.system.clearHfToken = false;
            }
          },
          onVerify: (token) => api.adminVerifyHfToken(token),
          onClear: async () => {
            local.system.hfTokenInput = "";
            local.system.clearHfToken = true;
            const updated = await api.patchAdminSettings({ hf_token: null });
            local.system.hf_token_configured = Boolean(updated?.hf_token_configured);
            local.system.hf_token_masked = updated?.hf_token_masked ?? null;
            local.system.clearHfToken = false;
            store.addToast({ type: "info", message: t("hf_token.cleared") });
            render();
          },
        })
      ),
      el(
        "div",
        { className: "card-footer" },
        el("button", {
          id: "settings-unload-models-btn",
          type: "button",
          className: "btn btn-secondary",
          disabled: local.unloadingModels,
          textContent: local.unloadingModels ? t("common.loading") : t("settings.system.unload_models"),
          onClick: handleUnloadModels,
        }),
        el("button", {
          id: "settings-system-save",
          type: "submit",
          className: "btn btn-primary",
          disabled: local.savingSystem,
          textContent: local.savingSystem ? t("common.loading") : t("settings.system.save"),
        })
      )
    );
  }

  function render() {
    devicesListEl = null;
    const state = store.getState();
    const isAdmin = state.user?.role === "admin";

    const header = el(
      "div",
      { className: "section-header" },
      el("h2", { className: "section-title", textContent: t("settings.title") })
    );

    const nodes = [header, renderAccountCard(state)];
    if (isAdmin) {
      nodes.push(renderSystemCard());
    }

    container.replaceChildren(...nodes);
  }

  const unsubscribe = store.subscribe((state, prevState) => {
    if (state.user?.role === "admin" && !local.systemLoaded && isAuthenticated(state)) {
      if (state.activeTab === "settings" || prevState.user !== state.user || prevState.authStatus !== state.authStatus) {
        void ensureAdminSettingsLoaded();
      }
    }

    if (prevState.user !== state.user || prevState.preferences !== state.preferences) {
      if (!local.accountDirty || prevState.user?.id !== state.user?.id) {
        local.account = extractAccountForm(state);
        local.accountDirty = false;
      }
      render();
      return;
    }

    if (
      prevState.locale !== state.locale ||
      prevState.locales !== state.locales ||
      prevState.profiles !== state.profiles
    ) {
      if (!local.account.default_profile_id && Array.isArray(state.profiles) && state.profiles.length > 0) {
        local.account.default_profile_id = state.profiles[0].id;
      }
      render();
      return;
    }

    if (prevState.devices !== state.devices && devicesListEl) {
      renderDevicesList(devicesListEl);
    }
  });

  render();
  void ensureLocalesLoaded();
  if (store.getState().user?.role === "admin") {
    void ensureAdminSettingsLoaded();
  }

  const cleanup = () => {
    unsubscribe();
    containerCleanups.delete(container);
  };
  containerCleanups.set(container, cleanup);
  return cleanup;
}

export default { mount };
