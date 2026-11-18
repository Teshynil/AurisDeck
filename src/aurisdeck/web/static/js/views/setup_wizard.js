/**
 * First-run Setup Wizard view (Language -> Admin Account -> HF_TOKEN -> Summary).
 */

import { api } from "../api.js";
import { buildHfTokenPanel } from "../components/hf_token_panel.js";
import { i18n, t, translateError } from "../i18n.js";

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

/**
 * Mount the Setup Wizard into `container` and invoke `onCompleted(result)` when finished.
 */
export function mountSetupWizard(container, { locales = [], initialLocale = "es", onCompleted }) {
  const state = {
    step: 1,
    locale: initialLocale || "es",
    username: "admin",
    password: "",
    confirmPassword: "",
    showPassword: false,
    showConfirm: false,
    hfToken: "",
    hfVerified: false,
    error: "",
    submitting: false,
  };
  let nextBtnEl = null;

  const availableLocales =
    Array.isArray(locales) && locales.length > 0
      ? locales
      : [
          { code: "es", name: "Español" },
          { code: "en", name: "English" },
        ];

  function validateStep2() {
    const username = state.username.trim();
    if (!username || /\s/.test(username)) {
      state.error = t("setup.errors.invalid_username");
      return false;
    }
    if (state.password.length < 6) {
      state.error = t("setup.errors.password_short");
      return false;
    }
    if (state.password !== state.confirmPassword) {
      state.error = t("setup.errors.password_mismatch");
      return false;
    }
    state.error = "";
    return true;
  }

  async function selectLocale(code) {
    state.locale = code;
    await i18n.load(code).catch(() => null);
    render();
  }

  async function handleFinish() {
    if (state.submitting) {
      return;
    }
    if (!validateStep2()) {
      state.step = 2;
      render();
      return;
    }
    state.submitting = true;
    state.error = "";
    render();
    try {
      const result = await api.setupComplete({
        locale: state.locale,
        username: state.username.trim(),
        password: state.password,
        hf_token: state.hfToken.trim() || null,
      });
      if (typeof onCompleted === "function") {
        await onCompleted(result);
      }
    } catch (err) {
      state.error = translateError(err);
      state.submitting = false;
      render();
    }
  }

  function renderStep1() {
    return el(
      "div",
      null,
      el("h2", { className: "panel-title", textContent: t("setup.lang_title") }),
      el("p", { className: "panel-desc", textContent: t("setup.lang_desc") }),
      el(
        "div",
        { className: "lang-grid" },
        availableLocales.map((loc) =>
          el(
            "button",
            {
              type: "button",
              className: `lang-card ${state.locale === loc.code ? "is-selected" : ""}`,
              onClick: () => void selectLocale(loc.code),
            },
            el("span", { className: "lang-code", textContent: String(loc.code).toUpperCase() }),
            el("div", { className: "lang-name", textContent: loc.name || loc.code })
          )
        )
      )
    );
  }

  function renderStep2() {
    return el(
      "div",
      null,
      el("h2", { className: "panel-title", textContent: t("setup.admin_title") }),
      el("p", { className: "panel-desc", textContent: t("setup.admin_desc") }),
      el(
        "div",
        { className: "form-group" },
        el("label", { className: "label", for: "setup-username", textContent: t("setup.username") }),
        el("input", {
          id: "setup-username",
          className: "input",
          type: "text",
          value: state.username,
          autocomplete: "username",
          onInput: (ev) => {
            state.username = ev.target.value;
          },
        })
      ),
      el(
        "div",
        { className: "form-group" },
        el("label", { className: "label", for: "setup-password", textContent: t("setup.password") }),
        el(
          "div",
          { className: "input-wrap" },
          el("input", {
            id: "setup-password",
            className: "input",
            type: state.showPassword ? "text" : "password",
            value: state.password,
            autocomplete: "new-password",
            onInput: (ev) => {
              state.password = ev.target.value;
            },
          }),
          el("button", {
            type: "button",
            className: "input-action",
            textContent: state.showPassword ? t("common.hide") : t("common.show"),
            onClick: () => {
              state.showPassword = !state.showPassword;
              render();
            },
          })
        )
      ),
      el(
        "div",
        { className: "form-group" },
        el("label", {
          className: "label",
          for: "setup-password-confirm",
          textContent: t("setup.confirm_password"),
        }),
        el(
          "div",
          { className: "input-wrap" },
          el("input", {
            id: "setup-password-confirm",
            className: "input",
            type: state.showConfirm ? "text" : "password",
            value: state.confirmPassword,
            autocomplete: "new-password",
            onInput: (ev) => {
              state.confirmPassword = ev.target.value;
            },
          }),
          el("button", {
            type: "button",
            className: "input-action",
            textContent: state.showConfirm ? t("common.hide") : t("common.show"),
            onClick: () => {
              state.showConfirm = !state.showConfirm;
              render();
            },
          })
        )
      )
    );
  }

  function renderStep3() {
    return el(
      "div",
      null,
      el("h2", { className: "panel-title", textContent: t("setup.hf_title") }),
      el("p", { className: "panel-desc", textContent: t("setup.hf_desc") }),
      buildHfTokenPanel({
        inputId: "setup-hf-token",
        value: state.hfToken,
        optionalLabel: true,
        onInput: (val) => {
          state.hfToken = val;
          state.hfVerified = false;
          if (nextBtnEl && state.step === 3) {
            nextBtnEl.disabled = true;
          }
        },
        onVerify: (token) => api.setupVerifyHfToken(token),
        onVerifyResult: (result) => {
          state.hfVerified = Boolean(result && result.valid && state.hfToken.trim());
          if (nextBtnEl && state.step === 3) {
            nextBtnEl.disabled = !state.hfVerified;
          }
        },
      })
    );
  }

  function renderStep4() {
    const selectedLocaleObj = availableLocales.find((loc) => loc.code === state.locale);
    const hasToken = Boolean(state.hfToken.trim());
    return el(
      "div",
      null,
      el("h2", { className: "panel-title", textContent: t("setup.finish_title") }),
      el("p", { className: "panel-desc", textContent: t("setup.finish_desc") }),
      el(
        "div",
        { className: "summary-box" },
        el(
          "div",
          { className: "summary-row" },
          el("span", { className: "summary-key", textContent: t("setup.step_lang") }),
          el("span", {
            className: "summary-val",
            textContent: selectedLocaleObj?.name || state.locale,
          })
        ),
        el(
          "div",
          { className: "summary-row" },
          el("span", { className: "summary-key", textContent: t("setup.step_admin") }),
          el("span", { className: "summary-val", textContent: state.username.trim() })
        ),
        el(
          "div",
          { className: "summary-row" },
          el("span", { className: "summary-key", textContent: "HF_TOKEN" }),
          el("span", {
            className: `status-pill ${hasToken ? "is-ok" : "is-warn"}`,
            textContent: hasToken ? t("setup.hf_configured") : t("setup.hf_skipped"),
          })
        )
      )
    );
  }

  function render() {
    const steps = [
      { num: 1, label: t("setup.step_lang") },
      { num: 2, label: t("setup.step_admin") },
      { num: 3, label: t("setup.step_hf") },
      { num: 4, label: t("setup.step_finish") },
    ];

    const bodyNode =
      state.step === 1
        ? renderStep1()
        : state.step === 2
          ? renderStep2()
          : state.step === 3
            ? renderStep3()
            : renderStep4();

    const card = el(
      "div",
      { className: "wizard-container" },
      el(
        "div",
        { className: "wizard-card" },
        el(
          "nav",
          { className: "stepper", "aria-label": t("setup.title") },
          steps.map((s) =>
            el(
              "div",
              {
                className: `step-item ${
                  s.num === state.step ? "is-active" : s.num < state.step ? "is-done" : ""
                }`,
              },
              el("span", {
                className: "step-num",
                textContent: s.num < state.step ? "✓" : String(s.num),
              }),
              el("span", { className: "step-label", textContent: s.label })
            )
          )
        ),
        el(
          "div",
          { className: "wizard-body" },
          bodyNode,
          state.error
            ? el("div", { className: "form-error", role: "alert", textContent: state.error })
            : null
        ),
        el(
          "footer",
          { className: "wizard-footer" },
          state.step > 1
            ? el("button", {
                type: "button",
                className: "btn btn-secondary",
                textContent: t("setup.prev"),
                disabled: state.submitting,
                onClick: () => {
                  state.error = "";
                  state.step -= 1;
                  render();
                },
              })
            : el("span"),
          el(
            "div",
            { className: "footer-right" },
            state.step === 3
              ? el("button", {
                  type: "button",
                  className: "btn btn-ghost",
                  textContent: t("setup.skip"),
                  onClick: () => {
                    state.hfToken = "";
                    state.hfVerified = false;
                    state.error = "";
                    state.step = 4;
                    render();
                  },
                })
              : null,
            state.step < 4
              ? (nextBtnEl = el("button", {
                  type: "button",
                  className: "btn btn-primary",
                  textContent: t("setup.next"),
                  disabled: state.step === 3 && !state.hfVerified,
                  onClick: () => {
                    if (state.step === 2 && !validateStep2()) {
                      render();
                      return;
                    }
                    if (state.step === 3 && !state.hfVerified) {
                      return;
                    }
                    state.error = "";
                    state.step += 1;
                    render();
                  },
                }))
              : ((nextBtnEl = null),
                el("button", {
                  type: "button",
                  className: "btn btn-primary",
                  textContent: state.submitting ? t("common.loading") : t("setup.finish"),
                  disabled: state.submitting,
                  onClick: () => void handleFinish(),
                }))
          )
        )
      )
    );

    container.replaceChildren(card);
  }

  render();
}
