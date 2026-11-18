/**
 * Reusable HF_TOKEN input, verification status, and collapsible mini-tutorial component.
 * Shared between the initial Setup Wizard and Admin System Settings.
 */

import { t } from "../i18n.js";

function createEl(tag, attrs = null, ...children) {
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

const HF_HUB_ORIGIN = ["https", "huggingface.co"].join("://");

function buildTutorialLink(path, label, isCode = false) {
  const content = isCode ? createEl("code", { textContent: label }) : label;
  return createEl(
    "a",
    {
      href: `${HF_HUB_ORIGIN}${path}`,
      target: "_blank",
      rel: "noopener noreferrer",
    },
    content
  );
}

/**
 * Build the collapsible HF_TOKEN mini-tutorial <details> element.
 */
export function buildHfTutorialDetails() {
  return createEl(
    "details",
    { className: "tutorial-details" },
    createEl("summary", {
      className: "tutorial-summary",
      textContent: t("hf_token.tutorial_title"),
    }),
    createEl(
      "div",
      { className: "tutorial-body" },
      createEl(
        "ol",
        { className: "tutorial-steps" },
        createEl(
          "li",
          null,
          t("hf_token.tutorial_step_1_prefix"),
          buildTutorialLink("/join", "Hugging Face"),
          t("hf_token.tutorial_step_1_suffix")
        ),
        createEl(
          "li",
          null,
          t("hf_token.tutorial_step_2_prefix"),
          buildTutorialLink(
            "/pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization-3.1",
            true
          ),
          "."
        ),
        createEl(
          "li",
          null,
          t("hf_token.tutorial_step_3_prefix"),
          buildTutorialLink(
            "/pyannote/segmentation-3.0",
            "pyannote/segmentation-3.0",
            true
          ),
          "."
        ),
        createEl(
          "li",
          null,
          t("hf_token.tutorial_step_4_prefix"),
          createEl("strong", { textContent: "Read-Only" }),
          t("hf_token.tutorial_step_4_mid"),
          buildTutorialLink(
            "/settings/tokens",
            "Settings → Access Tokens"
          ),
          "."
        ),
        createEl(
          "li",
          null,
          t("hf_token.tutorial_step_5_prefix"),
          createEl("code", { textContent: "hf_" }),
          t("hf_token.tutorial_step_5_mid"),
          createEl("strong", { textContent: t("hf_token.verify_btn") }),
          "."
        )
      )
    )
  );
}

function formatVerifyStatus(result) {
  if (!result) {
    return { cls: "status-pill is-neutral", text: t("hf_token.status_unverified") };
  }
  if (result.valid || result.status === "ok") {
    return { cls: "status-pill is-ok", text: t("hf_token.status_valid") };
  }
  if (result.status === "missing_token") {
    return { cls: "status-pill is-warn", text: t("hf_token.status_missing") };
  }
  if (result.status === "invalid_token") {
    return { cls: "status-pill is-danger", text: t("hf_token.status_invalid") };
  }
  if (result.status === "gated_terms_required") {
    const models = Array.isArray(result.missing_models) ? result.missing_models.join(", ") : "";
    return {
      cls: "status-pill is-warn",
      text: models
        ? `${t("hf_token.status_gated")} (${models})`
        : t("hf_token.status_gated"),
    };
  }
  return { cls: "status-pill is-warn", text: t("hf_token.status_network_error") };
}

/**
 * Render an interactive HF_TOKEN panel with optional clear button and verification.
 */
export function buildHfTokenPanel({
  inputId = "hf-token-input",
  value = "",
  configured = false,
  maskedToken = null,
  optionalLabel = false,
  onInput = null,
  onVerify = null,
  onClear = null,
}) {
  let showToken = false;
  let verifying = false;
  let verifyResult = null;

  const inputEl = createEl("input", {
    id: inputId,
    className: "input input-mono",
    type: "password",
    value,
    placeholder: configured && maskedToken ? maskedToken : "hf_...",
    autocomplete: "off",
    onInput: (event) => {
      verifyResult = null;
      refreshStatusPill();
      if (typeof onInput === "function") {
        onInput(event.target.value);
      }
    },
  });

  const toggleBtn = createEl("button", {
    type: "button",
    className: "input-action",
    textContent: t("common.show"),
    onClick: () => {
      showToken = !showToken;
      inputEl.type = showToken ? "text" : "password";
      toggleBtn.textContent = showToken ? t("common.hide") : t("common.show");
    },
  });

  const statusPill = createEl("span", {
    className: "status-pill is-neutral",
    textContent: configured ? t("hf_token.status_configured") : t("hf_token.status_unverified"),
  });

  function refreshStatusPill() {
    if (!verifyResult) {
      statusPill.className = configured ? "status-pill is-ok" : "status-pill is-neutral";
      statusPill.textContent = configured
        ? t("hf_token.status_configured")
        : t("hf_token.status_unverified");
      return;
    }
    const formatted = formatVerifyStatus(verifyResult);
    statusPill.className = formatted.cls;
    statusPill.textContent = formatted.text;
  }

  const verifyBtn = createEl("button", {
    type: "button",
    className: "btn btn-secondary btn-sm",
    textContent: t("hf_token.verify_btn"),
    onClick: async () => {
      if (verifying || typeof onVerify !== "function") {
        return;
      }
      verifying = true;
      verifyBtn.disabled = true;
      verifyBtn.textContent = t("common.loading");
      try {
        const raw = inputEl.value.trim();
        verifyResult = await onVerify(raw || null);
      } catch {
        verifyResult = { valid: false, status: "network_error" };
      } finally {
        verifying = false;
        verifyBtn.disabled = false;
        verifyBtn.textContent = t("hf_token.verify_btn");
        refreshStatusPill();
      }
    },
  });

  const clearBtn =
    configured && typeof onClear === "function"
      ? createEl("button", {
          type: "button",
          className: "btn btn-ghost btn-sm",
          textContent: t("hf_token.clear_btn"),
          onClick: () => onClear(),
        })
      : null;

  return createEl(
    "div",
    { className: "form-group" },
    createEl(
      "div",
      { className: "label-row" },
      createEl("label", { className: "label", for: inputId, textContent: "HF_TOKEN" }),
      optionalLabel
        ? createEl("span", { className: "label-optional", textContent: t("setup.optional") })
        : null
    ),
    createEl("div", { className: "input-wrap" }, inputEl, toggleBtn),
    createEl(
      "div",
      { className: "verify-row" },
      createEl("div", { className: "verify-actions" }, verifyBtn, clearBtn),
      statusPill
    ),
    buildHfTutorialDetails()
  );
}
