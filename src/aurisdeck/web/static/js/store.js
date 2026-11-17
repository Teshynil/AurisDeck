/**
 * Lightweight reactive global store for AurisDeck Web UI.
 */

const DEFAULT_INITIAL_STATE = {
  user: null,
  authStatus: { auth_mode: "none", authenticated: false },
  preferences: {},
  locales: [],
  locale: "es",
  profiles: [],
  glossaries: [],
  replacements: [],
  devices: [],
  jobs: [],
  activeTab: "simple",
  selectedJobId: null,
  toasts: [],
  theme: "system",
};

let toastCounter = 0;

/**
 * Create a reactive state store with shallow patch updates and toast helpers.
 * @param {Record<string, any>} initialState
 */
export function createStore(initialState = {}) {
  let state = {
    ...DEFAULT_INITIAL_STATE,
    ...initialState,
    toasts: Array.isArray(initialState.toasts) ? [...initialState.toasts] : [],
  };
  const listeners = new Set();
  const toastTimers = new Map();

  function getState() {
    return state;
  }

  function setState(patchOrUpdater) {
    const patch =
      typeof patchOrUpdater === "function" ? patchOrUpdater(state) : patchOrUpdater;
    if (!patch || typeof patch !== "object") {
      return state;
    }

    const prevState = state;
    const nextState = { ...state, ...patch };
    const changed = Object.keys(patch).some(
      (key) => !Object.is(prevState[key], nextState[key])
    );
    if (!changed) {
      return state;
    }

    state = nextState;
    for (const listener of Array.from(listeners)) {
      try {
        listener(state, prevState);
      } catch (err) {
        console.error("Store listener failed:", err);
      }
    }
    return state;
  }

  function subscribe(listener) {
    if (typeof listener !== "function") {
      throw new TypeError("Store listener must be a function");
    }
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }

  function removeToast(id) {
    const timer = toastTimers.get(id);
    if (timer !== undefined) {
      clearTimeout(timer);
      toastTimers.delete(id);
    }
    setState((current) => ({
      toasts: current.toasts.filter((item) => item.id !== id),
    }));
  }

  function addToast(toast) {
    const entry = typeof toast === "string" ? { message: toast } : { ...(toast || {}) };
    const id = entry.id || `toast-${Date.now()}-${++toastCounter}`;
    const duration =
      typeof entry.duration === "number" ? entry.duration : 5000;
    const normalized = {
      id,
      type: entry.type || "info",
      title: entry.title ? String(entry.title) : "",
      message: entry.message ? String(entry.message) : "",
      createdAt: Date.now(),
    };

    setState((current) => ({
      toasts: [...current.toasts, normalized],
    }));

    if (duration > 0) {
      const timer = setTimeout(() => {
        removeToast(id);
      }, duration);
      toastTimers.set(id, timer);
    }

    return id;
  }

  return {
    getState,
    setState,
    subscribe,
    addToast,
    removeToast,
  };
}

export const store = createStore();
export default store;
