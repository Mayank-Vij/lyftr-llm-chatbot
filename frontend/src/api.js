// Backend base URL, injected at build time (VITE_API_URL).
export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export function newSessionId() {
  const rand =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 8)
      : Math.random().toString(16).slice(2, 10);
  return `s-${rand}`;
}

/** Turn any failure into a readable message. Never throws. */
function describeError(status, body) {
  const detail = body && body.detail;
  if (status === 503) return "All AI providers are unavailable right now. Please try again in a moment.";
  if (status === 422) {
    if (Array.isArray(detail) && detail.length) {
      const first = detail[0];
      const field = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : "input";
      return `Invalid input (${field}): ${first.msg}`;
    }
    return typeof detail === "string" ? `Invalid input: ${detail}` : "Invalid input.";
  }
  if (typeof detail === "string") return detail;
  return `Request failed (HTTP ${status}).`;
}

export async function sendChat({ sessionId, message, systemPrompt, simulateFailure }) {
  const payload = { session_id: sessionId, message };
  if (systemPrompt && systemPrompt.trim()) payload.system_prompt = systemPrompt.trim();
  const headers = { "Content-Type": "application/json" };
  if (simulateFailure && simulateFailure.trim()) headers["X-Simulate-Failure"] = simulateFailure.trim();

  let res;
  try {
    res = await fetch(`${API_URL}/chat`, { method: "POST", headers, body: JSON.stringify(payload) });
  } catch {
    return { ok: false, error: "Cannot reach the server. Check your connection and try again." };
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) return { ok: false, error: describeError(res.status, body) };
  if (!body || typeof body.reply !== "string" || !body.metadata) {
    return { ok: false, error: "The server sent an unexpected response." };
  }
  return { ok: true, data: body };
}
