import { useEffect, useRef, useState } from "react";
import { newSessionId, sendChat, API_URL } from "./api.js";

const SUGGESTIONS = ["What is the capital of France?", "Explain caching in one sentence."];

function Meta({ m }) {
  const cached = m.resolution_source === "LLM_CACHE";
  return (
    <div className="meta">
      <span className={`pill ${cached ? "pill-cache" : "pill-live"}`} data-testid="meta-source">
        {m.resolution_source}
      </span>
      <span className="meta-item">
        model <b data-testid="meta-model">{m.model}</b>
      </span>
      <span className="meta-item">
        latency <b data-testid="meta-latency">{m.latency_ms} ms</b>
      </span>
      <span className="meta-item">
        in <b data-testid="meta-input-tokens">{m.input_tokens}</b>
      </span>
      <span className="meta-item">
        out <b data-testid="meta-output-tokens">{m.output_tokens}</b>
      </span>
      {m.key_id && (
        <span className="meta-item dim">
          {m.key_id}
          {m.fallback_attempts > 0 ? ` · ${m.fallback_attempts} failover` : ""}
        </span>
      )}
    </div>
  );
}

export default function App() {
  // A new session starts on every page load.
  const [sessionId] = useState(newSessionId);
  const [items, setItems] = useState([]); // {id, kind: user|assistant|error, text, meta?}
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulate, setSimulate] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);
  const idRef = useRef(0);
  const nextId = () => ++idRef.current;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [items, busy]);

  async function send(text) {
    const message = (text ?? input).trim();
    if (busy) return; // validation (empty / too long) is the backend's job; its 422 is shown in the chat
    setInput("");
    if (message) setItems((prev) => [...prev, { id: nextId(), kind: "user", text: message }]);
    setBusy(true);
    const result = await sendChat({ sessionId, message, systemPrompt, simulateFailure: simulate });
    setBusy(false);
    setItems((prev) => [
      ...prev,
      result.ok
        ? { id: nextId(), kind: "assistant", text: result.data.reply, meta: result.data.metadata }
        : { id: nextId(), kind: "error", text: result.error },
    ]);
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>LLM Chatbot</h1>
          <p className="sub">
            Session <code>{sessionId}</code> · key failover · response cache
          </p>
        </div>
        <button className="ghost" onClick={() => setShowSettings((v) => !v)} aria-expanded={showSettings}>
          {showSettings ? "Hide" : "Settings"}
        </button>
      </header>

      <section className={`settings ${showSettings ? "open" : ""}`}>
        <label htmlFor="sp">System prompt (optional, applies to this session)</label>
        <textarea
          id="sp"
          data-testid="system-prompt-input"
          rows={2}
          placeholder="e.g. You are a pirate. Answer in one sentence."
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
        />
        <label htmlFor="sim">Failover demo: X-Simulate-Failure header (needs backend ENABLE_FAILURE_SIMULATION=true)</label>
        <input
          id="sim"
          data-testid="simulate-failure-input"
          placeholder="key_1=429   ·   key_1=429,key_2=500   ·   *=429"
          value={simulate}
          onChange={(e) => setSimulate(e.target.value)}
        />
      </section>

      <main className="chat" aria-live="polite">
        {items.length === 0 && (
          <div className="empty">
            <p>Ask anything. Each reply shows where it came from, the model, latency and token usage.</p>
            <div className="chips">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="chip" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {items.map((it) =>
          it.kind === "assistant" ? (
            <div className="row left" key={it.id}>
              <div className="bubble assistant" data-testid="assistant-message">
                <div className="text">{it.text}</div>
                <Meta m={it.meta} />
              </div>
            </div>
          ) : it.kind === "user" ? (
            <div className="row right" key={it.id}>
              <div className="bubble user" data-testid="user-message">{it.text}</div>
            </div>
          ) : (
            <div className="row left" key={it.id}>
              <div className="bubble error" role="alert" data-testid="error-message">{it.text}</div>
            </div>
          )
        )}
        {busy && (
          <div className="row left">
            <div className="bubble assistant typing" aria-label="Assistant is typing">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </main>

      <footer className="composer">
        <textarea
          data-testid="chat-input"
          rows={1}
          placeholder="Type a message… (Enter to send, Shift+Enter for a new line)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button data-testid="send-button" onClick={() => send()} disabled={busy}>
          {busy ? "…" : "Send"}
        </button>
      </footer>
      <div className="api-note">API: {API_URL}</div>
    </div>
  );
}
