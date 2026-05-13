(function () {
  // ── Config ────────────────────────────────────────────────
  const API_BASE = document.currentScript?.src
    ? new URL(document.currentScript.src).origin
    : "http://localhost:8000";

  const SESSION_ID = "session_" + Math.random().toString(36).slice(2, 10);

  // ── Inject styles ─────────────────────────────────────────
  const style = document.createElement("style");
  style.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');

    #rag-widget-btn {
      position: fixed;
      bottom: 28px;
      right: 28px;
      width: 52px;
      height: 52px;
      border-radius: 50%;
      background: #111;
      border: none;
      cursor: pointer;
      box-shadow: 0 2px 12px rgba(0,0,0,.18);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 99998;
      transition: transform .2s ease, box-shadow .2s ease, background .2s;
    }
    #rag-widget-btn:hover {
      transform: scale(1.07);
      background: #222;
      box-shadow: 0 4px 20px rgba(0,0,0,.22);
    }
    #rag-widget-btn svg { width: 22px; height: 22px; fill: #fff; }

    #rag-widget-panel {
      position: fixed;
      bottom: 92px;
      right: 28px;
      width: 360px;
      max-height: 680px;
      background: #fff;
      border: 1px solid #e5e5e5;
      border-radius: 16px;
      box-shadow: 0 8px 40px rgba(0,0,0,.10), 0 1px 4px rgba(0,0,0,.06);
      display: flex;
      flex-direction: column;
      z-index: 99999;
      font-family: 'DM Sans', sans-serif;
      overflow: hidden;
      transform: scale(0.94) translateY(10px);
      opacity: 0;
      pointer-events: none;
      transition: transform .22s cubic-bezier(.34,1.56,.64,1), opacity .18s ease;
    }
    #rag-widget-panel.open {
      transform: scale(1) translateY(0);
      opacity: 1;
      pointer-events: all;
    }

    /* Header */
    #rag-header {
      background: #111;
      padding: 14px 16px;
      display: flex;
      align-items: center;
      gap: 10px;
      border-bottom: 1px solid #000;
    }
    #rag-header-dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: #fff;
      opacity: 0.9;
      animation: ragPulse 2.4s ease infinite;
    }
    @keyframes ragPulse {
      0%, 100% { opacity: 0.9; } 50% { opacity: 0.3; }
    }
    #rag-header-title {
      flex: 1;
      color: #fff;
      font-weight: 500;
      font-size: .85rem;
      letter-spacing: .01em;
    }
    #rag-header-sub {
      font-size: .65rem;
      color: rgba(255,255,255,.4);
      font-weight: 300;
      letter-spacing: .02em;
    }
    #rag-clear-btn {
      background: rgba(255,255,255,.08);
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 5px;
      color: rgba(255,255,255,.45);
      font-size: .65rem;
      font-family: inherit;
      padding: 4px 9px;
      cursor: pointer;
      transition: background .15s, color .15s;
      letter-spacing: .02em;
    }
    #rag-clear-btn:hover {
      background: rgba(255,255,255,.15);
      color: rgba(255,255,255,.85);
    }

    /* Messages */
    #rag-messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px 14px 8px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      scrollbar-width: thin;
      scrollbar-color: #e0e0e0 transparent;
      background: #fafafa;
    }
    .rag-msg {
      max-width: 86%;
      padding: 9px 13px;
      border-radius: 12px;
      font-size: .81rem;
      line-height: 1.55;
      animation: ragFadeIn .2s ease;
    }
    @keyframes ragFadeIn {
      from { opacity: 0; transform: translateY(5px); }
      to   { opacity: 1; transform: none; }
    }
    .rag-msg.user {
      align-self: flex-end;
      background: #111;
      color: #fff;
      border-bottom-right-radius: 3px;
      font-weight: 400;
    }
    .rag-msg.bot {
      align-self: flex-start;
      background: #fff;
      color: #222;
      border-bottom-left-radius: 3px;
      border: 1px solid #e8e8e8;
      white-space: pre-wrap;
    }
    .rag-msg.bot .rag-sources {
      margin-top: 7px;
      padding-top: 7px;
      border-top: 1px solid #ebebeb;
      font-size: .67rem;
      color: #aaa;
      font-family: monospace;
      line-height: 1.6;
    }

    /* Typing indicator */
    #rag-typing {
      display: none;
      align-self: flex-start;
      padding: 10px 14px;
      background: #fff;
      border: 1px solid #e8e8e8;
      border-radius: 12px;
      border-bottom-left-radius: 3px;
      gap: 4px;
      align-items: center;
    }
    #rag-typing.show { display: flex; }
    .rag-dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: #bbb;
      animation: ragBounce 1.1s ease infinite;
    }
    .rag-dot:nth-child(2) { animation-delay: .18s; }
    .rag-dot:nth-child(3) { animation-delay: .36s; }
    @keyframes ragBounce {
      0%, 80%, 100% { transform: translateY(0); opacity: .4; }
      40%            { transform: translateY(-5px); opacity: 1; }
    }

    /* Input area */
    #rag-input-area {
      padding: 10px 12px 12px;
      border-top: 1px solid #ebebeb;
      display: flex;
      gap: 7px;
      background: #fff;
      align-items: flex-end;
    }
    #rag-input {
      flex: 1;
      background: #f5f5f5;
      border: 1px solid #e8e8e8;
      border-radius: 9px;
      color: #111;
      font-family: inherit;
      font-size: .81rem;
      padding: 9px 12px;
      resize: none;
      outline: none;
      transition: border-color .15s, background .15s;
      height: 40px;
      max-height: 100px;
      overflow-y: auto;
    }
    #rag-input:focus {
      border-color: #aaa;
      background: #fff;
    }
    #rag-input::placeholder { color: #bbb; }

    /* Mic button */
    #rag-mic-btn {
      width: 38px; height: 38px;
      border-radius: 9px;
      background: #f5f5f5;
      border: 1px solid #e8e8e8;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background .15s, border-color .15s;
      flex-shrink: 0;
    }
    #rag-mic-btn:hover {
      background: #ececec;
      border-color: #d0d0d0;
    }
    #rag-mic-btn svg {
      width: 17px; height: 17px;
      fill: #888;
      transition: fill .15s;
    }

    /* Recording state */
    #rag-mic-btn.recording {
      background: #111;
      border-color: #111;
      animation: ragMicPulse 1s ease infinite;
    }
    #rag-mic-btn.recording svg { fill: #fff; }
    @keyframes ragMicPulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(0,0,0,.2); }
      50%       { box-shadow: 0 0 0 5px rgba(0,0,0,0); }
    }

    /* Transcribing state */
    #rag-mic-btn.transcribing {
      background: #f0f0f0;
      border-color: #ccc;
      cursor: not-allowed;
    }
    #rag-mic-btn.transcribing svg { fill: #555; }

    /* Send button */
    #rag-send-btn {
      width: 38px; height: 38px;
      border-radius: 9px;
      background: #111;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background .15s, transform .15s;
      flex-shrink: 0;
    }
    #rag-send-btn:hover  { background: #333; transform: scale(1.05); }
    #rag-send-btn:active { transform: scale(.96); }
    #rag-send-btn svg { width: 16px; height: 16px; fill: white; }
    #rag-send-btn:disabled { opacity: .35; cursor: not-allowed; transform: none; }

    /* Mic status toast */
    #rag-mic-status {
      position: absolute;
      bottom: 70px;
      left: 50%;
      transform: translateX(-50%);
      background: #111;
      border-radius: 20px;
      padding: 5px 14px;
      font-size: .70rem;
      color: rgba(255,255,255,.85);
      white-space: nowrap;
      pointer-events: none;
      opacity: 0;
      transition: opacity .2s;
      letter-spacing: .01em;
    }
    #rag-mic-status.show { opacity: 1; }

    @media (max-width: 420px) {
      #rag-widget-panel { width: calc(100vw - 24px); right: 12px; }
    }
  `;
  document.head.appendChild(style);

  // ── Build DOM ─────────────────────────────────────────────
  const btn = document.createElement("button");
  btn.id = "rag-widget-btn";
  btn.title = "Chat with Amazon Agent";
  btn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>`;
  document.body.appendChild(btn);

  const panel = document.createElement("div");
  panel.id = "rag-widget-panel";
  panel.innerHTML = `
    <div id="rag-header">
      <div id="rag-header-dot"></div>
      <div>
        <div id="rag-header-title">Amazon Customer Agent</div>
        <div id="rag-header-sub">Multilingual · RAG-powered</div>
      </div>
      <button id="rag-clear-btn">Clear chat</button>
    </div>
    <div id="rag-messages">
      <div class="rag-msg bot">Hi! 👋 Ask me anything about Amazon India — returns, refunds, Prime, deliveries. I speak English, Hindi, and Hinglish!</div>
    </div>
    <div id="rag-typing">
      <div class="rag-dot"></div>
      <div class="rag-dot"></div>
      <div class="rag-dot"></div>
    </div>
    <div id="rag-input-area">
      <textarea id="rag-input" placeholder="Ask in English, Hindi, or Hinglish…" rows="1"></textarea>
      <button id="rag-mic-btn" title="Click to record voice">
        <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z"/></svg>
      </button>
      <button id="rag-send-btn">
        <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
      </button>
    </div>
    <div id="rag-mic-status"></div>
  `;
  document.body.appendChild(panel);

  // ── State ─────────────────────────────────────────────────
  let isOpen        = false;
  let isWaiting     = false;
  let mediaRecorder = null;
  let audioChunks   = [];
  let isRecording   = false;

  const messagesEl = panel.querySelector("#rag-messages");
  const inputEl    = panel.querySelector("#rag-input");
  const sendBtn    = panel.querySelector("#rag-send-btn");
  const typingEl   = panel.querySelector("#rag-typing");
  const clearBtn   = panel.querySelector("#rag-clear-btn");
  const micBtn     = panel.querySelector("#rag-mic-btn");
  const micStatus  = panel.querySelector("#rag-mic-status");

  // ── Toggle panel ──────────────────────────────────────────
  btn.addEventListener("click", () => {
    isOpen = !isOpen;
    panel.classList.toggle("open", isOpen);
    if (isOpen) inputEl.focus();
  });

  // ── Clear session ─────────────────────────────────────────
  clearBtn.addEventListener("click", async () => {
    await fetch(`${API_BASE}/session/${SESSION_ID}`, { method: "DELETE" });
    messagesEl.innerHTML = `<div class="rag-msg bot">Chat cleared! Ask me anything. 🙂</div>`;
  });

  // ── Mic status toast helper ───────────────────────────────
  function showStatus(text, duration = 0) {
    micStatus.textContent = text;
    micStatus.classList.add("show");
    if (duration > 0) setTimeout(() => micStatus.classList.remove("show"), duration);
  }

  // ── Mic button: click to start / click to stop ───────────
  micBtn.addEventListener("click", async () => {
    if (isRecording) {
      mediaRecorder.stop();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks  = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        isRecording = false;
        micBtn.classList.remove("recording");
        micBtn.classList.add("transcribing");
        showStatus("Transcribing…");

        const blob     = new Blob(audioChunks, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("file", blob, "recording.webm");

        try {
          const res  = await fetch(`${API_BASE}/transcribe`, {
            method: "POST",
            body: formData,
          });
          if (!res.ok) throw new Error(`Server error ${res.status}`);
          const data = await res.json();

          if (data.transcript) {
            inputEl.value = data.transcript;
            inputEl.style.height = "40px";
            inputEl.style.height = Math.min(inputEl.scrollHeight, 100) + "px";
            inputEl.focus();
            showStatus("✓ Transcribed", 1800);
          } else {
            showStatus("Nothing heard — try again", 2000);
          }
        } catch (err) {
          console.error("[STT]", err);
          showStatus("Transcription failed", 2000);
        } finally {
          micBtn.classList.remove("transcribing");
        }
      };

      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add("recording");
      showStatus("● Recording… click to stop");

    } catch (err) {
      console.error("[MIC]", err);
      showStatus("Microphone access denied", 2500);
    }
  });

  // ── Add message bubble ────────────────────────────────────
  function addMessage(text, role) {
    const div = document.createElement("div");
    div.className = `rag-msg ${role}`;

    if (role === "bot") {
      const parts = text.split("\n📚 Sources:");
      div.textContent = parts[0].trim();
      if (parts[1]) {
        const src = document.createElement("div");
        src.className = "rag-sources";
        src.textContent = "📚 Sources:" + parts[1];
        div.appendChild(src);
      }
    } else {
      div.textContent = text;
    }

    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  // ── Send question ─────────────────────────────────────────
  async function sendMessage() {
    const question = inputEl.value.trim();
    if (!question || isWaiting) return;

    isWaiting = true;
    sendBtn.disabled = true;
    inputEl.value = "";
    inputEl.style.height = "40px";

    addMessage(question, "user");

    typingEl.classList.add("show");
    messagesEl.appendChild(typingEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: SESSION_ID }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();

      typingEl.classList.remove("show");
      addMessage(data.answer, "bot");
    } catch (err) {
      typingEl.classList.remove("show");
      addMessage("Sorry, something went wrong. Please try again.", "bot");
      console.error("[RAG Widget]", err);
    } finally {
      isWaiting = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  sendBtn.addEventListener("click", sendMessage);

  inputEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  inputEl.addEventListener("input", () => {
    inputEl.style.height = "40px";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 100) + "px";
  });

})();