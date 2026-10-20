App.register("agent", async (view) => {
  let convId = null;
  let controller = null;
  view.innerHTML = `
    <div class="page-head"><h1>Agent</h1>
      <p class="muted">자연어로 질문하세요. 예측 값은 Agent가 아니라 서비스의 예측 모델(Model 1~5)이 계산합니다.</p></div>
    <div class="chat">
      <div class="card"><div class="row" style="margin-bottom:8px"><h3 style="margin:0">대화</h3><span class="spacer"></span>
        <button class="btn ghost sm" id="new-conv">새 대화</button></div><div class="chat-list" id="conv-list"></div></div>
      <div class="card chat-main">
        <div class="messages" id="msgs"><div class="empty">예: "아스피린(CC(=O)Oc1ccccc1C(=O)O)의 수용해도를 예측해줘"</div></div>
        <div class="status-line" id="status"></div>
        <form class="composer" id="composer">
          <textarea id="input" placeholder="메시지 입력 (Enter 전송, Shift+Enter 줄바꿈)" required></textarea>
          <button class="btn" id="send">전송</button>
        </form>
      </div>
    </div>`;
  const msgs = view.querySelector("#msgs");
  const statusEl = view.querySelector("#status");
  const input = view.querySelector("#input");

  const add = (cls, text) => {
    if (msgs.querySelector(".empty")) msgs.innerHTML = "";
    const d = document.createElement("div"); d.className = cls; d.textContent = text || "";
    msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight; return d;
  };
  // assistant replies are rendered as Markdown (sanitized); user text stays plain
  const addMarkdown = (text) => {
    const d = add("msg assistant md", "");
    d._raw = text || "";
    d.innerHTML = UI.markdown(d._raw);
    return d;
  };
  const toolChip = (name, payload, label) => {
    if (msgs.querySelector(".empty")) msgs.innerHTML = "";
    const d = document.createElement("div"); d.className = "tool-chip";
    d.innerHTML = UI.html`🔧 <b>${name}</b> ${label}<details><summary>상세</summary><pre>${JSON.stringify(payload, null, 2)}</pre></details>`;
    msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight;
  };

  async function loadList() {
    const { conversations } = await API.get("/api/agent/conversations");
    const list = view.querySelector("#conv-list");
    list.innerHTML = conversations.map((c) => UI.html`<button data-id="${c.id}" class="${c.id === convId ? "active" : ""}">${c.title || "대화 " + c.id}</button>`).join("") || '<div class="muted small">대화 없음</div>';
    list.querySelectorAll("button").forEach((b) => (b.onclick = () => openConv(Number(b.dataset.id))));
  }
  async function openConv(id) {
    convId = id;
    const c = await API.get(`/api/agent/conversations/${id}`);
    msgs.innerHTML = "";
    for (const m of c.messages) {
      if (m.role === "user") add("msg user", m.content);
      else if (m.role === "assistant" && m.content) addMarkdown(m.content);
      else if (m.role === "tool") { let r = m.content; try { r = JSON.parse(m.content); } catch (_) {} toolChip(m.tool_name, r, "결과"); }
    }
    loadList();
  }
  view.querySelector("#new-conv").onclick = () => { convId = null; msgs.innerHTML = '<div class="empty">새 대화를 시작하세요.</div>'; loadList(); };

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = ""; add("msg user", text);
    const sendBtn = view.querySelector("#send"); sendBtn.disabled = true;
    let bubble = null;
    controller = new AbortController();
    try {
      await API.stream("POST", "/api/agent/chat", { message: text, conversation_id: convId }, (ev, data) => {
        if (ev === "status") {
          if (data.conversation_id) convId = data.conversation_id;
          statusEl.innerHTML = data.state === "model_call" ? `<span class="spin"></span> ${UI.esc(data.model)} 응답 생성 중` : "";
        } else if (ev === "token") {
          if (!bubble) bubble = addMarkdown("");
          bubble._raw += data.text;
          bubble.innerHTML = UI.markdown(bubble._raw); msgs.scrollTop = msgs.scrollHeight;
        } else if (ev === "tool_start") {
          bubble = null;
          statusEl.innerHTML = `<span class="spin"></span> tool 실행: ${UI.esc(data.name)}`;
          toolChip(data.name, data.arguments, "호출");
        } else if (ev === "tool_result") {
          toolChip(data.name, data.result, data.result && data.result.error ? "오류" : "결과");
        } else if (ev === "workflow_step") {
          statusEl.textContent = `workflow #${data.workflow_id}: step ${data.current_step} (${data.status})`;
        } else if (ev === "error") {
          add("alert err", data.error);
        } else if (ev === "done") {
          statusEl.textContent = "";
        }
      }, controller.signal);
    } catch (err) {
      if (err.name !== "AbortError") add("alert err", err.message);
    } finally { sendBtn.disabled = false; statusEl.textContent = ""; loadList(); }
  }
  view.querySelector("#composer").onsubmit = (e) => { e.preventDefault(); send(); };
  input.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } };
  App.cleanup = () => controller && controller.abort();
  loadList();
});
