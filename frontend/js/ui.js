const UI = {
  esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  },
  html(strings, ...vals) {
    return strings.reduce((acc, s, i) => acc + s + (i < vals.length ? (vals[i] && vals[i].__raw ? vals[i].s : UI.esc(vals[i])) : ""), "");
  },
  raw(s) { return { __raw: true, s }; },
  num(v, d = 3) { return v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d); },
  date(s) { return s ? new Date(s).toLocaleString("ko-KR") : ""; },
  toast(msg, ms = 3200) {
    const t = document.getElementById("toast");
    t.textContent = msg; t.hidden = false;
    clearTimeout(UI._t); UI._t = setTimeout(() => (t.hidden = true), ms);
  },
  statusBadge(s) {
    const cls = { SERVING: "ok", READY: "acc", SUCCEEDED: "ok", COMPLETED: "ok", VALIDATED: "ok", RUNNING: "warn", QUEUED: "warn",
      VALIDATING: "warn", ACTIVE: "acc", FAILED: "err", INVALID: "err", ARCHIVED: "", UPLOADED: "", MAPPED: "", CANCELLED: "" }[s] ?? "";
    return UI.raw(`<span class="badge ${cls}">${UI.esc(s)}</span>`);
  },
  // Validation metrics as one line per metric (HTML; use with UI.raw)
  metricsList(metrics, task) {
    if (!metrics) return '<span class="muted">–</span>';
    const names = { rmse: "RMSE", r2: "R²", pearson: "Pearson", spearman: "Spearman", roc_auc: "ROC-AUC", roc_auc_macro: "ROC-AUC",
      f1_macro: "F1", accuracy: "Accuracy", accuracy_macro: "Accuracy" };
    const keys = task === "regression" ? ["rmse", "r2", "pearson"] : task === "multilabel" ? ["roc_auc_macro", "f1_macro", "accuracy_macro"] : ["roc_auc", "f1_macro", "accuracy"];
    return '<div class="metrics">' + keys.filter((k) => metrics[k]).map((k) => {
      const m = metrics[k];
      const mean = typeof m === "object" ? m.mean : m;
      const std = typeof m === "object" && m.std != null ? ` <span class="muted">± ${UI.num(m.std)}</span>` : "";
      return `<div><span class="metric-name">${names[k] || UI.esc(k)}</span> ${UI.num(mean)}${std}</div>`;
    }).join("") + "</div>";
  },
  metricNote(task) {
    return task === "regression" ? "RMSE는 target의 원래 단위입니다." : "";
  },
  // Markdown -> sanitized HTML (assistant messages are model output: always sanitize)
  markdown(text) {
    if (!window.marked || !window.DOMPurify) return UI.esc(text).replace(/\n/g, "<br>");
    const html = marked.parse(String(text ?? ""), { gfm: true, breaks: true });
    return DOMPurify.sanitize(html, { ADD_ATTR: ["target"] });
  },
  // Modal with explicit choices. Resolves to the chosen value, or null on cancel / Esc / outside click.
  choose({ title, message = "", choices }) {
    return new Promise((resolve) => {
      const wrap = document.createElement("div");
      wrap.className = "modal-backdrop";
      wrap.innerHTML = UI.html`<div class="modal" role="dialog" aria-modal="true">
        <h3>${title}</h3><div class="modal-body">${UI.raw(UI.esc(message).replace(/\n/g, "<br>"))}</div>
        <div class="modal-actions"><button class="btn ghost" data-v="__cancel">취소</button>${UI.raw(
          choices.map((c, i) => `<button class="btn ${c.cls || ""}" data-v="${i}">${UI.esc(c.label)}</button>`).join(""))}</div></div>`;
      const done = (v) => { document.removeEventListener("keydown", onKey); wrap.remove(); resolve(v); };
      const onKey = (e) => { if (e.key === "Escape") done(null); };
      wrap.addEventListener("click", (e) => {
        if (e.target === wrap) return done(null);
        const b = e.target.closest("button[data-v]");
        if (b) done(b.dataset.v === "__cancel" ? null : choices[Number(b.dataset.v)].value);
      });
      document.addEventListener("keydown", onKey);
      document.body.appendChild(wrap);
      wrap.querySelector('[data-v="__cancel"]').focus();
    });
  },
  view() { return document.getElementById("view"); },
};
