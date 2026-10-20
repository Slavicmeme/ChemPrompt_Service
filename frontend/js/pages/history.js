App.register("history", async (view) => {
  const { predictions } = await API.get("/api/predictions?limit=100");
  const rows = predictions.map((p) => {
    const r = p.result || {};
    const val = p.status !== "SUCCEEDED" ? UI.esc(p.error || "") :
      p.task_type === "regression" ? `${UI.num(r.value)} ${UI.esc((p.property || {}).unit || "")}` :
      (r.tasks || []).map((t) => `${UI.esc((((p.property || {}).target_info || {})[t.name] || {}).name_ko || t.name)} ${UI.num(t.probability)}`).join(", ");
    return UI.html`<tr data-id="${p.prediction_id}"><td>${UI.date(p.created_at)}</td><td class="mono">${p.smiles}</td>
      <td>${p.property && p.property.name_ko ? `${p.property.name_ko} (${p.property.code})` : p.model_version}<div class="muted small">${p.model_version}</div></td><td>${UI.raw(val)}</td><td>${p.descriptor_mode} · ${p.split_strategy}</td>
      <td>${p.source}</td><td>${UI.statusBadge(p.status)}</td></tr>`;
  }).join("");
  view.innerHTML = UI.html`<div class="page-head"><h1>예측 기록</h1><p class="muted">행을 클릭하면 모델별(Model 1~5) 결과와 재현 정보를 볼 수 있습니다.</p></div>
    <div class="card table-wrap">${UI.raw(predictions.length ? `<table><tr><th>시각</th><th>SMILES</th><th>Property / 모델</th><th>결과</th><th>설정</th><th>경로</th><th>상태</th></tr>${rows}</table>` : '<div class="empty">아직 예측 기록이 없습니다.</div>')}</div>
    <div id="detail" style="margin-top:16px"></div>`;
  view.querySelectorAll("tr[data-id]").forEach((tr) => (tr.style.cursor = "pointer", tr.onclick = async () => {
    const p = await API.get(`/api/predictions/${tr.dataset.id}`);
    view.querySelector("#detail").innerHTML = renderPredictionResult(p);
    view.querySelector("#detail details").open = true;
  }));
});
