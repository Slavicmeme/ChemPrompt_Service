function renderPredictionResult(p) {
  if (p.status !== "SUCCEEDED") {
    return UI.html`<div class="card"><div class="row"><code>${p.smiles}</code><span class="spacer"></span>${UI.statusBadge(p.status)}</div>
      <div class="alert err">${p.error || "예측 실패"}</div></div>`;
  }
  const r = p.result;
  const prop = p.property || {};
  const info = prop.target_info || {};
  let main;
  if (p.task_type === "regression") {
    const ti = info[r.target] || {};
    main = UI.html`<div class="muted small">${prop.name_ko || ti.name_ko || ""}${prop.code ? ` (${prop.code})` : ""}</div>
      <div class="result-value">${UI.num(r.value, 3)} <span class="muted small">${prop.unit || ""}</span></div>
      <div class="muted small">Model 1~${p.n_folds} 평균 · 모델 간 표준편차 ±${UI.num(r.fold_std, 3)}</div>
      ${UI.raw(ti.description ? UI.html`<div class="small" style="margin-top:6px">${ti.description}</div>` : "")}`;
  } else {
    main = `<table><tr><th>항목</th><th class="num">확률</th><th>판정 (기준 ${UI.num(r.threshold, 2)})</th><th class="num">모델 간 std</th></tr>` +
      r.tasks.map((t) => {
        const ti = info[t.name] || {};
        const verdict = t.label ? `1 · ${ti.positive || "positive"}` : `0 · ${ti.negative || "negative"}`;
        return UI.html`<tr><td><b>${ti.name_ko || t.name}</b> <span class="muted small">${t.name}</span>
          ${UI.raw(ti.description ? UI.html`<div class="muted small">${ti.description}</div>` : "")}</td>
          <td class="num">${UI.num(t.probability, 3)}</td>
          <td>${UI.raw(`<span class="badge ${t.label ? "warn" : ""}">${UI.esc(verdict)}</span>`)}</td>
          <td class="num">${UI.num(t.fold_std, 3)}</td></tr>`;
      }).join("") + "</table>" +
      UI.html`<div class="muted small" style="margin-top:6px">확률은 Model 1~${p.n_folds}의 평균이며, 기준 ${UI.num(r.threshold, 2)} 이상이면 1로 판정함</div>`;
  }
  const folds = (p.fold_results || []).map((f) => UI.html`<tr><td>Model ${f.fold}</td><td class="mono">${
    f.value !== undefined ? UI.num(f.value, 4) : Object.entries(f.probabilities).map(([k, v]) => `${(info[k] || {}).name_ko || k}=${UI.num(v, 3)}`).join(", ")}</td></tr>`).join("");
  return UI.html`<div class="card">
    <div class="row"><code>${r.canonical_smiles || p.smiles}</code><span class="spacer"></span>${UI.statusBadge(p.status)}</div>
    <div style="margin:10px 0">${UI.raw(main)}</div>
    <details><summary>모델별 결과 · 재현 정보</summary>
      <table>${UI.raw(folds)}</table>
      <dl class="kv" style="margin-top:10px">
        <dt>Model</dt><dd>${p.model_version}</dd><dt>Descriptor</dt><dd>${p.descriptor_mode}</dd>
        <dt>모델</dt><dd>Model 1~${p.n_folds} (${p.split_strategy} 분할)</dd>
        <dt>Representation</dt><dd>${p.representation_model} (${p.prompt_version})</dd>
        <dt>집계 방식</dt><dd>${p.task_type === "regression" ? `Model 1~${p.n_folds} 예측 평균` : `Model 1~${p.n_folds} 확률 평균 → 기준값 판정`}</dd><dt>Prediction ID</dt><dd>${p.prediction_id}</dd>
      </dl></details></div>`;
}

App.register("predict", async (view) => {
  const [{ properties }, { model_versions }] = await Promise.all([API.get("/api/properties"), API.get("/api/models")]);
  const custom = model_versions.filter((m) => m.scope === "USER" && m.usable);
  let selected = (properties.find((p) => p.serving_model) || {}).code || null;
  let selectedMv = null;

  const cards = properties.map((p) => UI.html`
    <div class="card pick ${p.code === selected ? "selected" : ""}" data-code="${p.code}">
      <div class="row"><h3 style="margin:0">${p.code}</h3><span class="spacer"></span>
        ${UI.raw(p.serving_model ? '<span class="badge ok">serving</span>' : '<span class="badge">모델 없음</span>')}</div>
      <div style="font-weight:600;margin-top:4px">${p.name_ko || p.name}</div>
      <div class="muted small">${p.name}</div>
      <div class="muted small" style="margin-top:4px">${{ regression: "회귀 (수치 예측)", classification: "분류 (예/아니오)", multilabel: "분류 (예/아니오 여러 항목)" }[p.task_type] || p.task_type}${p.unit ? " · " + p.unit : ""}</div>
      ${UI.raw(p.description ? UI.html`<div class="small" style="margin-top:6px">${p.description}</div>` : "")}
      ${UI.raw(p.task_type !== "regression" && p.target_info ? `<ul class="small" style="margin:6px 0 0;padding-left:18px">${
        Object.entries(p.target_info).map(([k, v]) => UI.html`<li><b>${v.name_ko || k}</b> <span class="muted">(${k})</span>: 1 = ${v.positive || "예"}, 0 = ${v.negative || "아니오"}</li>`).join("")}</ul>` : "")}
      ${UI.raw(p.serving_model ? UI.html`<div class="muted small" style="margin-top:6px">${p.serving_model.label}<br>${p.serving_model.descriptor_mode} · ${p.serving_model.split_strategy}</div>` : "")}
    </div>`).join("");
  const customOpts = custom.map((m) => UI.html`<option value="${m.model_version_id}">${m.label} (${m.task_type}, ${m.owner_id === App.user.id ? "내 모델" : "승인된 공개 모델"}, ${m.status})</option>`).join("");

  view.innerHTML = UI.html`
    <div class="page-head"><h1>분자 물성 예측</h1>
      <p class="muted">Property를 선택하고 SMILES를 입력하세요. 값은 여러 모델(Model 1~5)의 예측 평균입니다.</p></div>
    <div class="grid cards" id="props">${UI.raw(cards)}</div>
    ${UI.raw(custom.length ? UI.html`<div class="card" style="margin-top:16px"><label for="mv">또는 Custom 모델 사용 (내 모델 + Admin이 승인한 공개 모델)</label>
      <select id="mv"><option value="">— built-in property 사용 —</option>${UI.raw(customOpts)}</select></div>` : "")}
    <div class="card" style="margin-top:16px">
      <label for="smiles">SMILES (한 줄에 하나, 최대 100개)</label>
      <textarea id="smiles" placeholder="CCO&#10;c1ccccc1O"></textarea>
      <div class="row" style="margin-top:12px"><button class="btn" id="run">예측 실행</button>
        <span class="muted small" id="target-label"></span></div>
    </div>
    <div id="results" class="grid" style="margin-top:16px"></div>`;

  const label = () => {
    view.querySelector("#target-label").textContent = selectedMv ? `Custom 모델 #${selectedMv}` : selected ? `Property: ${selected}${(properties.find((x) => x.code === selected) || {}).name_ko ? " (" + properties.find((x) => x.code === selected).name_ko + ")" : ""}` : "Property를 선택하세요";
  };
  label();
  view.querySelectorAll("#props .card").forEach((c) => (c.onclick = () => {
    selected = c.dataset.code;
    view.querySelectorAll("#props .card").forEach((x) => x.classList.toggle("selected", x === c));
    label();
  }));
  const mvSel = view.querySelector("#mv");
  if (mvSel) mvSel.onchange = () => { selectedMv = mvSel.value ? Number(mvSel.value) : null; label(); };

  view.querySelector("#run").onclick = async (e) => {
    const smiles = view.querySelector("#smiles").value.split("\n").map((s) => s.trim()).filter(Boolean);
    const out = view.querySelector("#results");
    if (!smiles.length) { UI.toast("SMILES를 입력하세요."); return; }
    e.target.disabled = true;
    out.innerHTML = `<div class="empty"><span class="spin"></span> descriptor 계산 → representation → Model 1~5 예측 중…</div>`;
    try {
      const body = selectedMv ? { smiles, model_version_id: selectedMv } : { smiles, property: selected };
      const r = await API.post("/api/predictions", body);
      out.innerHTML = r.predictions.map(renderPredictionResult).join("");
    } catch (err) {
      out.innerHTML = UI.html`<div class="alert err">${err.message}</div>`;
    } finally { e.target.disabled = false; }
  };
});
