App.register("admin", async (view) => {
  if (!App.user || App.user.role !== "ADMIN") { view.innerHTML = '<div class="alert err">관리자 권한이 필요합니다.</div>'; return; }
  const [ov, mvs, bds, jobs, props, users] = await Promise.all([
    API.get("/api/admin/overview"), API.get("/api/admin/model-versions"), API.get("/api/admin/builtin-datasets"),
    API.get("/api/admin/training-jobs"), API.get("/api/properties"), API.get("/api/admin/users")]);
  const propOpts = props.properties.map((p) => UI.html`<option value="${p.code}">${p.code} (${p.task_type})</option>`).join("");
  const rep = ov.representation;

  const actions = (m) => {
    const b = [];
    if (["UPLOADED", "READY", "INVALID"].includes(m.status)) b.push(`<button class="btn ghost sm" data-act="validate" data-id="${m.model_version_id}">검증</button>`);
    if (m.status === "READY") b.push(`<button class="btn sm" data-act="serve" data-id="${m.model_version_id}">승인·Serving</button>`);
    if (["READY", "SERVING", "INVALID"].includes(m.status)) b.push(`<button class="btn ghost sm" data-act="archive" data-id="${m.model_version_id}">Archive</button>`);
    b.push(`<button class="btn ghost sm" data-act="detail" data-id="${m.model_version_id}">상세</button>`);
    return b.join(" ");
  };

  view.innerHTML = UI.html`
    <div class="page-head"><h1>Admin</h1><p class="muted">모델 검증·버전·serving 전환, 사용자 custom 모델 승인, built-in 데이터/학습 관리</p></div>
    <div class="grid two">
      <div class="card"><h2>Representation</h2><dl class="kv"><dt>Backend</dt><dd>${rep.label}</dd>
        <dt>Health</dt><dd>${rep.health ? JSON.stringify(rep.health) : "mock (GPU 서비스 미사용)"}</dd></dl>
        ${UI.raw(rep.backend === "mock" ? '<div class="alert warn">REPRESENTATION_MODE=mock — 개발용 hashing embedding입니다. Aya 결과가 아닙니다.</div>' : "")}</div>
      <div class="card"><h2>현황</h2><dl class="kv">${UI.raw(Object.entries(ov.model_version_counts).map(([k, v]) => UI.html`<dt>${k}</dt><dd>${v}</dd>`).join(""))}
        <dt>Users</dt><dd>${ov.users}</dd><dt>Jobs</dt><dd>${JSON.stringify(ov.training_jobs)}</dd></dl></div>
    </div>

    <div class="card" style="margin-top:16px"><h2>Model versions</h2><div class="table-wrap"><table>
      <tr><th>#</th><th>모델</th><th>scope</th><th>property</th><th>상태</th><th>설정</th><th>Representation</th><th>성능</th><th></th></tr>
      ${UI.raw(mvs.model_versions.map((m) => UI.html`<tr><td>${m.model_version_id}</td><td>${m.label}</td><td>${m.scope}${m.owner_id ? " (user " + m.owner_id + ")" : ""}</td>
        <td>${m.property || "–"}</td><td>${UI.statusBadge(m.status)}</td><td class="small">${m.descriptor_mode}<br>${m.split_strategy} ${JSON.stringify(m.split_seeds)}</td>
        <td class="small">${m.representation_model}</td><td class="small">${UI.raw(UI.metricsList(m.metrics, m.task_type))}</td><td><div class="actions">${UI.raw(actions(m))}</div></td></tr>`).join(""))}
    </table></div><div id="mv-detail"></div></div>

    <div class="grid two" style="margin-top:16px">
      <div class="card"><h2>모델 패키지 업로드</h2>
        <p class="muted small">zip: metadata.json, descriptor_catalog.json, fold_1..5/model.joblib, fold_k/selected_descriptors.json (GA: ga_flags.json). 저장 경로는 서버가 생성합니다.</p>
        <form id="up-form"><label>Property</label><select id="up-prop">${UI.raw(propOpts)}</select>
          <label>모델 이름</label><input id="up-name" required><label>Package (.zip)</label><input id="up-file" type="file" accept=".zip" required>
          <div class="row" style="margin-top:12px"><button class="btn">업로드 + 검증</button></div></form></div>
      <div class="card"><h2>Built-in 데이터 · 학습</h2>
        <form id="bd-form"><label>Property</label><select id="bd-prop">${UI.raw(propOpts)}</select>
          <label>CSV (smiles + property target columns)</label><input id="bd-file" type="file" accept=".csv" required>
          <div class="row" style="margin-top:12px"><button class="btn ghost">데이터 버전 등록</button></div></form>
        <hr style="border:none;border-top:1px solid var(--border);margin:16px 0">
        <form id="bt-form"><label>Dataset version</label><select id="bt-dv">${UI.raw(bds.datasets.filter((d) => d.status === "VALIDATED").map((d) =>
          UI.html`<option value="${d.dataset_version_id}" data-prop="${d.name}">${d.name} v${d.version} (#${d.dataset_version_id}, ${d.valid_row_count} rows)</option>`).join(""))}</select>
          <div class="grid two"><div><label>Descriptor</label><select id="bt-mode"><option>FULL</option><option>FEATURE_SELECTION</option><option>GENETIC</option></select></div>
          <div><label>Split</label><select id="bt-split"><option>RANDOM</option><option>SCAFFOLD</option></select></div></div>
          <div class="row" style="margin-top:12px"><button class="btn">5-fold 학습 시작</button></div></form></div>
    </div>

    <div class="card" style="margin-top:16px"><h2>Training jobs</h2><div class="table-wrap"><table>
      <tr><th>#</th><th>owner</th><th>상태</th><th>단계</th><th>모드</th><th>split</th><th>성능</th><th>오류</th><th></th></tr>
      ${UI.raw(jobs.jobs.map((j) => UI.html`<tr><td>${j.training_job_id}</td><td>${j.owner_id ?? "builtin"}</td><td>${UI.statusBadge(j.status)}</td><td class="small">${j.stage}</td>
        <td>${j.descriptor_mode}</td><td>${j.config.split_strategy}</td><td class="small">${UI.raw(UI.metricsList(j.metrics, j.task_type))}</td><td class="small">${(j.error || "").slice(0, 120)}</td>
        <td>${UI.raw(["QUEUED", "RUNNING"].includes(j.status) ? `<button class="btn ghost sm" data-job-cancel="${j.training_job_id}">중단</button>` : "")}</td></tr>`).join(""))}
    </table></div></div>

    <div class="card" style="margin-top:16px"><h2>Users</h2><table><tr><th>#</th><th>username</th><th>role</th><th>가입</th></tr>
      ${UI.raw(users.users.map((u) => UI.html`<tr><td>${u.id}</td><td>${u.username}</td><td>${u.role}</td><td>${UI.date(u.created_at)}</td></tr>`).join(""))}</table></div>`;

  view.querySelectorAll("[data-act]").forEach((b) => (b.onclick = async () => {
    const id = b.dataset.id, act = b.dataset.act;
    try {
      if (act === "detail") {
        const d = await API.get(`/api/admin/model-versions/${id}`);
        view.querySelector("#mv-detail").innerHTML = UI.html`<h3 style="margin-top:16px">Model version #${id}</h3>
          ${UI.raw(d.validation_report ? `<pre class="small">${UI.esc(JSON.stringify(d.validation_report, null, 2))}</pre>` : "")}
          <details open><summary>metadata.json</summary><pre class="small" style="max-height:360px;overflow:auto">${JSON.stringify(d.metadata, null, 2)}</pre></details>
          <details><summary>artifacts</summary><pre class="small">${JSON.stringify(d.artifacts, null, 2)}</pre></details>`;
        return;
      }
      if (act === "serve" && !confirm(`모델 버전 #${id}을(를) 승인하고 serving으로 전환할까요? (같은 모델의 기존 serving 버전은 READY로 내려갑니다)`)) return;
      if (act === "archive" && !confirm(`모델 버전 #${id}을(를) archive 할까요?`)) return;
      await API.post(`/api/admin/model-versions/${id}/${act}`, act === "validate" ? undefined : { note: null });
      UI.toast("완료되었습니다."); App.route();
    } catch (err) { UI.toast(err.message, 6000); }
  }));
  view.querySelectorAll("[data-job-cancel]").forEach((b) => (b.onclick = async () => {
    const id = b.dataset.jobCancel;
    if (!confirm(`학습 #${id}을(를) 중단할까요?\n진행 중인 단계가 끝나는 시점에 멈춤`)) return;
    try { const r = await API.post(`/api/admin/training-jobs/${id}/cancel`); UI.toast(r.status === "CANCELLING" ? "중단 요청함" : "중단됨"); App.route(); }
    catch (err) { UI.toast(err.message, 6000); }
  }));
  view.querySelector("#up-form").onsubmit = async (e) => {
    e.preventDefault();
    const fd = new FormData(); fd.append("property_code", view.querySelector("#up-prop").value);
    fd.append("model_name", view.querySelector("#up-name").value); fd.append("file", view.querySelector("#up-file").files[0]);
    try { const r = await API.form("/api/admin/models/upload", fd); UI.toast(`업로드 완료: ${r.status}`); App.route(); }
    catch (err) { UI.toast(err.message, 6000); }
  };
  view.querySelector("#bd-form").onsubmit = async (e) => {
    e.preventDefault();
    const fd = new FormData(); fd.append("property_code", view.querySelector("#bd-prop").value); fd.append("file", view.querySelector("#bd-file").files[0]);
    try { const r = await API.form("/api/admin/builtin-datasets", fd); UI.toast(`dataset version #${r.dataset_version_id}: ${r.status}`); App.route(); }
    catch (err) { UI.toast(err.message, 6000); }
  };
  view.querySelector("#bt-form").onsubmit = async (e) => {
    e.preventDefault();
    const sel = view.querySelector("#bt-dv");
    if (!sel.value) { UI.toast("VALIDATED built-in dataset version이 없습니다."); return; }
    try {
      const r = await API.post("/api/admin/builtin-training", { property_code: sel.selectedOptions[0].dataset.prop, dataset_version_id: Number(sel.value),
        descriptor_mode: view.querySelector("#bt-mode").value, split_strategy: view.querySelector("#bt-split").value });
      UI.toast(`training job #${r.training_job_id} 시작`); App.route();
    } catch (err) { UI.toast(err.message, 6000); }
  };
});
