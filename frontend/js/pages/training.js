// Custom Dataset Training workflow (10 steps; steps 7-10 are executed by the backend).
const STEP_FORMS = {
  1: (wf, ctx) => `
    <label>Dataset 이름</label><input id="f-name" required placeholder="예: 내 화합물 logS 측정값">
    <label>설명 (선택)</label><input id="f-desc">
    <label>CSV 파일 (SMILES column + target column)</label><input id="f-file" type="file" accept=".csv" required>`,
  2: (wf, ctx) => {
    const s1 = wf.steps[0].result || {};
    const sug = s1.suggested_mapping || {};
    const prof = sug.columns_profile || (s1.columns || []).map((c) => ({ name: c, examples: [] }));
    const label = (p) => `${p.name}${p.examples && p.examples.length ? " — 예: " + p.examples.slice(0, 2).join(", ") : ""}`;
    const options = (selected) => `<option value="">— 열을 선택하세요 —</option>` +
      prof.map((p) => `<option value="${UI.esc(p.name)}" ${p.name === selected ? "selected" : ""}>${UI.esc(label(p))}</option>`).join("");
    const rows = prof.map((p) => UI.html`<tr><td><b>${p.name}</b></td><td class="mono small">${(p.examples || []).join(" · ")}</td>
      <td>${UI.raw(p.smiles_ratio >= 0.8 ? '<span class="badge ok">SMILES로 보임</span>' : p.numeric_ratio >= 0.9 ? '<span class="badge acc">숫자</span>' : '<span class="badge">문자</span>')}</td></tr>`).join("");
    return `<p class="muted small">CSV에서 <b>분자 구조(SMILES)</b>가 든 열과 <b>예측할 값(target)</b>이 든 열을 하나씩 고르세요.</p>
      <label>① SMILES 열 (분자 구조)</label><select id="f-smiles" required>${options(sug.smiles_column)}</select>
      <label>② Target 열 (예측할 숫자 값)</label><select id="f-target" required>${options((sug.target_columns || [])[0])}</select>
      <details style="margin-top:10px" open><summary>CSV 열 미리보기</summary>
        <table style="margin-top:6px"><tr><th>열</th><th>앞쪽 값</th><th>내용</th></tr>${rows}</table></details>`;
  },
  3: () => `<p class="muted small">결측/숫자가 아닌 target과 RDKit이 해석하지 못하는 SMILES를 제거하고, 모델 학습이 가능한지 확인합니다.</p>`,
  4: (wf) => {
    const rep = wf.steps[2].result || {};
    const t = rep.inferred_task_type;
    const opts = t === "classification" ? ["classification", "regression"] : [t];
    return `<p class="muted small">데이터에서 추론된 task: <b>${UI.esc(t)}</b></p>
      <label>Task type</label><select id="f-task">${opts.map((o) => `<option>${o}</option>`).join("")}</select>
      <label>모델 이름</label><input id="f-mname" placeholder="예: logS ChemPrompt v1">`;
  },
  5: () => `<label>Descriptor mode</label><select id="f-mode">
      <option value="FULL">FULL — 필터링된 전체 descriptor</option>
      <option value="FEATURE_SELECTION">FEATURE_SELECTION — 각 모델의 학습 데이터에서 Top-50 (f_regression)</option>
      <option value="GENETIC">GENETIC — Top-50 후보 + GA descriptor 선택 (모델별)</option></select>`,
  6: () => `<label>데이터 분할 방식 (Model 1~5를 학습하며, 각 모델은 데이터의 약 80%로 학습하고 나머지 20%로 검증)</label><select id="f-split">
      <option value="RANDOM">RANDOM — 무작위 분할</option>
      <option value="SCAFFOLD">SCAFFOLD — 분자 골격(scaffold) 기준 분할</option></select>
      <p class="muted small">이 단계를 완료하면 representation 생성 → Model 1~5 학습 → 평가 → 모델 등록이 자동으로 실행됩니다.</p>`,
};

async function collectStepInput(n, view) {
  const v = (id) => view.querySelector(id);
  if (n === 1) {
    const file = v("#f-file").files[0];
    if (!file) throw new Error("CSV 파일을 선택하세요.");
    const fd = new FormData(); fd.append("name", v("#f-name").value || file.name); fd.append("description", v("#f-desc").value); fd.append("file", file);
    const ds = await API.form("/api/datasets", fd);
    return { dataset_version_id: ds.dataset_version_id };
  }
  if (n === 2) {
    const smiles = v("#f-smiles").value, target = v("#f-target").value;
    if (!smiles || !target) throw new Error("SMILES 열과 Target 열을 모두 선택하세요.");
    if (smiles === target) throw new Error("SMILES 열과 Target 열은 서로 다른 열이어야 합니다.");
    return { smiles_column: smiles, target_columns: [target] };
  }
  if (n === 3) return {};
  if (n === 4) return { task_type: v("#f-task").value, model_name: v("#f-mname").value };
  if (n === 5) return { descriptor_mode: v("#f-mode").value };
  if (n === 6) return { split_strategy: v("#f-split").value };
  return {};
}

function renderValidation(rep) {
  if (!rep) return "";
  return UI.html`<dl class="kv"><dt>전체 row</dt><dd>${rep.rows_total}</dd><dt>유효 row</dt><dd>${rep.rows_valid}</dd>
    <dt>결측 제거</dt><dd>${rep.rows_removed_missing}</dd><dt>잘못된 SMILES</dt><dd>${rep.rows_removed_invalid_smiles}</dd>
    <dt>중복 SMILES</dt><dd>${rep.duplicate_smiles}</dd><dt>추론 task</dt><dd>${rep.inferred_task_type}</dd></dl>
    ${UI.raw((rep.errors || []).map((e) => UI.html`<div class="alert err">${e}</div>`).join(""))}`;
}

// ---- cancel / delete helpers ----------------------------------------------------------------
async function cancelWorkflow(id) {
  const ok = await UI.choose({ title: `Workflow #${id} 학습 중단`, message: "진행 중인 단계가 끝나는 시점에 멈춥니다.\n중단한 workflow는 이어서 진행할 수 없습니다.",
    choices: [{ value: true, label: "학습 중단", cls: "danger" }] });
  if (!ok) return false;
  const r = await API.post(`/api/workflows/${id}/cancel`);
  UI.toast(r.status === "CANCELLING" ? "중단 요청함, 현재 단계가 끝나면 멈춤" : "중단됨");
  return true;
}

async function deleteWorkflow(w) {
  const id = w.workflow_id;
  const mvId = w.context && w.context.model_version_id;
  const base = "업로드한 데이터셋, 학습 기록, embedding이 함께 삭제되며 되돌릴 수 없습니다.";
  const choice = await UI.choose(mvId
    ? { title: `Workflow #${id} 삭제`, message: `${base}\n\n이 workflow로 만든 모델 #${mvId}을(를) 어떻게 할까요?\n모델을 지우면 그 모델로 한 예측 기록도 함께 지워집니다.`,
        choices: [{ value: "keep", label: "Workflow만 삭제 (모델 유지)", cls: "ghost" }, { value: "all", label: "모델까지 모두 삭제", cls: "danger" }] }
    : { title: `Workflow #${id} 삭제`, message: base, choices: [{ value: "keep", label: "삭제", cls: "danger" }] });
  if (!choice) return false;
  const r = await API.req("DELETE", `/api/workflows/${id}?delete_model=${choice === "all"}`);
  UI.toast(`Workflow #${id} 삭제함` + (r.deleted_model_versions ? `, 모델 ${r.deleted_model_versions}개 삭제` : r.kept_model_version_id ? `, 모델 #${r.kept_model_version_id}는 남김` : ""));
  return true;
}

async function deleteModelVersion(m) {
  const ok = await UI.choose({ title: `모델 ${m.label} 삭제`, message: "이 모델로 한 예측 기록도 함께 삭제되며 되돌릴 수 없습니다.",
    choices: [{ value: true, label: "모델 삭제", cls: "danger" }] });
  if (!ok) return false;
  await API.req("DELETE", `/api/models/versions/${m.model_version_id}`);
  UI.toast(`모델 ${m.label} 삭제함`);
  return true;
}

App.register("training", async (view) => {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const wfId = params.get("wf");
  if (wfId) return renderWorkflow(view, Number(wfId));

  const [{ workflows }, { jobs }, { model_versions }] = await Promise.all([
    API.get("/api/workflows"), API.get("/api/training/jobs"), API.get("/api/models")]);
  const mine = model_versions.filter((m) => m.scope === "USER");
  view.innerHTML = UI.html`
    <div class="page-head row"><div><h1>Custom 학습</h1><p class="muted">내 CSV 데이터로 ChemPrompt 모델(Model 1~5)을 학습합니다.</p></div>
      <span class="spacer"></span><button class="btn" id="new-wf">+ 새 학습 시작</button></div>
    <div class="grid">
      <div class="card"><h2>Workflows</h2>${UI.raw(workflows.length ? `<div class="table-wrap"><table><tr><th>#</th><th>상태</th><th>step</th><th>생성</th><th></th></tr>${
        workflows.map((w) => UI.html`<tr><td><a href="#/training?wf=${w.workflow_id}">#${w.workflow_id}</a></td><td>${UI.statusBadge(w.status)}</td><td>${w.current_step}/10</td><td>${UI.date(w.created_at)}</td>
          <td>${UI.raw(w.status === "RUNNING" ? `<button class="btn ghost sm" data-wf-cancel="${w.workflow_id}">중단</button>` : `<button class="btn ghost sm" data-wf-delete="${w.workflow_id}">삭제</button>`)}</td></tr>`).join("")}</table></div>` : '<div class="empty">없음</div>')}</div>
      <div class="card"><h2>내 모델</h2>${UI.raw(mine.length ? `<div class="table-wrap"><table><tr><th>모델</th><th>상태</th><th>설정</th><th>검증 성능 (Model 1~5 평균 ± 표준편차)</th><th></th></tr>${
        mine.map((m) => UI.html`<tr><td>${m.label}</td><td>${UI.statusBadge(m.status)}</td><td class="small">${m.descriptor_mode}<br>${m.split_strategy}</td><td class="small">${UI.raw(UI.metricsList(m.metrics, m.task_type))}</td>
          <td>${UI.raw(m.status === "SERVING" ? '<span class="muted small">서비스 중</span>' : `<button class="btn ghost sm" data-mv-delete="${m.model_version_id}">삭제</button>`)}</td></tr>`).join("")}</table></div>
        <p class="muted small">READY 모델은 본인이 바로 예측에 사용할 수 있습니다. 공개 serving은 관리자 승인이 필요합니다.</p>` : '<div class="empty">없음</div>')}</div>
    </div>
    <div class="card" style="margin-top:16px"><h2>Training jobs</h2>${UI.raw(jobs.length ? `<div class="table-wrap"><table><tr><th>#</th><th>상태</th><th>단계</th><th>모드</th><th>split</th><th>검증 성능 (Model 1~5 평균 ± 표준편차)</th></tr>${
      jobs.map((j) => UI.html`<tr><td>${j.training_job_id}</td><td>${UI.statusBadge(j.status)}</td><td class="small">${j.stage}</td><td>${j.descriptor_mode}</td><td>${j.config.split_strategy}</td><td class="small">${UI.raw(UI.metricsList(j.metrics, j.task_type))}</td></tr>`).join("")}</table></div>` : '<div class="empty">없음</div>')}</div>`;
  view.querySelector("#new-wf").onclick = async () => {
    const wf = await API.post("/api/workflows");
    location.hash = `#/training?wf=${wf.workflow_id}`;
  };
  const act = (fn) => async (e) => {
    e.target.disabled = true;
    try { if (await fn()) App.route(); } catch (err) { UI.toast(err.message, 6000); } finally { e.target.disabled = false; }
  };
  view.querySelectorAll("[data-wf-cancel]").forEach((b) => (b.onclick = act(() => cancelWorkflow(Number(b.dataset.wfCancel)))));
  view.querySelectorAll("[data-wf-delete]").forEach((b) => (b.onclick = act(() =>
    deleteWorkflow(workflows.find((w) => w.workflow_id === Number(b.dataset.wfDelete))))));
  view.querySelectorAll("[data-mv-delete]").forEach((b) => (b.onclick = act(() =>
    deleteModelVersion(mine.find((m) => m.model_version_id === Number(b.dataset.mvDelete))))));
});

async function renderWorkflow(view, id) {
  let es = null;
  const draw = async (wf) => {
    const next = wf.next_step;
    const nextDef = next ? wf.steps[next - 1] : null;
    const isUser = nextDef && nextDef.executor === "USER" && wf.status === "ACTIVE";
    let job = null;
    if (wf.context.training_job_id) job = await API.get(`/api/training/jobs/${wf.context.training_job_id}`);
    view.innerHTML = UI.html`
      <div class="page-head row"><div><a href="#/training" class="small">← 목록</a><h1>Workflow #${wf.workflow_id}</h1></div>
        <span class="spacer"></span>${UI.statusBadge(wf.status)}
        ${UI.raw(wf.status === "RUNNING" ? '<button class="btn danger sm" id="wf-cancel">학습 중단</button>' : '<button class="btn ghost sm" id="wf-delete">Workflow 삭제</button>')}</div>
      <div class="grid two">
        <div class="card"><h2>단계</h2><ol class="steps">${UI.raw(wf.steps.map((s) => UI.html`
          <li class="${s.status}"><span class="n">${s.status === "COMPLETED" ? "✓" : s.step_no}</span>${s.name}
          <span class="spacer"></span><span class="muted small">${s.executor === "SYSTEM" ? "자동" : ""}</span></li>`).join(""))}</ol></div>
        <div class="card" id="panel">
          ${UI.raw(isUser ? UI.html`<h2>Step ${next}. ${nextDef.name}</h2><form id="step-form">${UI.raw(STEP_FORMS[next](wf))}
            <div id="step-err"></div><div class="row" style="margin-top:14px"><button class="btn" type="submit">다음</button></div></form>` : "")}
          ${UI.raw(wf.status === "RUNNING" ? UI.html`<h2>학습 진행 중</h2><p><span class="spin"></span> <span id="live-stage">${job ? job.stage : ""}</span></p>
            <p class="muted small">진행 상태는 서버의 실제 작업 단계(DB 기록)를 그대로 표시합니다.</p>` : "")}
          ${UI.raw(wf.status === "FAILED" ? UI.html`<div class="alert err">${wf.error}</div>` : "")}
          ${UI.raw(wf.status === "CANCELLED" ? UI.html`<h2>중단됨</h2><div class="alert warn">${wf.error || "학습이 중단됨"}</div>
            <p class="muted small">이 workflow는 다시 이어서 진행할 수 없음, 필요하면 삭제한 뒤 새로 시작</p>` : "")}
          ${UI.raw(wf.status === "COMPLETED" ? UI.html`<h2>완료</h2><p>모델 버전 #${wf.context.model_version_id} 이(가) 등록되었습니다.</p>
            <a class="btn" href="#/predict">예측하러 가기</a>` : "")}
          ${UI.raw(wf.steps[2].result ? `<h3 style="margin-top:16px">Validation</h3>${renderValidation(wf.steps[2].result)}` : "")}
        </div>
      </div>
      ${UI.raw(job && job.folds && job.folds.length ? renderJobFolds(job) : "")}`;
    const cancelBtn = view.querySelector("#wf-cancel");
    if (cancelBtn) cancelBtn.onclick = async () => {
      cancelBtn.disabled = true;
      try {
        if (await cancelWorkflow(id)) { cancelBtn.textContent = "중단 요청됨…"; listen(); } else cancelBtn.disabled = false;
      } catch (err) { UI.toast(err.message, 6000); cancelBtn.disabled = false; }
    };
    const delBtn = view.querySelector("#wf-delete");
    if (delBtn) delBtn.onclick = async () => {
      delBtn.disabled = true;
      try { if (await deleteWorkflow(wf)) location.hash = "#/training"; else delBtn.disabled = false; }
      catch (err) { UI.toast(err.message, 6000); delBtn.disabled = false; }
    };
    const form = view.querySelector("#step-form");
    if (form) form.onsubmit = async (e) => {
      e.preventDefault();
      const btn = form.querySelector("button"); btn.disabled = true;
      try {
        const inp = await collectStepInput(next, view);
        const r = await API.post(`/api/workflows/${id}/steps/${next}`, inp);
        await draw(r); if (r.status === "RUNNING") listen();
      } catch (err) {
        const extra = err.data && err.data.report ? renderValidation(err.data.report) : "";
        view.querySelector("#step-err").innerHTML = UI.html`<div class="alert err">${err.message}</div>${UI.raw(extra)}`;
        btn.disabled = false;
      }
    };
  };
  const listen = () => {
    if (es) return;
    es = new AbortController();
    API.stream("GET", `/api/workflows/${id}/events`, null, async (ev, data) => {
      if (ev === "status") { const el = view.querySelector("#live-stage"); if (el) el.textContent = `${data.stage} (${data.job_status})`; }
      if (ev === "workflow_step" || ev === "done") draw(await API.get(`/api/workflows/${id}`));
    }, es.signal).catch(() => {}).finally(() => (es = null));
  };
  App.cleanup = () => es && es.abort();
  const wf = await API.get(`/api/workflows/${id}`);
  await draw(wf);
  if (wf.status === "RUNNING") listen();
}

function renderJobFolds(job) {
  const metricNames = Object.keys(job.folds[0].metrics);
  return UI.html`<div class="card" style="margin-top:16px"><h2>모델별 검증 결과</h2>
    <p class="muted small">${UI.metricNote(job.task_type)}</p>
    <div class="table-wrap"><table><tr><th>모델</th><th class="num">train</th><th class="num">valid</th><th class="num">descriptors</th>${
      UI.raw(metricNames.map((m) => `<th class="num">${UI.esc(m)}</th>`).join(""))}</tr>${UI.raw(job.folds.map((f) => UI.html`<tr><td>Model ${f.fold}</td>
      <td class="num">${f.n_train}</td><td class="num">${f.n_valid}</td><td class="num">${f.n_selected}</td>${
      UI.raw(metricNames.map((m) => `<td class="num">${UI.num(f.metrics[m])}</td>`).join(""))}</tr>`).join(""))}
      ${UI.raw(job.metrics ? `<tr><td colspan="4"><b>평균 ± std</b></td>${metricNames.map((m) => `<td class="num"><b>${UI.num(job.metrics[m].mean)}</b><br><span class="muted">±${UI.num(job.metrics[m].std)}</span></td>`).join("")}</tr>` : "")}
    </table></div></div>`;
}
