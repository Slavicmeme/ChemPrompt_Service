App.register("login", async (view) => {
  if (App.user) { location.hash = "#/predict"; return; }
  let mode = "login";
  view.innerHTML = `
    <div class="auth-wrap">
      <div class="page-head"><h1>⌬ ChemPrompt</h1>
        <p class="muted">LLM 기반 분자 표현(ChemPrompt)과 5-fold ensemble 모델로 분자 물성을 예측합니다.</p></div>
      <div class="card">
        <div class="tabs"><button data-m="login" class="active">로그인</button><button data-m="register">회원가입</button></div>
        <form id="auth-form">
          <label for="u">Username</label><input id="u" autocomplete="username" required>
          <label for="p">Password</label><input id="p" type="password" autocomplete="current-password" required minlength="8">
          <div id="auth-err"></div>
          <div class="row" style="margin-top:14px"><button class="btn" type="submit" id="auth-submit">로그인</button></div>
        </form>
      </div>
    </div>`;
  view.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => {
    mode = b.dataset.m;
    view.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
    view.querySelector("#auth-submit").textContent = mode === "login" ? "로그인" : "가입하기";
  }));
  view.querySelector("#auth-form").onsubmit = async (e) => {
    e.preventDefault();
    const body = { username: view.querySelector("#u").value.trim(), password: view.querySelector("#p").value };
    try {
      const r = await API.post(`/api/auth/${mode}`, body);
      App.user = r.user; location.hash = "#/predict";
    } catch (err) {
      view.querySelector("#auth-err").innerHTML = UI.html`<div class="alert err">${err.message}</div>`;
    }
  };
});
