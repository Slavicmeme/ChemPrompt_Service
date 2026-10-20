const App = {
  user: null,
  routes: {},
  cleanup: null,
  register(name, render) { App.routes[name] = render; },
  async boot() {
    try { App.user = (await API.get("/api/auth/me")).user; } catch (_) { App.user = null; }
    window.addEventListener("hashchange", App.route);
    document.getElementById("logout-btn").onclick = async () => {
      await API.post("/api/auth/logout"); App.user = null; location.hash = "#/login";
    };
    App.route();
  },
  async route() {
    const name = (location.hash.replace(/^#\//, "").split("?")[0] || "predict");
    if (App.cleanup) { try { App.cleanup(); } catch (_) {} App.cleanup = null; }
    const authed = !!App.user;
    document.getElementById("topbar").hidden = !authed;
    if (!authed && name !== "login") { location.hash = "#/login"; return; }
    if (authed) {
      document.getElementById("user-name").textContent = `${App.user.username}${App.user.role === "ADMIN" ? " (admin)" : ""}`;
      document.getElementById("nav-admin").hidden = App.user.role !== "ADMIN";
      document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === name));
    }
    const render = App.routes[name] || App.routes.predict;
    const view = UI.view();
    view.innerHTML = `<div class="empty"><span class="spin"></span></div>`;
    try { await render(view); } catch (e) { view.innerHTML = UI.html`<div class="alert err">${e.message}</div>`; }
  },
};
document.addEventListener("DOMContentLoaded", App.boot);
