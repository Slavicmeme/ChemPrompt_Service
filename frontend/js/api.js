// Thin API client. Auth uses the HttpOnly session cookie; no secrets are stored in the browser.
const API = {
  async req(method, path, body, isForm) {
    const opt = { method, credentials: "same-origin", headers: {} };
    if (body !== undefined) {
      if (isForm) opt.body = body;
      else { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
    }
    const r = await fetch(path, opt);
    let data = null;
    try { data = await r.json(); } catch (_) { /* empty */ }
    if (!r.ok) {
      const err = new Error((data && (data.detail || data.error)) || `HTTP ${r.status}`);
      err.status = r.status; err.data = data;
      if (r.status === 401 && !path.startsWith("/api/auth")) { App.user = null; location.hash = "#/login"; }
      throw err;
    }
    return data;
  },
  get: (p) => API.req("GET", p),
  post: (p, b) => API.req("POST", p, b === undefined ? {} : b),
  form: (p, fd) => API.req("POST", p, fd, true),

  // Server-Sent Events over fetch (works for POST bodies and cookies).
  async stream(method, path, body, onEvent, signal) {
    const r = await fetch(path, {
      method, credentials: "same-origin", signal,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
        let ev = "message", data = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) ev = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        let parsed = data;
        try { parsed = JSON.parse(data); } catch (_) {}
        onEvent(ev, parsed);
      }
    }
  },
};
