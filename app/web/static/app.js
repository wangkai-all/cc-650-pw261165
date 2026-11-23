// 路由、API 封装与全局状态。所有页面为纯前端渲染，数据来自 /api。

const Api = {
  async req(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `请求失败 ${res.status}`);
    return data;
  },
  get(p) { return this.req(p); },
  post(p, body) { return this.req(p, { method: "POST", body }); },
  del(p) { return this.req(p, { method: "DELETE" }); },
};

const State = { datasets: [], current: null };

function toast(msg, bad = false) {
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  t.style.background = bad ? "var(--critical)" : "var(--ink)";
  t.classList.add("show");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove("show"), 3200);
}

function pct(v, d = 1) { return (v * 100).toFixed(d) + "%"; }
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

async function loadDatasets() {
  State.datasets = (await Api.get("/api/datasets")).datasets;
  return State.datasets;
}

const routes = {
  "#/datasets": renderDatasetsPage,
  "#/analysis": renderAnalysisPage,
  "#/diagnostics": renderDiagnosticsPage,
};

async function router() {
  const hash = location.hash || "#/datasets";
  document.querySelectorAll("nav.tabs a").forEach((a) =>
    a.classList.toggle("active", a.getAttribute("href") === hash));
  const fn = routes[hash] || renderDatasetsPage;
  const main = document.getElementById("view");
  main.innerHTML = `<div class="empty">加载中…</div>`;
  try {
    await fn(main);
  } catch (e) {
    main.innerHTML = `<div class="alert bad">页面加载失败：${esc(e.message)}</div>`;
  }
}
window.addEventListener("hashchange", router);
window.addEventListener("DOMContentLoaded", router);
