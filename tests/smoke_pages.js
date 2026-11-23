// 页面级渲染冒烟测试：用持久化 DOM 桩执行分析页/诊断页的完整渲染函数，
// 数据来自 tests/fixtures 的真实 API 响应，捕获字段缺失与引用错误。
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function makeNode() {
  const n = {
    style: {},
    classList: { add() {}, remove() {}, toggle() {} },
    dataset: {},
    attrs: {},
    children: [],
    _html: "",
    value: "",
    set innerHTML(v) { this._html = v; this.children = []; },
    get innerHTML() { return this._html; },
    set textContent(v) { this._t = v; },
    get textContent() { return this._t || ""; },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    insertAdjacentHTML() {},
    addEventListener() {},
    querySelector() { return makeNode(); },
    querySelectorAll() { return []; },
    onclick: null,
  };
  return n;
}
const listeners = {};
const document = {
  createElementNS: () => makeNode(),
  createElement: (tag) => {
    const n = makeNode();
    if (tag === "template") n.content = { firstElementChild: makeNode() };
    return n;
  },
  getElementById: () => makeNode(),
  querySelector: () => makeNode(),
  querySelectorAll: () => [],
  addEventListener(ev, fn) { listeners[ev] = fn; },
  body: makeNode(),
};
// 浏览器中 window 即全局对象：window.Charts 等价于裸标识符 Charts
const sandbox = {
  document, console, Math, location: { hash: "#/analysis" },
  fetch: async () => ({ json: async () => ({}) }),
  setTimeout, clearTimeout,
};
const window = sandbox;
window.window = window;
window.addEventListener = (ev, fn) => { listeners[ev] = fn; };
vm.createContext(sandbox);
const dir = path.join(__dirname, "../app/web/static");
const bundle = ["charts.js", "app.js", "datasets.js", "analysis.js", "diagnostics.js"]
  .map((f) => fs.readFileSync(path.join(dir, f), "utf8"))
  .join("\n;\n");
vm.runInContext(bundle, sandbox, { filename: "bundle.js" });

const fx = (f) => JSON.parse(
  fs.readFileSync(path.join(__dirname, "fixtures", f), "utf8")
);
let count = 0;
function check(name, fn) { fn(); count++; console.log("  ✓", name); }

const root = makeNode();
check("分析页渲染 场景A", () =>
  vm.runInContext("renderAnalysisResult", sandbox)(root, fx("analysis_a.json")));
check("分析页渲染 场景B", () =>
  vm.runInContext("renderAnalysisResult", sandbox)(root, fx("analysis_b.json")));
check("诊断页渲染 场景A", () =>
  vm.runInContext("renderDiagResult", sandbox)(root, fx("analysis_a.json"), "scenario_a"));
check("诊断页渲染 场景B", () =>
  vm.runInContext("renderDiagResult", sandbox)(root, fx("analysis_b.json"), "scenario_b"));
// 零列/近似列等边界：手工构造一个带 warning 的结果
const edge = JSON.parse(JSON.stringify(fx("analysis_a.json")));
edge.identifiability.warnings = ["测试告警：类别 X 缺席"];
edge.identifiability.condition_number = 2.5e7;
check("可识别性告警分支", () =>
  vm.runInContext("renderAnalysisResult", sandbox)(root, edge));

console.log(`\n全部 ${count} 个页面级渲染场景通过（真实 API 数据驱动）`);
