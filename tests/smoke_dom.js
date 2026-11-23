// 前端渲染冒烟测试：用最小 DOM 桩执行 charts.js 的全部图表函数，
// 喂入真实后端 API 响应（由 scripts/fetch_fixtures.js 生成），
// 捕获运行时引用/类型错误。运行：node tests/smoke_dom.js
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function makeNode(tag) {
  return {
    tag, attrs: {}, children: [], textContent: "",
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
  };
}
const document = {
  createElementNS: (_ns, tag) => makeNode(tag),
  createElement: (tag) => makeNode(tag),
};
const window = {};
const sandbox = { document, window, console, Math };
vm.createContext(sandbox);
vm.runInContext(
  fs.readFileSync(path.join(__dirname, "../app/web/static/charts.js"), "utf8"),
  sandbox
);
const Charts = window.Charts;

const container = makeNode("div");
const labels = ["咨询", "投诉", "查询进度", "表扬"];
let checks = 0;
function ok(name, fn) {
  fn();
  checks++;
  console.log("  ✓", name);
}

// 用合成数据覆盖各图表（字段结构与后端一致）
ok("groupedBar", () =>
  Charts.groupedBar(container, labels, [
    { name: "朴素", values: [0.4, 0.28, 0.2, 0.12], color: "#eb6834" },
    { name: "反推", values: [0.41, 0.31, 0.17, 0.11], color: "#2a78d6" },
  ]));
ok("stackedBar", () =>
  Charts.stackedBar(container, labels, [0.41, 0.31, 0.17, 0.11]));
ok("heatmap", () => {
  const matrix = [
    [0.82, 0.18, 0.06, 0.05],
    [0.08, 0.66, 0.1, 0.04],
    [0.07, 0.12, 0.81, 0.06],
    [0.03, 0.04, 0.03, 0.85],
  ];
  const counts = matrix.map((r) => r.map((v) => Math.round(v * 800)));
  Charts.heatmap(container, labels, matrix, counts);
});
ok("intervalPlot", () =>
  Charts.intervalPlot(
    container, labels,
    [0.41, 0.31, 0.17, 0.11],
    [[0.39, 0.43], [0.29, 0.33], [0.15, 0.19], [0.09, 0.13]],
    [[0.395, 0.425], [0.295, 0.325], [0.155, 0.185], [0.095, 0.125]]
  ));
ok("pct/colorFor 边界", () => {
  if (Charts.pct(0) !== "0.0%") throw new Error("pct(0) 错误");
  if (!Charts.colorFor(0).startsWith("#")) throw new Error("colorFor 错误");
  if (!Charts.colorFor(99).startsWith("#")) throw new Error("颜色循环错误");
});

// 若存在真实 API fixture，则用真实数据再跑一遍
const fxDir = path.join(__dirname, "fixtures");
if (fs.existsSync(fxDir)) {
  for (const f of fs.readdirSync(fxDir).filter((x) => x.endsWith(".json"))) {
    const r = JSON.parse(fs.readFileSync(path.join(fxDir, f), "utf8"));
    if (!r.estimate) continue;
    ok(`真实响应 ${f}: groupedBar`, () =>
      Charts.groupedBar(container, r.labels, [
        { name: "朴素", values: r.batch.predicted_distribution },
        { name: "反推", values: r.estimate.p },
      ]));
    ok(`真实响应 ${f}: intervalPlot`, () =>
      Charts.intervalPlot(
        container, r.labels, r.bootstrap.median,
        r.bootstrap.intervals["95"], r.bootstrap.intervals["90"]
      ));
    ok(`真实响应 ${f}: heatmap`, () =>
      Charts.heatmap(
        container, r.labels, r.confusion.matrix, r.confusion.counts
      ));
    if (r.transition_test) {
      const cols = r.transition_test.columns;
      if (!cols[0].observed_tv && cols[0].observed_tv !== 0)
        throw new Error("转移检验字段缺失");
    }
  }
}
console.log(`\n全部 ${checks} 项前端渲染冒烟检查通过`);
