// 轻量 SVG 图表库：分组条形图、堆叠条、混淆热力图、区间点图。
// 颜色遵循 dataviz 调色板，类别按固定顺序着色，绝不按排名重排。

const PALETTE = {
  series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"],
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  baseline: "#c3c2b7",
  good: "#0ca30c",
  warning: "#fab219",
  critical: "#d03b3b",
  seq: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
};

function colorFor(i) {
  return PALETTE.series[i % PALETTE.series.length];
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c != null) node.appendChild(c);
  }
  return node;
}

function pct(v) {
  return (v * 100).toFixed(1) + "%";
}

// 分组条形图：每个类别多系列（朴素预测占比 / 反推真实占比）
function groupedBar(container, labels, series) {
  const W = Math.max(560, labels.length * 120), H = 320;
  const m = { t: 24, r: 16, b: 64, l: 48 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "chart" });
  const maxV = Math.max(0.001, ...series.flatMap((s) => s.values)) * 1.15;
  const y = (v) => m.t + ih - (v / maxV) * ih;
  const groupW = iw / labels.length;
  const barW = (groupW * 0.7) / series.length;

  // 网格 + y 轴刻度
  for (let g = 0; g <= 4; g++) {
    const val = (maxV * g) / 4;
    const yy = y(val);
    svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, stroke: PALETTE.grid, "stroke-width": 1 }));
    svg.appendChild(el("text", { x: m.l - 6, y: yy + 4, "text-anchor": "end", fill: PALETTE.muted, "font-size": 11, text: pct(val) }));
  }
  svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: PALETTE.baseline }));

  labels.forEach((lab, i) => {
    const gx = m.l + groupW * i + groupW * 0.15;
    series.forEach((s, si) => {
      const v = s.values[i];
      const x = gx + si * barW;
      const yy = y(v);
      const rect = el("rect", {
        x, y: yy, width: barW - 2, height: m.t + ih - yy,
        rx: 4, fill: s.color || colorFor(si), class: "bar",
      });
      const tip = el("title", { text: `${lab} · ${s.name}: ${pct(v)}` });
      rect.appendChild(tip);
      svg.appendChild(rect);
    });
    const lbl = el("text", {
      x: gx + (groupW * 0.7) / 2, y: m.t + ih + 18, "text-anchor": "middle",
      fill: PALETTE.ink2, "font-size": 12, text: lab,
    });
    svg.appendChild(lbl);
  });

  // 图例
  const legend = el("g", {});
  series.forEach((s, si) => {
    const lx = m.l + si * 150;
    legend.appendChild(el("rect", { x: lx, y: 4, width: 12, height: 12, rx: 3, fill: s.color || colorFor(si) }));
    legend.appendChild(el("text", { x: lx + 18, y: 14, fill: PALETTE.ink2, "font-size": 12, text: s.name }));
  });
  svg.appendChild(legend);
  container.innerHTML = "";
  container.appendChild(svg);
}

// 100% 堆叠条：展示一个占比向量（带可选区间）
function stackedBar(container, labels, values, intervals) {
  const W = 560, H = 90;
  const m = { t: 30, r: 8, b: 34, l: 8 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "chart" });
  const iw = W - m.l - m.r;
  let x = m.l;
  labels.forEach((lab, i) => {
    const w = values[i] * iw;
    const rect = el("rect", {
      x: x + 1, y: m.t, width: Math.max(0, w - 2), height: 26,
      fill: colorFor(i), rx: 3,
    });
    rect.appendChild(el("title", { text: `${lab}: ${pct(values[i])}` }));
    svg.appendChild(rect);
    if (w > 34) {
      svg.appendChild(el("text", {
        x: x + w / 2, y: m.t + 17, "text-anchor": "middle",
        fill: "#fff", "font-size": 12, "font-weight": 600, text: pct(values[i]),
      }));
    }
    // 图例
    svg.appendChild(el("rect", { x: m.l + i * (iw / labels.length), y: m.t + 38, width: 10, height: 10, rx: 2, fill: colorFor(i) }));
    svg.appendChild(el("text", {
      x: m.l + i * (iw / labels.length) + 14, y: m.t + 47,
      fill: PALETTE.ink2, "font-size": 11, text: lab,
    }));
    x += w;
  });
  container.innerHTML = "";
  container.appendChild(svg);
}

// 混淆矩阵热力图（行=预测，列=真实）
function heatmap(container, labels, matrix, counts) {
  const cell = 78, header = 88;
  const n = labels.length;
  const W = header + n * cell + 20, H = header + n * cell + 20;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "chart heatmap" });

  labels.forEach((lab, j) => {
    svg.appendChild(el("text", { x: header + j * cell + cell / 2, y: header - 12, "text-anchor": "middle", fill: PALETTE.ink2, "font-size": 12, text: "真:" + lab }));
    svg.appendChild(el("text", { x: header - 10, y: header + j * cell + cell / 2 + 4, "text-anchor": "end", fill: PALETTE.ink2, "font-size": 12, text: "预:" + labels[j] }));
  });

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = matrix[i][j];
      const shade = Math.min(1, v);
      const fill = seqBlue(shade);
      const cx = header + j * cell, cy = header + i * cell;
      const rect = el("rect", { x: cx + 2, y: cy + 2, width: cell - 4, height: cell - 4, rx: 6, fill });
      rect.appendChild(el("title", {
        text: `真 ${labels[j]} → 预 ${labels[i]}: ${pct(v)}（${counts ? counts[i][j] : ""} 条）`,
      }));
      svg.appendChild(rect);
      const dark = shade > 0.45;
      svg.appendChild(el("text", {
        x: cx + cell / 2, y: cy + cell / 2 - 2, "text-anchor": "middle",
        fill: dark ? "#fff" : PALETTE.ink, "font-size": 14, "font-weight": 600, text: pct(v),
      }));
      if (counts) {
        svg.appendChild(el("text", {
          x: cx + cell / 2, y: cy + cell / 2 + 16, "text-anchor": "middle",
          fill: dark ? "rgba(255,255,255,0.85)" : PALETTE.muted, "font-size": 11, text: `${counts[i][j]}`,
        }));
      }
    }
  }
  container.innerHTML = "";
  container.appendChild(svg);
}

function seqBlue(t) {
  // 在浅蓝→深蓝 7 档间插值
  const stops = PALETTE.seq;
  const x = Math.max(0, Math.min(0.999, t)) * (stops.length - 1);
  const lo = Math.floor(x), frac = x - lo;
  const a = hexRgb(stops[lo]), b = hexRgb(stops[Math.min(lo + 1, stops.length - 1)]);
  const mix = a.map((v, k) => Math.round(v + (b[k] - v) * frac));
  return `rgb(${mix.join(",")})`;
}

function hexRgb(hex) {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((o) => parseInt(h.slice(o, o + 2), 16));
}

// 区间点图：点估计 + 90/95 自助区间
function intervalPlot(container, labels, points, ci95, ci90) {
  const rowH = 52, m = { l: 96, r: 24, t: 12, b: 24 };
  const W = 620, H = m.t + m.b + labels.length * rowH;
  const iw = W - m.l - m.r;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "chart" });
  const maxV = Math.max(0.05, ...ci95.flat()) * 1.1;
  const x = (v) => m.l + (v / maxV) * iw;

  for (let g = 0; g <= 4; g++) {
    const val = (maxV * g) / 4, xx = x(val);
    svg.appendChild(el("line", { x1: xx, x2: xx, y1: m.t, y2: H - m.b, stroke: PALETTE.grid }));
    svg.appendChild(el("text", { x: xx, y: H - 6, "text-anchor": "middle", fill: PALETTE.muted, "font-size": 10, text: pct(val) }));
  }

  labels.forEach((lab, i) => {
    const yy = m.t + i * rowH + rowH / 2;
    svg.appendChild(el("text", { x: m.l - 10, y: yy + 4, "text-anchor": "end", fill: PALETTE.ink2, "font-size": 12, text: lab }));
    // 95 外区间（浅色粗），90 内区间（深色细）
    svg.appendChild(el("line", { x1: x(ci95[i][0]), x2: x(ci95[i][1]), y1: yy, y2: yy, stroke: colorFor(i), "stroke-width": 5, opacity: 0.3, "stroke-linecap": "round" }));
    svg.appendChild(el("line", { x1: x(ci90[i][0]), x2: x(ci90[i][1]), y1: yy, y2: yy, stroke: colorFor(i), "stroke-width": 2.5, "stroke-linecap": "round" }));
    svg.appendChild(el("circle", { cx: x(points[i]), cy: yy, r: 5.5, fill: colorFor(i), stroke: "#fff", "stroke-width": 1.5 }));
    svg.appendChild(el("text", { x: x(ci95[i][1]) + 6, y: yy + 4, fill: PALETTE.muted, "font-size": 11, text: `${pct(points[i])} [${pct(ci95[i][0])}, ${pct(ci95[i][1])}]` }));
  });
  container.innerHTML = "";
  container.appendChild(svg);
}

window.Charts = { groupedBar, stackedBar, heatmap, intervalPlot, pct, colorFor, PALETTE };
