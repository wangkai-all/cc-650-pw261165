// 诊断页：两组虚构流量 A（仅比例漂移）/ B（比例+类内漂移），
// 展示边际拟合检验的饱和无功效事实，以及带真值时转移矩阵两样本检验的识别能力。

async function renderDiagnosticsPage(main) {
  await loadDatasets();
  const synthetic = State.datasets.filter((d) => d.kind === "synthetic");
  main.innerHTML = `
    <div class="card">
      <h2>虚构流量拟合诊断</h2>
      <div class="alert info">
        本页使用带真值的虚构数据，区分两类变化：
        <strong>场景 A</strong> 仅改变类别比例、类内误判方式不变；
        <strong>场景 B</strong> 同时改变类内误判方式。页面据此检验能否识别"拟合明显不符"。
      </div>
      <div class="alert warn">
        <strong>关键统计事实：</strong>当转移矩阵为方阵且满秩、真实占比为自由参数时，
        反推模型对边际预测分布是饱和拟合（自由度 0），仅靠新批次的预测标签
        <em>无法</em>识别类内漂移——"看不出异常"绝不等于类内稳定假设已被证明。
        要直接验证类内稳定性，需要带真值的复核子样本做转移矩阵两样本检验。
      </div>
      <label class="field"><span>虚构数据集</span><select id="d-ds">
        ${synthetic.map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join("")}
      </select></label>
      <div class="row">
        <button id="d-run-a">分析场景 A（仅比例漂移）</button>
        <button class="secondary" id="d-run-b">分析场景 B（比例+类内漂移）</button>
      </div>
    </div>
    <div id="d-result"></div>`;

  if (!synthetic.length) {
    main.querySelector("#d-result").innerHTML =
      `<div class="card empty">没有虚构数据集，请到「校验集管理」页点击"生成内置虚构演示"。</div>`;
    return;
  }

  main.querySelector("#d-run-a").onclick = () => runDiag(main, "scenario_a");
  main.querySelector("#d-run-b").onclick = () => runDiag(main, "scenario_b");
}

async function runDiag(main, scenario) {
  const dsId = main.querySelector("#d-ds").value;
  const el = main.querySelector("#d-result");
  el.innerHTML = `<div class="card empty spin">分析中…</div>`;
  try {
    const r = await Api.post(`/api/datasets/${dsId}/analyze`, {
      scenario, n_bootstrap: 300, n_null: 400,
    });
    renderDiagResult(el, r, scenario);
  } catch (e) {
    el.innerHTML = `<div class="alert bad">分析失败：${esc(e.message)}</div>`;
  }
}

function renderDiagResult(el, r, scenario) {
  const meta = r.scenario_meta;
  const mt = r.marginal_test;
  const tt = r.transition_test;
  const labels = r.labels;

  const marginalBadge = mt.saturated_no_power
    ? `<span class="badge neutral">饱和模型 · 无检验功效</span>`
    : (mt.p_value_t2 < 0.05 ? `<span class="badge bad">检出不符</span>` : `<span class="badge good">未检出异常</span>`);

  const transBadge = tt.misfit_flagged
    ? `<span class="badge bad">检出类内漂移</span>`
    : `<span class="badge good">未发现漂移</span>`;

  const expected = meta.class_behavior_changed
    ? `<div class="alert bad">该场景<strong>人为改变了类内误判方式</strong>，期望诊断能识别明显不符。</div>`
    : `<div class="alert info">该场景<strong>类内行为保持不变</strong>，仅比例漂移，期望不误报。</div>`;

  el.innerHTML = `
    <div class="card">
      <h2>${esc(meta.name)} ${transBadge}</h2>
      <p class="muted small">${esc(meta.description)}</p>
      ${expected}
      <div class="grid3">
        <div class="stat"><div class="k">真实占比（虚构真值，仅供对照）</div>
          <div class="v" style="font-size:14px">${meta.true_proportions.map((v) => pct(v)).join(" / ")}</div></div>
        <div class="stat"><div class="k">反推真实占比</div>
          <div class="v" style="font-size:14px">${r.estimate.p.map((v) => pct(v)).join(" / ")}</div></div>
        <div class="stat"><div class="k">批次样本量</div><div class="v">${r.batch.size}</div></div>
      </div>
    </div>

    <div class="card">
      <h2>① 边际拟合优度检验（仅预测标签，业务真实可得）${marginalBadge}</h2>
      <p class="muted small">统计量 T² = ${mt.statistic_t2.toExponential(2)}，自助 p = ${mt.p_value_t2.toFixed(3)}。</p>
      <div class="alert ${mt.saturated_no_power ? "warn" : "info"}">${esc(mt.fitted_residual_note)}</div>
      <div class="alert info small">${esc(mt.interpretation_note)}</div>
    </div>

    <div class="card">
      <h2>② 转移矩阵两样本检验（需要新批次真值，仅诊断/复核可用）${transBadge}</h2>
      <p class="muted small">逐列比较校验矩阵与新批次矩阵的总变差距离 TV，零分布由参数自助（匹配两批样本量）产生，超过 95% 分位标记漂移。</p>
      <div class="scroll"><table>
        <thead><tr><th>真实类别（列）</th><th>观测列 TV</th><th>零分布 95% 分位</th><th>零分布 99% 分位</th><th>自助 p</th><th>判定</th></tr></thead>
        <tbody>${tt.columns.map((c) => `<tr>
          <td>${esc(c.label)}</td>
          <td><strong>${c.observed_tv.toFixed(3)}</strong></td>
          <td>${c.null_tv_q95.toFixed(3)}</td>
          <td>${c.null_tv_q99.toFixed(3)}</td>
          <td>${c.p_value.toFixed(3)}</td>
          <td>${c.strongly_drifted
            ? `<span class="badge bad">显著漂移</span>`
            : c.drifted ? `<span class="badge warn">漂移</span>`
            : `<span class="badge good">正常</span>`}</td>
        </tr>`).join("")}</tbody></table></div>
      <div id="tv-chart" style="margin-top:16px"></div>
      <div class="alert info small" style="margin-top:12px">${esc(tt.availability_note)}</div>
      <div class="alert warn small">${esc(tt.interpretation_note)}</div>
    </div>

    <div class="card">
      <h2>反推占比与自助区间</h2>
      <div id="d-interval"></div>
    </div>`;

  // 列 TV 对比条形图（观测 vs 95% 阈值）
  renderTvChart(el.querySelector("#tv-chart"), tt);
  Charts.intervalPlot(
    el.querySelector("#d-interval"), labels,
    r.bootstrap.median, r.bootstrap.intervals["95"], r.bootstrap.intervals["90"]
  );
}

function renderTvChart(container, tt) {
  const labels = tt.columns.map((c) => c.label);
  const W = 620, H = 280, m = { l: 90, r: 20, t: 16, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "chart");
  const maxV = Math.max(0.05, ...tt.columns.flatMap((c) => [c.observed_tv, c.null_tv_q95])) * 1.15;
  const x = (v) => m.l + (v / maxV) * iw;
  const rowH = ih / labels.length;
  const barH = 13;

  for (let g = 0; g <= 4; g++) {
    const v = (maxV * g) / 4, xx = x(v);
    svg.appendChild(line(xx, m.t, xx, H - m.b, Charts.PALETTE.grid));
    svg.appendChild(text(xx, H - 14, pct(v, 0), Charts.PALETTE.muted, 10, "middle"));
  }
  labels.forEach((lab, i) => {
    const c = tt.columns[i];
    const yy = m.t + i * rowH + rowH / 2;
    svg.appendChild(text(m.l - 10, yy + 4, lab, Charts.PALETTE.ink2, 12, "end"));
    // 95% 阈值条（橙）
    svg.appendChild(rect(x(0), yy - barH - 1, x(c.null_tv_q95) - x(0), barH, Charts.PALETTE.series[1], 4));
    // 观测 TV（蓝/红）
    const fill = c.drifted ? Charts.PALETTE.critical : Charts.PALETTE.series[0];
    svg.appendChild(rect(x(0), yy + 1, x(c.observed_tv) - x(0), barH, fill, 4));
  });
  // 图例
  svg.appendChild(rect(m.l, 2, 12, 12, Charts.PALETTE.series[1], 3));
  svg.appendChild(text(m.l + 18, 12, "零分布 95% 分位阈值", Charts.PALETTE.ink2, 11));
  svg.appendChild(rect(m.l + 180, 2, 12, 12, Charts.PALETTE.series[0], 3));
  svg.appendChild(text(m.l + 198, 12, "观测列 TV（红色=漂移）", Charts.PALETTE.ink2, 11));
  container.innerHTML = "";
  container.appendChild(svg);
}

function line(x1, y1, x2, y2, stroke) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", "line");
  n.setAttribute("x1", x1); n.setAttribute("x2", x2);
  n.setAttribute("y1", y1); n.setAttribute("y2", y2); n.setAttribute("stroke", stroke);
  return n;
}
function rect(x, y, w, h, fill, rx = 0) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  n.setAttribute("x", x); n.setAttribute("y", y); n.setAttribute("width", Math.max(0, w));
  n.setAttribute("height", h); n.setAttribute("fill", fill); n.setAttribute("rx", rx);
  return n;
}
function text(x, y, s, fill, size, anchor = "start") {
  const n = document.createElementNS("http://www.w3.org/2000/svg", "text");
  n.setAttribute("x", x); n.setAttribute("y", y); n.setAttribute("fill", fill);
  n.setAttribute("font-size", size); n.setAttribute("text-anchor", anchor);
  n.textContent = s;
  return n;
}
