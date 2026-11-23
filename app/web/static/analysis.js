// 占比反推分析页：直接数预测标签 vs 约束反推真实占比，自助区间，正向推回差距。

let analysisCache = null;

async function renderAnalysisPage(main) {
  await loadDatasets();
  main.innerHTML = `
    <div class="card">
      <h2>选择校验集与新批次</h2>
      <div class="grid2">
        <label class="field"><span>校验集</span><select id="a-ds"></select></label>
        <label class="field"><span>合并视图（可选，默认用原始类别）</span><select id="a-merge"><option value="">原始类别（不合并）</option></select></label>
      </div>
      <label class="field"><span>新批次预测类别计数（JSON，仅预测、无真值）</span>
        <textarea id="a-batch" placeholder='{"咨询": 1200, "投诉": 540, "查询进度": 480, "表扬": 180}'></textarea></label>
      <div class="row">
        <button id="a-run">反推真实占比</button>
        <span class="muted small">自助重抽 <input id="a-boot" value="300" style="width:70px;display:inline-block"> 次</span>
      </div>
    </div>
    <div id="a-result"></div>`;

  const dsSel = main.querySelector("#a-ds");
  dsSel.innerHTML = State.datasets.map((d) =>
    `<option value="${d.id}">${esc(d.name)}（${d.n_rows}）</option>`).join("")
    || `<option value="">（请先导入校验集）</option>`;

  async function refreshMerges() {
    const dsId = dsSel.value;
    if (!dsId) return;
    const { merges } = await Api.get(`/api/datasets/${dsId}/merges`);
    main.querySelector("#a-merge").innerHTML =
      `<option value="">原始类别（不合并）</option>` +
      merges.map((m) => `<option value="${m.id}">${esc(m.name)}</option>`).join("");
  }
  dsSel.onchange = refreshMerges;
  if (dsSel.value) refreshMerges();

  main.querySelector("#a-run").onclick = () => runAnalysis(main);
}

async function runAnalysis(main) {
  const dsId = main.querySelector("#a-ds").value;
  const mergeId = main.querySelector("#a-merge").value;
  const nBoot = parseInt(main.querySelector("#a-boot").value, 10) || 300;
  let predCounts;
  try {
    predCounts = JSON.parse(main.querySelector("#a-batch").value);
  } catch (e) { toast("批次 JSON 解析失败：" + e.message, true); return; }

  const url = mergeId
    ? `/api/merges/${mergeId}/analyze`
    : `/api/datasets/${dsId}/analyze`;
  const resultEl = main.querySelector("#a-result");
  resultEl.innerHTML = `<div class="card empty spin">正在求解约束二次规划与自助重采样…</div>`;
  try {
    const r = await Api.post(url, { pred_counts: predCounts, n_bootstrap: nBoot, n_null: nBoot });
    analysisCache = r;
    renderAnalysisResult(resultEl, r);
  } catch (e) {
    resultEl.innerHTML = `<div class="alert bad">分析失败：${esc(e.message)}</div>`;
  }
}

function renderAnalysisResult(el, r) {
  const labels = r.labels;
  const naive = r.batch.predicted_distribution;
  const p = r.estimate.p;
  const boot = r.bootstrap;
  const ident = r.identifiability;

  const identAlerts = ident.warnings.length
    ? ident.warnings.map((w) => `<div class="alert warn">${esc(w)}</div>`).join("")
    : `<div class="alert info">可识别性正常：无缺席类别、无近似重复列。</div>`;

  el.innerHTML = `
    <div class="card">
      <h2>求解质量与约束保证</h2>
      <div class="grid3">
        <div class="stat"><div class="k">估计占比之和</div><div class="v">${r.estimate.p.reduce((a, b) => a + b, 0).toFixed(6)}</div></div>
        <div class="stat"><div class="k">最小占比（非负）</div><div class="v">${Math.min(...r.estimate.p).toExponential(1)}</div></div>
        <div class="stat"><div class="k">求解方法</div><div class="v" style="font-size:16px">${esc(r.estimate.solver.method)}</div></div>
      </div>
      <p class="muted small" style="margin-top:10px">
        反推通过单纯形约束二次规划（SLSQP，投影梯度兜底）直接在 <span class="kbd">p≥0、Σp=1</span> 上求解，
        代码路径不含任何矩阵求逆/伪逆，因此从不需要"先求逆再截断"。约束满足：
        <span class="badge ${r.estimate.constraints_ok ? "good" : "bad"}">${r.estimate.constraints_ok ? "通过" : "异常"}</span>
      </p>
    </div>

    <div class="card">
      <h2>可识别性</h2>${identAlerts}
      <div class="small muted">有效秩 ${ident.effective_rank}/${ident.active_columns}
      ${ident.condition_number ? "｜条件数 " + ident.condition_number : ""}</div>
    </div>

    <div class="card">
      <h2>直接数预测标签 vs 反推真实占比</h2>
      <div id="bar-chart"></div>
      <div class="scroll" style="margin-top:14px"><table>
        <thead><tr><th>类别</th><th>朴素：预测标签占比</th><th>反推：真实占比估计</th><th>修正量</th><th>校验样本数</th></tr></thead>
        <tbody>${r.per_class.map((c, i) => `<tr>
          <td><span class="dot" style="background:${Charts.colorFor(i)}"></span> ${esc(c.label)}</td>
          <td>${pct(c.naive_predicted_share)}</td>
          <td><strong>${pct(c.estimated_true_share)}</strong></td>
          <td class="${c.correction >= 0 ? "" : ""}">${c.correction >= 0 ? "+" : ""}${pct(c.correction, 2)}</td>
          <td>${c.validation_support}</td></tr>`).join("")}
        </tbody></table></div>
    </div>

    <div class="card">
      <h2>反推真实占比的自助稳定区间</h2>
      <p class="muted small">按校验样本逐行重采样重建转移矩阵后重解，共 ${boot.n_valid} 次有效重抽
      （零列跳过 ${boot.skipped_zero_column} 次）。粗线为 90% 区间，浅色为 95% 区间。</p>
      <div id="interval-chart"></div>
      <div id="stack-naive" style="margin-top:18px"></div>
      <div id="stack-est" style="margin-top:8px"></div>
    </div>

    <div class="card">
      <h2>正向推回：估计占比能否还原新批次预测分布？</h2>
      <p class="muted small">把反推的真实占比经校验矩阵正向推回得到 X·p̂，与新批次实际观测的预测分布对照。</p>
      <div class="grid3">
        <div class="stat"><div class="k">总变差距离 TV</div><div class="v">${pct(r.forward_check.total_variation_distance, 2)}</div></div>
        <div class="stat"><div class="k">RMSE</div><div class="v">${r.forward_check.rmse.toExponential(1)}</div></div>
        <div class="stat"><div class="k">最大单类差距</div><div class="v">${pct(Math.max(...r.forward_check.residual.map(Math.abs)), 2)}</div></div>
      </div>
      <div class="scroll" style="margin-top:12px"><table>
        <thead><tr><th>类别</th><th>观测预测占比 q</th><th>正向推回 X·p̂</th><th>差距</th></tr></thead>
        <tbody>${labels.map((l, i) => `<tr>
          <td>${esc(l)}</td><td>${pct(r.forward_check.q_observed[i])}</td>
          <td>${pct(r.forward_check.q_forward_from_estimate[i])}</td>
          <td>${pct(r.forward_check.residual[i], 2)}</td></tr>`).join("")}
        </tbody></table></div>
      <div class="alert info small" style="margin-top:12px">${esc(r.marginal_test.fitted_residual_note)}</div>
    </div>`;

  Charts.groupedBar(el.querySelector("#bar-chart"), labels, [
    { name: "朴素：预测标签占比", values: naive, color: Charts.PALETTE.series[1] },
    { name: "反推：真实占比", values: p, color: Charts.PALETTE.series[0] },
  ]);
  Charts.intervalPlot(
    el.querySelector("#interval-chart"), labels,
    boot.median, boot.intervals["95"], boot.intervals["90"]
  );
  Charts.stackedBar(el.querySelector("#stack-naive"), labels, naive);
  el.querySelector("#stack-naive").insertAdjacentHTML("afterbegin",
    `<h3 style="margin:0 0 6px">朴素预测占比构成</h3>`);
  Charts.stackedBar(el.querySelector("#stack-est"), labels, p);
  el.querySelector("#stack-est").insertAdjacentHTML("afterbegin",
    `<h3 style="margin:0 0 6px">反推真实占比构成</h3>`);
}
