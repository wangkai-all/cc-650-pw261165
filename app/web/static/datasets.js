// 数据集 / 校验集管理页：上传、混淆矩阵检查、可识别性诊断、类别合并。

async function renderDatasetsPage(main) {
  await loadDatasets();
  main.innerHTML = `
    <div class="card">
      <h2>导入校验集（带真值）</h2>
      <p class="muted small">CSV/JSON，需含 <span class="kbd">y_true</span>（真值）与 <span class="kbd">y_pred</span>（预测）两列。
      支持中文列名"真值/真实类别/预测/预测类别"。原始文件存入独立目录，不会被当作代码执行。</p>
      <div class="grid2">
        <label class="field"><span>数据集名称</span><input id="ds-name" placeholder="例如：2026Q4 客服校验集"></label>
        <label class="field"><span>文件名（用于识别格式）</span><input id="ds-fname" value="validation.csv"></label>
      </div>
      <label class="field"><span>文件内容（粘贴 CSV 或 JSON 文本）</span>
        <textarea id="ds-content" placeholder="y_true,y_pred&#10;咨询,咨询&#10;投诉,咨询&#10;..."></textarea></label>
      <div class="row">
        <button id="ds-upload">导入校验集</button>
        <button class="secondary" id="ds-synthetic">生成内置虚构演示（A/B 两组流量）</button>
      </div>
    </div>
    <div class="card">
      <h2>已导入数据集</h2>
      <div id="ds-list"></div>
    </div>
    <div id="ds-detail"></div>`;

  main.querySelector("#ds-upload").onclick = async () => {
    try {
      const r = await Api.post("/api/validation", {
        name: main.querySelector("#ds-name").value.trim() || "未命名校验集",
        filename: main.querySelector("#ds-fname").value.trim() || "upload.csv",
        content: main.querySelector("#ds-content").value,
      });
      toast(`已导入 ${r.n_rows} 条样本`);
      renderDatasetsPage(main);
    } catch (e) { toast(e.message, true); }
  };
  main.querySelector("#ds-synthetic").onclick = async () => {
    try {
      const r = await Api.post("/api/synthetic", { name: "内置虚构演示（A/B 两组流量）" });
      toast("已生成虚构演示数据 " + r.id);
      renderDatasetsPage(main);
    } catch (e) { toast(e.message, true); }
  };
  await renderDatasetList(main);
}

async function renderDatasetList(main) {
  const list = main.querySelector("#ds-list");
  if (!State.datasets.length) {
    list.innerHTML = `<div class="empty">还没有数据集，请先导入或生成虚构演示。</div>`;
    return;
  }
  const rows = State.datasets.map((d) => `
    <tr>
      <td><strong>${esc(d.name)}</strong></td>
      <td><span class="badge neutral">${d.kind === "synthetic" ? "虚构演示" : "校验集"}</span></td>
      <td>${d.n_rows}</td>
      <td>${esc(d.labels.join("、"))}</td>
      <td class="small muted">${new Date(d.created_at * 1000).toLocaleString()}</td>
      <td>
        <button class="secondary" data-open="${d.id}">检查</button>
        <button class="ghost" data-del="${d.id}">删除</button>
      </td>
    </tr>`).join("");
  list.innerHTML = `<div class="scroll"><table>
    <thead><tr><th>名称</th><th>类型</th><th>样本数</th><th>类别</th><th>创建时间</th><th>操作</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;

  list.querySelectorAll("[data-del]").forEach((b) => (b.onclick = async () => {
    if (!confirm("确认删除该数据集及其目录？")) return;
    await Api.del(`/api/datasets/${b.dataset.del}`);
    toast("已删除");
    renderDatasetsPage(main);
  }));
  list.querySelectorAll("[data-open]").forEach((b) => (b.onclick = () =>
    renderDatasetDetail(main, b.dataset.open)));
}

async function renderDatasetDetail(main, dsId) {
  const detail = main.querySelector("#ds-detail");
  detail.innerHTML = `<div class="card empty">加载混淆矩阵…</div>`;
  const [record, cm, merges, ident] = await Promise.all([
    Api.get(`/api/datasets/${dsId}`),
    Api.get(`/api/datasets/${dsId}/confusion`),
    Api.get(`/api/datasets/${dsId}/merges`),
    Api.get(`/api/datasets/${dsId}/identifiability`),
  ]);

  detail.innerHTML = `
    <div class="card">
      <h2>混淆矩阵检查 — ${esc(record.name)}</h2>
      <p class="muted small">行＝预测类别，列＝真实类别；格内为条件概率 P(预测|真实) 与样本数。每列和为 100%。点击列可查看该真实类别最容易与哪些预测混淆。</p>
      <div id="heatmap"></div>
      <h3>逐类混淆明细</h3>
      <div id="per-class-conf"></div>
    </div>
    <div class="card" id="ident-card"></div>
    <div class="card">
      <h2>类别合并（保留合并前定义）</h2>
      <p class="muted small">勾选两个及以上类别创建合并视图，反推与区间在合并后的类别空间重算；原始类别定义始终保留，可随时回到未合并视图。</p>
      <div id="merge-builder"></div>
      <div id="merge-list" style="margin-top:16px"></div>
    </div>`;

  Charts.heatmap(detail.querySelector("#heatmap"), cm.labels, cm.matrix, cm.counts);
  renderPerClassConf(detail, cm);
  renderIdentifiability(detail.querySelector("#ident-card"), cm, ident);
  renderMergeBuilder(detail, record, cm, merges.merges, dsId, main);
}

// 可识别性诊断卡片（数据来自后端完整诊断：零列、近似列并查集分组、SVD 秩、条件数）
function renderIdentifiability(card, cm, ident) {
  const groupRows = ident.groups.map((g) => `
    <tr>
      <td>${g.members.map(esc).join("、")}</td>
      <td>${g.identifiable
        ? `<span class="badge good">可识别</span>`
        : g.missing
          ? `<span class="badge bad">缺席（零列）</span>`
          : `<span class="badge warn">组内分不清</span>`}</td>
      <td class="muted small">${g.max_pair_similarity.toFixed(4)}</td>
    </tr>`).join("");
  card.innerHTML = `
    <h2>可识别性诊断</h2>
    ${ident.warnings.length
      ? ident.warnings.map((w) => `<div class="alert warn">${esc(w)}</div>`).join("")
      : `<div class="alert info">所有真实类别均有校验样本，且列间余弦相似度均低于 ${ident.similarity_threshold}，未发现结构性不可分。</div>`}
    <div class="grid3" style="margin-bottom:12px">
      <div class="stat"><div class="k">有效秩 / 活动列</div><div class="v" style="font-size:18px">${ident.effective_rank}/${ident.active_columns}</div></div>
      <div class="stat"><div class="k">条件数</div><div class="v" style="font-size:18px">${ident.condition_number == null ? "∞" : ident.condition_number.toExponential(1)}</div></div>
      <div class="stat"><div class="k">缺席类别</div><div class="v" style="font-size:18px">${ident.missing_classes.length}</div></div>
    </div>
    <h3>类别可识别性分组</h3>
    <div class="scroll"><table>
      <thead><tr><th>类别 / 不可区分组</th><th>状态</th><th>组内最大余弦相似度</th></tr></thead>
      <tbody>${groupRows}</tbody></table></div>
    <p class="muted small" style="margin-top:10px">SVD 奇异值：${ident.svd_singular_values.map((v) => v.toExponential(1)).join("、")}。
    若出现缺席类别或近似重复列，建议在下方合并对应类别后重新反推；合并只生成派生视图，原始定义保留。</p>`;
}

function renderPerClassConf(detail, cm) {
  const wrap = detail.querySelector("#per-class-conf");
  const rows = cm.labels.map((trueLab, j) => {
    const probs = cm.matrix.map((row, i) => ({ pred: cm.labels[i], p: row[j], n: cm.counts[i][j] }))
      .sort((a, b) => b.p - a.p);
    const top = probs.slice(0, 3).map((x) =>
      `${esc(x.pred)} <span class="muted">${pct(x.p)}（${x.n}）</span>`).join(" ｜ ");
    return `<tr><td><strong>${esc(trueLab)}</strong></td><td>${cm.col_support[j]}</td><td>${top}</td></tr>`;
  }).join("");
  wrap.innerHTML = `<div class="scroll"><table>
    <thead><tr><th>真实类别</th><th>校验样本数</th><th>最常被预测为（前三）</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function renderMergeBuilder(detail, record, cm, merges, dsId, main) {
  const builder = detail.querySelector("#merge-builder");
  builder.innerHTML = `
    <div class="pill-group" id="merge-pills">
      ${cm.labels.map((l) => `<span class="pill" data-lab="${esc(l)}">${esc(l)}</span>`).join("")}
    </div>
    <label class="field" style="margin-top:12px"><span>合并视图名称</span><input id="merge-name" placeholder="例如：投诉类合并"></label>
    <button id="merge-go">创建合并视图并重算</button>
    <p class="muted small" id="merge-hint">点击选择要合并的类别（至少 2 个）。</p>`;
  const picked = new Set();
  builder.querySelectorAll(".pill").forEach((p) => (p.onclick = () => {
    const lab = p.dataset.lab;
    if (picked.has(lab)) { picked.delete(lab); p.classList.remove("on"); }
    else { picked.add(lab); p.classList.add("on"); }
  }));
  builder.querySelector("#merge-go").onclick = async () => {
    if (picked.size < 2) { toast("请至少选择两个类别合并", true); return; }
    try {
      await Api.post(`/api/datasets/${dsId}/merges`, {
        name: builder.querySelector("#merge-name").value.trim() || "合并视图",
        groups: [Array.from(picked)],
      });
      toast("合并视图已创建");
      renderDatasetDetail(main, dsId);
    } catch (e) { toast(e.message, true); }
  };

  const listEl = detail.querySelector("#merge-list");
  if (!merges.length) { listEl.innerHTML = `<p class="muted small">暂无合并视图。</p>`; return; }
  listEl.innerHTML = `<h3>已有合并视图（保留合并前定义）</h3><div class="scroll"><table>
    <thead><tr><th>名称</th><th>合并组</th><th>合并前类别</th><th>操作</th></tr></thead><tbody>
    ${merges.map((m) => `<tr>
      <td>${esc(m.name)}</td>
      <td>${m.plan.groups.map((g) => g.map(esc).join("＋")).join("；")}</td>
      <td class="muted small">${esc(m.plan.original_labels.join("、"))}</td>
      <td><button class="ghost" data-del-merge="${m.id}">删除</button></td>
    </tr>`).join("")}</tbody></table></div>`;
  listEl.querySelectorAll("[data-del-merge]").forEach((b) => (b.onclick = async () => {
    await Api.del(`/api/merges/${b.dataset.delMerge}`);
    toast("已删除合并视图");
    renderDatasetDetail(main, dsId);
  }));
}
