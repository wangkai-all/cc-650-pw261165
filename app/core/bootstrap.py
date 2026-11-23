"""自助法稳定区间与拟合诊断。

三类分析，各自的适用边界在返回值 note 中写明：

1. 校验集重采样区间（bootstrap_proportions）
   逐行重抽校验样本重建转移矩阵 X* 并重解 QP，得到真实占比 p 的稳定区间。

2. 边际拟合优度检验（marginal_fit_test）
   比较新批次观测预测分布 q 与零假设中心 X·p_hat。
   **重要统计事实**：当 X 为方阵且满秩、p 为自由参数时，QP 可令
   X p_hat ≈ q 恒成立（饱和模型，自由度 0），此时该检验在数学上没有功效，
   观测统计量恒为 0。函数仍如实计算并返回饱和标志，页面据此提示
   "看不出异常 ≠ 类内稳定假设被证明"。

3. 转移矩阵两样本检验（transition_matrix_test，仅诊断/虚构场景）
   当新批次也带真值标签时（虚构流量、或未来有人工复核子样本），
   可直接构建新批次转移矩阵，与校验矩阵做逐列 TV 距离比较，
   并用校验集自助得到抽样噪声阈值。这是识别类内漂移的有效手段。
"""

from __future__ import annotations

import numpy as np

from .solver import estimate_true_proportions


def _percentiles(
    samples: np.ndarray, levels: list[float]
) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lo = np.percentile(samples, 100.0 * alpha, axis=0)
        hi = np.percentile(samples, 100.0 * (1.0 - alpha), axis=0)
        out[str(int(round(level * 100)))] = [
            [float(a), float(b)] for a, b in zip(lo, hi)
        ]
    return out


def bootstrap_proportions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    q: np.ndarray,
    *,
    n_replicates: int = 500,
    rng: np.random.Generator | None = None,
    levels: list[float] | None = None,
) -> dict:
    """校验集重采样 -> 真实占比估计的稳定区间。"""
    if rng is None:
        rng = np.random.default_rng()
    if levels is None:
        levels = [0.90, 0.95]

    n = len(y_true)
    k = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    true_idx = np.array([index[v] for v in y_true])
    pred_idx = np.array([index[v] for v in y_pred])

    # 结构性缺席列：原校验集里就没有该真实类别的样本（可识别性诊断会另行
    # 报告其占比不可识别）。重抽时这些列必然仍为零，属于预期行为而非失败，
    # 应保留零列继续求解；只有"活动列在某次重抽中偶然丢光"才跳过该次重抽。
    original_support = np.array(
        [np.sum(true_idx == kk) for kk in range(k)], dtype=int
    )
    active_cols = original_support > 0

    samples = []
    skipped_zero_column = 0
    solver_failures = 0
    for _ in range(n_replicates):
        draw = rng.integers(0, n, size=n)
        counts = np.zeros((k, k), dtype=float)
        np.add.at(counts, (pred_idx[draw], true_idx[draw]), 1.0)
        support = counts.sum(axis=0)
        if np.any(support[active_cols] == 0):
            skipped_zero_column += 1
            continue
        # 结构性零列保持零向量；仅对有样本的列做列归一化
        X_b = np.zeros((k, k), dtype=float)
        np.divide(counts, support, out=X_b, where=support > 0)
        try:
            p_b, _ = estimate_true_proportions(X_b, q)
            if np.isfinite(p_b).all():
                samples.append(p_b)
            else:
                solver_failures += 1
        except Exception:
            solver_failures += 1

    if not samples:
        raise RuntimeError("所有自助重抽均失败（可能稀有类别在小校验集中频繁缺席）")

    arr = np.vstack(samples)
    return {
        "n_replicates": n_replicates,
        "n_valid": int(arr.shape[0]),
        "skipped_zero_column": skipped_zero_column,
        "solver_failures": solver_failures,
        "intervals": _percentiles(arr, levels),
        "median": [float(v) for v in np.median(arr, axis=0)],
        "mean": [float(v) for v in arr.mean(axis=0)],
        "std": [float(v) for v in arr.std(axis=0, ddof=1)],
    }


def _largest_remainder_allocation(p: np.ndarray, n: int) -> np.ndarray:
    raw = p * n
    counts = np.floor(raw).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        counts[order[:remainder]] += 1
    return counts


def marginal_fit_test(
    X: np.ndarray,
    q: np.ndarray,
    p_hat: np.ndarray,
    batch_size: int,
    *,
    n_replicates: int = 500,
    rng: np.random.Generator | None = None,
) -> dict:
    """边际拟合优度检验（参数自助），并显式报告饱和模型导致的无功效情形。"""
    if rng is None:
        rng = np.random.default_rng()

    k = len(p_hat)
    center = X @ p_hat
    center = center / center.sum()
    batch_true_counts = _largest_remainder_allocation(p_hat, batch_size)

    square_full_rank = X.shape[0] == X.shape[1] and np.linalg.matrix_rank(X) == k

    def _stats(q_b: np.ndarray):
        diff = q_b - center
        safe = np.where(center > 0, center, np.nan)
        t2 = float(
            batch_size * np.nansum(np.where(center > 0, diff**2 / safe, 0.0))
        )
        tv = float(0.5 * np.sum(np.abs(diff)))
        return t2, tv

    null_t2, null_tv = [], []
    for _ in range(n_replicates):
        pred_counts_b = np.zeros(k, dtype=float)
        for kk in range(k):
            n_k = int(batch_true_counts[kk])
            if n_k > 0:
                pred_counts_b += rng.multinomial(n_k, X[:, kk])
        q_b = pred_counts_b / batch_size
        t2, tv = _stats(q_b)
        null_t2.append(t2)
        null_tv.append(tv)

    null_t2 = np.array(null_t2)
    null_tv = np.array(null_tv)
    obs_t2, obs_tv = _stats(q)
    p_value_t2 = float((np.sum(null_t2 >= obs_t2) + 1) / (n_replicates + 1))

    component_checks = []
    se = np.sqrt(np.where(center > 0, center / batch_size, np.nan))
    obs_z = np.where(center > 0, (q - center) / se, 0.0)
    for j in range(k):
        component_checks.append(
            {
                "label_index": j,
                "observed_share": float(q[j]),
                "expected_share_under_h0": float(center[j]),
                "diff_vs_h0": float(q[j] - center[j]),
                "z": float(obs_z[j]),
            }
        )

    return {
        "n_replicates": n_replicates,
        "batch_size": int(batch_size),
        "expected_distribution_under_h0": [float(v) for v in center],
        "statistic_t2": obs_t2,
        "p_value_t2": p_value_t2,
        "statistic_tv": obs_tv,
        "null_tv_ci95": [
            float(np.percentile(null_tv, 2.5)),
            float(np.percentile(null_tv, 97.5)),
        ],
        "component_checks": component_checks,
        "saturated_no_power": bool(square_full_rank and obs_t2 < 1e-8),
        "fitted_residual_note": (
            "若 X 为方阵且满秩、真实占比为自由参数，模型对边际预测分布是饱和拟合"
            "（自由度 0），QP 可令 X p_hat ≈ q 恒成立，本检验观测统计量恒为 0、"
            "对类内漂移无检验功效。"
        ),
        "interpretation_note": (
            "未检出异常仅表示当前信息下没有证据拒绝类内稳定假设，"
            "绝不等于该假设已被证明；要直接验证类内稳定性，需要带真值的"
            "新批次复核子样本，做转移矩阵两样本检验。"
        ),
    }


def transition_matrix_test(
    X_val: np.ndarray,
    val_true: np.ndarray,
    val_pred: np.ndarray,
    X_new: np.ndarray,
    new_true: np.ndarray,
    labels: list[str],
    *,
    n_replicates: int = 500,
    rng: np.random.Generator | None = None,
) -> dict:
    """转移矩阵两样本检验（仅当新批次带真值时可用）。

    逐列计算校验矩阵与新批次矩阵的 TV 距离。零分布用参数自助：以经验校验
    矩阵 X_val 为共同机制，按两批各自的各类真值计数独立重抽预测标签，
    重建两个矩阵并计算其列 TV，从而精确匹配两批的样本量与真值结构。
    """
    if rng is None:
        rng = np.random.default_rng()
    k = X_val.shape[1]
    index = {label: i for i, label in enumerate(labels)}
    val_true_idx = np.array([index[v] for v in val_true])
    new_true_idx = np.array([index[v] for v in new_true])
    n_val = len(val_true)
    n_new = len(new_true)

    val_support = np.array(
        [np.sum(val_true_idx == kk) for kk in range(k)], dtype=int
    )
    new_support = np.array(
        [np.sum(new_true_idx == kk) for kk in range(k)], dtype=int
    )

    observed_col_tv = np.array(
        [0.5 * np.sum(np.abs(X_val[:, kk] - X_new[:, kk])) for kk in range(k)]
    )

    def _resample_matrix(support: np.ndarray) -> np.ndarray:
        counts = np.zeros((k, k), dtype=float)
        for kk in range(k):
            n_k = int(support[kk])
            if n_k > 0:
                counts[:, kk] = rng.multinomial(n_k, X_val[:, kk])
        return counts / support

    # 共同机制 X_val 下，一个 n_val 样本与一个 n_new 样本的矩阵差异零分布
    null_col_tv = []
    for _ in range(n_replicates):
        x_b_val = _resample_matrix(val_support)
        x_b_new = _resample_matrix(new_support)
        null_col_tv.append(
            [
                0.5 * np.sum(np.abs(x_b_val[:, kk] - x_b_new[:, kk]))
                for kk in range(k)
            ]
        )
    null_col_tv = np.array(null_col_tv)

    columns = []
    for kk, label in enumerate(labels):
        q95 = float(np.percentile(null_col_tv[:, kk], 95))
        q99 = float(np.percentile(null_col_tv[:, kk], 99))
        obs = float(observed_col_tv[kk])
        p_value = float(
            (np.sum(null_col_tv[:, kk] >= obs) + 1) / (null_col_tv.shape[0] + 1)
        )
        columns.append(
            {
                "label": label,
                "observed_tv": obs,
                "null_tv_q95": q95,
                "null_tv_q99": q99,
                "p_value": p_value,
                "drifted": bool(obs > q95),
                "strongly_drifted": bool(obs > q99),
            }
        )

    any_drift = any(c["drifted"] for c in columns)
    return {
        "n_replicates": null_col_tv.shape[0],
        "columns": columns,
        "misfit_flagged": any_drift,
        "overall_max_tv": float(observed_col_tv.max()),
        "availability_note": (
            "该检验需要新批次的真值标签；真实业务里新批次通常只有预测结果，"
            "此时只能依赖边际检验（饱和时无功效）。建议定期抽取小样本做人工复核，"
            "为该检验提供真值来源。"
        ),
        "interpretation_note": (
            "未标记漂移仅表示差异在校验集抽样噪声范围内，"
            "并不证明类内行为完全稳定；小幅、缓慢的漂移仍可能漏检。"
        ),
    }
