"""虚构流量生成：用于页面演示拟合诊断能识别什么、不能证明什么。

两套场景共用同一份校验集（由基线转移矩阵生成）：
  - 场景 A：新批次仅类别比例漂移，类内误判方式与校验集完全一致 ->
    残差检验不应报异常（但注意：不报异常 ≠ 假设被证明）；
  - 场景 B：类别比例与类内误判方式同时漂移 ->
    沿用校验矩阵反推会出现系统性正向残差，检验应识别出明显不符。

所有数据均为合成，真值占比随数据一起返回，仅供页面"对照真值"演示，
真实业务中新批次没有真值。
"""

from __future__ import annotations

import numpy as np

DEFAULT_LABELS = ["咨询", "投诉", "查询进度", "表扬"]

# 基线类内行为：列 = 真实类别，列和 = 1
BASELINE_X = np.array(
    [
        # 咨询   投诉  查询进度  表扬
        [0.82, 0.18, 0.06, 0.05],  # 预测为咨询
        [0.08, 0.66, 0.10, 0.04],  # 预测为投诉
        [0.07, 0.12, 0.81, 0.06],  # 预测为查询进度
        [0.03, 0.04, 0.03, 0.85],  # 预测为表扬
    ]
)

# 场景 B 的漂移后类内行为（投诉更容易被误判成咨询，查询进度流向也改变）
DRIFT_X = np.array(
    [
        [0.80, 0.34, 0.10, 0.08],
        [0.10, 0.50, 0.08, 0.05],
        [0.07, 0.11, 0.77, 0.07],
        [0.03, 0.05, 0.05, 0.80],
    ]
)

BASELINE_TRUE_P = np.array([0.55, 0.20, 0.20, 0.05])
DRIFT_TRUE_P = np.array([0.40, 0.33, 0.17, 0.10])


def _sample_batch(
    true_p: np.ndarray, X: np.ndarray, size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """按真实占比抽真实标签，再按类内条件分布抽预测标签。"""
    k = X.shape[0]
    true_idx = rng.choice(k, size=size, p=true_p)
    pred_idx = np.empty(size, dtype=int)
    for kk in range(k):
        mask = true_idx == kk
        n_k = int(mask.sum())
        if n_k:
            pred_idx[mask] = rng.choice(k, size=n_k, p=X[:, kk])
    return true_idx, pred_idx


def generate_synthetic_scenario(
    *,
    validation_size: int = 4000,
    batch_size_a: int = 3000,
    batch_size_b: int = 3000,
    seed: int = 20261123,
) -> dict:
    """生成校验集 + A/B 两个虚构新批次。"""
    rng = np.random.default_rng(seed)
    labels = DEFAULT_LABELS

    # 校验集：基线比例 + 基线类内行为，带真值
    val_true_idx, val_pred_idx = _sample_batch(
        BASELINE_TRUE_P, BASELINE_X, validation_size, rng
    )

    # 场景 A：只漂移比例，类内行为保持基线
    a_true_idx, a_pred_idx = _sample_batch(
        DRIFT_TRUE_P, BASELINE_X, batch_size_a, rng
    )
    # 场景 B：比例与类内行为同时漂移
    b_true_idx, b_pred_idx = _sample_batch(
        DRIFT_TRUE_P, DRIFT_X, batch_size_b, rng
    )

    def _to_labels(idx: np.ndarray) -> list[str]:
        return [labels[i] for i in idx]

    def _counts(idx: np.ndarray) -> dict[str, int]:
        c = np.bincount(idx, minlength=len(labels))
        return {labels[i]: int(c[i]) for i in range(len(labels))}

    return {
        "labels": labels,
        "validation": {
            "y_true": _to_labels(val_true_idx),
            "y_pred": _to_labels(val_pred_idx),
            "size": validation_size,
        },
        "scenario_a": {
            "name": "A：仅类别比例漂移",
            "description": (
                "新批次真实占比改变，但类内误判方式与校验集完全一致；"
                "拟合诊断预期不报明显异常。"
            ),
            "y_true": _to_labels(a_true_idx),
            "y_pred": _to_labels(a_pred_idx),
            "pred_counts": _counts(a_pred_idx),
            "true_counts": _counts(a_true_idx),
            "true_proportions": [float(v) for v in DRIFT_TRUE_P],
            "class_behavior_changed": False,
        },
        "scenario_b": {
            "name": "B：比例与类内误判同时漂移",
            "description": (
                "新批次真实占比改变，且投诉→咨询等类内误判方式也发生变化；"
                "沿用校验矩阵反推应出现系统性残差，诊断应识别明显不符。"
            ),
            "y_true": _to_labels(b_true_idx),
            "y_pred": _to_labels(b_pred_idx),
            "pred_counts": _counts(b_pred_idx),
            "true_counts": _counts(b_true_idx),
            "true_proportions": [float(v) for v in DRIFT_TRUE_P],
            "class_behavior_changed": True,
            "drift_matrix": DRIFT_X.tolist(),
        },
        "baseline_matrix": BASELINE_X.tolist(),
        "seed": seed,
    }
