"""一次完整分析的编排门面：混淆矩阵 -> 可识别性 -> 约束反推 -> 自助区间 -> 拟合诊断。"""

from __future__ import annotations

import numpy as np

from app.core.bootstrap import (
    bootstrap_proportions,
    marginal_fit_test,
    transition_matrix_test,
)
from app.core.confusion import (
    ConfusionMatrix,
    build_confusion,
    predicted_distribution,
)
from app.core.identifiability import diagnose_identifiability
from app.core.solver import estimate_true_proportions


def run_analysis(
    cm: ConfusionMatrix,
    batch_pred_counts: dict,
    *,
    y_true=None,
    y_pred=None,
    batch_true_counts: dict | None = None,
    batch_y_true: list | None = None,
    batch_y_pred: list | None = None,
    n_bootstrap: int = 500,
    n_null: int = 500,
    seed: int = 20261123,
    similarity_threshold: float = 0.98,
) -> dict:
    """对给定混淆矩阵与新批次预测计数执行完整分析。

    batch_true_counts / batch_y_true / batch_y_pred 仅在虚构/诊断场景提供
    （真实批次无真值）。转移矩阵检验必须基于逐行配对的标签序列，
    不能由两个独立的边际 count 重排重建（会破坏真值-预测配对）。
    """
    labels = cm.labels
    q = predicted_distribution(batch_pred_counts, labels)
    X = cm.matrix
    batch_size = int(sum(batch_pred_counts.values()))

    ident = diagnose_identifiability(
        labels, X, cm.col_support, similarity_threshold=similarity_threshold
    )

    p_hat, solve_info = estimate_true_proportions(X, q)
    q_forward = X @ p_hat

    per_class = []
    for i, label in enumerate(labels):
        per_class.append(
            {
                "label": label,
                "naive_predicted_share": float(q[i]),
                "estimated_true_share": float(p_hat[i]),
                "correction": float(p_hat[i] - q[i]),
                "forward_predicted_share": float(q_forward[i]),
                "forward_gap": float(q_forward[i] - q[i]),
                "validation_support": int(cm.col_support[i]),
            }
        )

    result: dict = {
        "labels": labels,
        "confusion": cm.to_dict(),
        "identifiability": ident,
        "estimate": {
            "p": [float(v) for v in p_hat],
            "solver": solve_info,
            "constraints_ok": bool(
                abs(p_hat.sum() - 1.0) < 1e-9 and p_hat.min() >= -1e-12
            ),
            "inverse_used": False,
        },
        "batch": {
            "pred_counts": {k: int(v) for k, v in batch_pred_counts.items()},
            "predicted_distribution": [float(v) for v in q],
            "size": batch_size,
        },
        "forward_check": {
            "q_observed": [float(v) for v in q],
            "q_forward_from_estimate": [float(v) for v in q_forward],
            "residual": [float(v) for v in (q_forward - q)],
            "rmse": float(np.sqrt(np.mean((q_forward - q) ** 2))),
            "total_variation_distance": float(0.5 * np.sum(np.abs(q_forward - q))),
        },
        "per_class": per_class,
    }

    rng = np.random.default_rng(seed)
    if y_true is not None and y_pred is not None and n_bootstrap > 0:
        result["bootstrap"] = bootstrap_proportions(
            np.asarray(y_true),
            np.asarray(y_pred),
            labels,
            q,
            n_replicates=n_bootstrap,
            rng=rng,
        )
    else:
        result["bootstrap"] = None

    if n_null > 0:
        result["marginal_test"] = marginal_fit_test(
            X, q, p_hat, batch_size, n_replicates=n_null, rng=rng
        )
    else:
        result["marginal_test"] = None

    if (
        batch_y_true is not None
        and batch_y_pred is not None
        and y_true is not None
    ):
        if len(batch_y_true) != len(batch_y_pred):
            raise ValueError("新批次真值与预测序列长度不一致")
        new_cm = build_confusion(
            batch_y_true, batch_y_pred, labels=labels
        )
        result["transition_test"] = transition_matrix_test(
            X,
            np.asarray(y_true),
            np.asarray(y_pred),
            new_cm.matrix,
            np.asarray(batch_y_true),
            labels,
            n_replicates=n_null,
            rng=rng,
        )
        if batch_true_counts is not None:
            result["batch"]["true_proportions"] = [
                float(batch_true_counts.get(label, 0) / batch_size)
                for label in labels
            ]
    else:
        result["transition_test"] = None

    return result
