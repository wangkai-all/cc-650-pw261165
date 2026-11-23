"""混淆矩阵与转移矩阵（列条件概率矩阵）。

不训练任何分类器，只基于校验集中的 (真值, 预测值) 计数。

约定：
  - 行 = 预测类别 j，列 = 真实类别 k
  - C[j, k] = 校验集中真实为 k、被预测为 j 的样本数
  - X = C @ diag(1 / n_k)，即 X[:, k] 为真实类别 k 在各预测类别上的分布
    （列和为 1，列随机矩阵）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfusionMatrix:
    labels: list[str]
    counts: np.ndarray  # shape (K, K)，行=预测，列=真实
    col_support: np.ndarray  # shape (K,)，每个真实类别的校验样本数
    matrix: np.ndarray  # shape (K, K)，列条件概率 X

    @property
    def n(self) -> int:
        return int(self.counts.sum())

    @property
    def k(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "counts": self.counts.tolist(),
            "matrix": self.matrix.tolist(),
            "col_support": self.col_support.tolist(),
            "n": self.n,
        }


def build_confusion(
    y_true: list,
    y_pred: list,
    labels: list | None = None,
) -> ConfusionMatrix:
    """由等长的真值/预测标签序列构建混淆矩阵。

    labels 为 None 时按真实与预测标签的并集、字典序确定类别顺序，
    保证仅在预测中出现、真值中缺席的类别也保留为一行（对应零列）。
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true 与 y_pred 长度不一致: {len(y_true)} vs {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("校验集为空，无法构建混淆矩阵")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    else:
        labels = list(labels)
        unknown = (set(y_true) | set(y_pred)) - set(labels)
        if unknown:
            raise ValueError(f"存在标签列表之外的类别: {sorted(unknown)}")

    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    counts = np.zeros((k, k), dtype=float)
    for t, p in zip(y_true, y_pred):
        counts[index[p], index[t]] += 1.0

    col_support = counts.sum(axis=0)
    matrix = np.zeros_like(counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        for kk in range(k):
            if col_support[kk] > 0:
                matrix[:, kk] = counts[:, kk] / col_support[kk]

    return ConfusionMatrix(
        labels=labels,
        counts=counts,
        col_support=col_support.astype(int),
        matrix=matrix,
    )


def predicted_distribution(pred_counts: dict, labels: list) -> np.ndarray:
    """把新批次的预测标签计数整理成与 labels 对齐的占比向量 q。"""
    q = np.zeros(len(labels), dtype=float)
    index = {label: i for i, label in enumerate(labels)}
    for label, count in pred_counts.items():
        if label not in index:
            raise ValueError(f"新批次出现校验集之外的预测类别: {label!r}")
        q[index[label]] += float(count)
    total = q.sum()
    if total <= 0:
        raise ValueError("新批次预测计数总和为 0，无法估计占比")
    return q / total
