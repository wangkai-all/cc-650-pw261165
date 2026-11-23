"""类别合并：把指定真实/预测类别池化后重新构建混淆矩阵。

合并是**派生视图**：原始混淆矩阵与原始类别定义始终保留，
合并结果记录分组方案，可随时丢弃合并视图回到原始定义。
合并按标签名指定（支持部分合并，未提及的类别各自独立）。
"""

from __future__ import annotations

import numpy as np

from .confusion import ConfusionMatrix, build_confusion


def _merged_label(group: tuple[str, ...]) -> str:
    return "+".join(group)


def merge_confusion(
    cm: ConfusionMatrix, groups: list[list[str]] | list[tuple[str, ...]]
) -> tuple[ConfusionMatrix, dict]:
    """按 groups 合并类别，返回 (新混淆矩阵, 合并方案说明)。

    groups 示例: [["投诉", "建议"], ["咨询"]]
      - 每个子列表合并为一个类别，标签用 "+" 连接；
      - 未出现在 groups 中的原始类别自动保持独立；
      - 同一类别不允许出现在多个组里。
    """
    seen: set[str] = set()
    normalized_groups: list[tuple[str, ...]] = []
    for group in groups:
        tup = tuple(group)
        if len(tup) < 2:
            continue
        for label in tup:
            if label not in cm.labels:
                raise ValueError(f"待合并类别 {label!r} 不在原始类别中")
            if label in seen:
                raise ValueError(f"类别 {label!r} 出现在多个合并组中")
            seen.add(label)
        normalized_groups.append(tup)

    if not normalized_groups:
        raise ValueError("没有提供任何包含两个及以上类别的合并组")

    mapping: dict[str, str] = {}
    new_labels: list[str] = []
    for group in normalized_groups:
        merged = _merged_label(group)
        new_labels.append(merged)
        for label in group:
            mapping[label] = merged

    # 未提及的类别保持独立，沿用原始顺序
    for label in cm.labels:
        if label not in seen:
            mapping[label] = label
            new_labels.append(label)

    new_index = {label: i for i, label in enumerate(new_labels)}
    old_index = {label: i for i, label in enumerate(cm.labels)}
    k_new = len(new_labels)
    counts = cm.counts
    merged_counts = np.zeros((k_new, k_new), dtype=float)
    for old_pred in cm.labels:
        for old_true in cm.labels:
            merged_counts[
                new_index[mapping[old_pred]], new_index[mapping[old_true]]
            ] += counts[old_index[old_pred], old_index[old_true]]

    # 复用标准构建路径，保证列归一化逻辑一致
    y_true, y_pred = [], []
    for j, pred_label in enumerate(cm.labels):
        for kk, true_label in enumerate(cm.labels):
            n = int(counts[j, kk])
            y_true.extend([mapping[true_label]] * n)
            y_pred.extend([mapping[pred_label]] * n)
    merged_cm = build_confusion(y_true, y_pred, labels=new_labels)

    plan = {
        "original_labels": list(cm.labels),
        "merged_labels": new_labels,
        "mapping": mapping,
        "groups": [list(g) for g in normalized_groups],
    }
    return merged_cm, plan
