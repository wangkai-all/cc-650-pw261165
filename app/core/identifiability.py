"""可识别性诊断：在不做任何反推之前，先判断 X 的哪些列分不清。

三类结构性问题：
  1. 某真实类别在校验集中完全缺席 -> 对应零列，该占比没有任何信息；
  2. 两列（或多列）几乎一样 -> 类别条件分布不可区分，只能识别其总和；
  3. 整体秩亏 / 条件数过大 -> 反推数值上不稳定。

只使用 SVD 做诊断（不参与求逆估计），并把不可区分的列按传递闭包聚类，
给出"哪些比例无法分清"的明确结论。
"""

from __future__ import annotations

import numpy as np


def diagnose_identifiability(
    labels: list[str],
    X: np.ndarray,
    col_support: np.ndarray,
    *,
    similarity_threshold: float = 0.98,
    rank_tol: float = 1e-10,
) -> dict:
    """返回结构化诊断结果。

    groups: 不可区分组列表，每组给出成员标签、组内最大列间余弦相似度；
            独立可识别的类别为单成员组。
    """
    k = len(labels)
    X = np.asarray(X, dtype=float)
    missing = [labels[i] for i in range(k) if col_support[i] == 0]

    # --- 列间余弦相似度（零列不参与相似度比较） ---
    norms = np.linalg.norm(X, axis=0)
    active = [i for i in range(k) if col_support[i] > 0]
    sim = np.zeros((k, k))
    near_pairs = []
    for a_pos, i in enumerate(active):
        sim[i, i] = 1.0
        for j in active[a_pos + 1:]:
            denom = norms[i] * norms[j]
            c = float(X[:, i] @ X[:, j] / denom) if denom > 0 else 0.0
            sim[i, j] = sim[j, i] = c
            if c >= similarity_threshold:
                near_pairs.append((labels[i], labels[j], c))

    # --- 并查集：按近似对做传递闭包，得到不可区分组 ---
    parent = list(range(k))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    label_index = {label: i for i, label in enumerate(labels)}
    for a, b, _ in near_pairs:
        union(label_index[a], label_index[b])

    grouped: dict[int, list[int]] = {}
    for i in range(k):
        grouped.setdefault(find(i), []).append(i)

    groups = []
    for members in grouped.values():
        members = sorted(members)
        member_labels = [labels[i] for i in members]
        max_sim = 1.0
        if len(members) > 1:
            max_sim = max(
                sim[a, b]
                for pos, a in enumerate(members)
                for b in members[pos + 1:]
            )
        groups.append(
            {
                "members": member_labels,
                "identifiable": len(members) == 1 and col_support[members[0]] > 0,
                "max_pair_similarity": round(float(max_sim), 6),
                "missing": col_support[members[0]] == 0,
            }
        )
    groups.sort(key=lambda g: (g["identifiable"], g["members"]))

    # --- SVD 秩与条件数（仅作诊断） ---
    active_X = X[:, active] if active else X
    s = np.linalg.svd(active_X, compute_uv=False)
    effective_rank = int(np.sum(s > rank_tol * (s[0] if s.size else 1.0)))
    cond = float(s[0] / s[-1]) if s.size and s[-1] > 0 else float("inf")
    rank_deficient = effective_rank < active_X.shape[1]

    warnings = []
    if missing:
        warnings.append(
            f"真实类别 {missing} 在校验集中完全缺席（零列），"
            "其占比无法从数据中识别，仅受单纯形约束牵制。"
        )
    ambiguous = [g["members"] for g in groups if len(g["members"]) > 1]
    if ambiguous:
        warnings.append(
            f"类别组 {ambiguous} 的条件分布余弦相似度 ≥ "
            f"{similarity_threshold}，组内各类占比无法分清，只能可靠识别组总和。"
        )
    if rank_deficient:
        warnings.append(
            f"转移矩阵有效秩 {effective_rank} < 活动列数 {active_X.shape[1]}，"
            "反推问题秩亏，建议合并类别后重算。"
        )
    elif cond > 1e6:
        warnings.append(
            f"转移矩阵条件数约 {cond:.2e}，反推对预测分布噪声较敏感，"
            "自助法区间会更宽，建议关注区间而非点估计。"
        )

    return {
        "missing_classes": missing,
        "near_identical_pairs": [
            {"a": a, "b": b, "cosine_similarity": round(float(c), 6)}
            for a, b, c in near_pairs
        ],
        "groups": groups,
        "svd_singular_values": [round(float(v), 8) for v in s],
        "effective_rank": effective_rank,
        "active_columns": len(active),
        "condition_number": None if np.isinf(cond) else round(cond, 4),
        "rank_deficient": bool(rank_deficient),
        "similarity_threshold": similarity_threshold,
        "warnings": warnings,
    }
