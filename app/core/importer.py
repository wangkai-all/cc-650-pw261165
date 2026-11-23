"""安全解析用户上传的校验集 / 新批次数据（CSV 或 JSON）。

解析层只做格式校验与结构整理，不执行任何文件中的代码。
调用方负责把内容写入独立目录，且解释器以 -I 运行。
"""

from __future__ import annotations

import csv
import io
import json

VALIDATION_REQUIRED = {"y_true", "y_pred"}
BATCH_REQUIRED = {"y_pred"}
ALIASES = {
    "y_true": ("y_true", "true", "label", "实际", "真值", "真实类别", "真实标签"),
    "y_pred": ("y_pred", "pred", "prediction", "预测", "预测类别", "预测标签"),
}


def _normalize_header(name: str) -> str | None:
    key = name.strip().lower()
    for canonical, aliases in ALIASES.items():
        if key in (a.lower() for a in aliases):
            return canonical
    return None


def parse_validation(content: str, filename: str = "") -> dict:
    """解析带真值的校验集，返回 {labels?, y_true, y_pred}。"""
    rows = _parse_rows(content, filename)
    if not rows:
        raise ValueError("校验集文件为空")
    headers = set(rows[0].keys())
    if not VALIDATION_REQUIRED.issubset(headers):
        raise ValueError(
            "校验集必须包含 y_true（真值）与 y_pred（预测）两列，"
            f"实际列: {sorted(headers)}"
        )
    y_true, y_pred = [], []
    for i, row in enumerate(rows, start=2):
        t = (row.get("y_true") or "").strip()
        p = (row.get("y_pred") or "").strip()
        if not t or not p:
            raise ValueError(f"第 {i} 行存在空标签: {row}")
        y_true.append(t)
        y_pred.append(p)
    return {"y_true": y_true, "y_pred": y_pred, "n": len(y_true)}


def parse_batch(content: str, filename: str = "") -> dict:
    """解析仅有预测类别的新批次。

    支持两种形态：
      1. 逐行记录，至少含 y_pred 列（可带 count 列聚合）；
      2. 纯映射 {"咨询": 120, "投诉": 30}（JSON）。
    """
    text = content.strip()
    if text[:1] in "{[":
        data = json.loads(text)
        if isinstance(data, dict) and all(
            isinstance(v, (int, float)) for v in data.values()
        ):
            counts = {str(k): int(v) for k, v in data.items() if int(v) > 0}
            if not counts:
                raise ValueError("批次计数全为 0")
            return {"pred_counts": counts, "n": sum(counts.values())}
        if isinstance(data, list):
            rows = data
        else:
            raise ValueError("JSON 批次需为标签->计数映射，或记录列表")
    else:
        rows = _parse_rows(content, filename)

    if not rows:
        raise ValueError("新批次文件为空")
    if isinstance(rows[0], dict) and "y_pred" not in rows[0]:
        raise ValueError("新批次记录必须包含 y_pred（预测类别）列")

    counts: dict[str, int] = {}
    for i, row in enumerate(rows, start=2):
        p = str(row.get("y_pred", "")).strip()
        if not p:
            raise ValueError(f"第 {i} 行预测标签为空")
        c = row.get("count")
        weight = int(float(c)) if c not in (None, "") else 1
        if weight < 0:
            raise ValueError(f"第 {i} 行计数为负")
        counts[p] = counts.get(p, 0) + weight
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("新批次样本数为 0")
    return {"pred_counts": counts, "n": total}


def _parse_rows(content: str, filename: str) -> list[dict]:
    if filename.lower().endswith(".json") or content.lstrip()[:1] in "[{":
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON 输入需为对象列表")
        normalized = []
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("JSON 列表的每个元素必须是对象")
            normalized.append(
                {
                    (_normalize_header(k) or k.strip()): (
                        "" if v is None else str(v)
                    )
                    for k, v in row.items()
                }
            )
        return normalized

    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        raise ValueError("CSV 缺少表头")
    renamed = [_normalize_header(h) or h.strip() for h in reader.fieldnames]
    rows = []
    for raw in reader:
        rows.append({renamed[i]: (v or "").strip() for i, v in enumerate(raw.values())})
    return rows
