"""纯标准库 HTTP 服务：静态页面 + JSON API。

不依赖 Flask/FastAPI。所有写操作只接受 JSON body（不解析 multipart），
上传的原始文件内容以字节存入数据集独立目录，解析与计算均在核心层完成。
"""

from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from app.core.confusion import build_confusion
from app.core.importer import parse_batch, parse_validation
from app.core.merge import merge_confusion
from app.core.pipeline import run_analysis
from app.core.storage import Store
from app.core.synthetic import generate_synthetic_scenario

WEB_DIR = os.path.join(os.path.dirname(__file__), "static")


def _json_default(obj):
    """兜底处理 numpy 标量，避免 bool_/int64 等无法序列化。"""
    import numpy as np

    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"不可序列化的类型: {type(obj)!r}")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _build_cm(store: Store, ds_id: str):
    record = store.get_dataset(ds_id)
    rows = store.load_rows(ds_id)
    y_true = [r["y_true"] for r in rows]
    y_pred = [r["y_pred"] for r in rows]
    cm = build_confusion(y_true, y_pred, labels=record["labels"])
    return record, cm, y_true, y_pred


class Handler(BaseHTTPRequestHandler):
    store: Store = None  # 由 make_server 注入类属性

    def log_message(self, fmt, *args):  # 精简日志
        pass

    # --- 基础工具 ---
    def _send_json(self, payload, status: int = 200):
        body = json.dumps(
            payload, ensure_ascii=False, default=_json_default
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            raise ApiError(400, "请求体为空")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, f"JSON 解析失败: {exc}")
        if not isinstance(data, dict):
            raise ApiError(400, "请求体必须是 JSON 对象")
        return data

    def _static(self, path: str):
        if path in ("", "/"):
            path = "/index.html"
        safe = os.path.normpath(path).lstrip("/")
        full = os.path.join(WEB_DIR, safe)
        if not full.startswith(WEB_DIR) or not os.path.isfile(full):
            self.send_error(404, "Not Found")
            return
        ctype, _ = mimetypes.guess_type(full)
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --- 路由 ---
    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path.startswith("/api/"):
                self._route_get(path)
            else:
                self._static(path)
        except ApiError as exc:
            self._send_json({"error": exc.message}, exc.status)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"服务器内部错误: {exc}"}, 500)

    def do_POST(self):
        self._mutate()

    def do_DELETE(self):
        self._mutate()

    def _mutate(self):
        try:
            path = urlparse(self.path).path
            self._route_write(path)
        except ApiError as exc:
            self._send_json({"error": exc.message}, exc.status)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"服务器内部错误: {exc}"}, 500)

    def _route_get(self, path: str):
        store = self.store
        # parts 去掉前导 "api"
        parts = [p for p in path.split("/") if p][1:]

        if path == "/api/datasets":
            self._send_json({"datasets": store.list_datasets()})
            return

        # /api/datasets/<id>
        if len(parts) == 2 and parts[0] == "datasets":
            record = store.get_dataset(parts[1])
            payload = dict(record)
            if record["kind"] == "synthetic":
                payload["scenarios"] = store.load_scenarios(parts[1])
            self._send_json(payload)
            return

        # /api/datasets/<id>/confusion
        if len(parts) == 3 and parts[0] == "datasets" and parts[2] == "confusion":
            _, cm, _, _ = _build_cm(store, parts[1])
            self._send_json(cm.to_dict())
            return

        # /api/datasets/<id>/identifiability
        if (
            len(parts) == 3
            and parts[0] == "datasets"
            and parts[2] == "identifiability"
        ):
            _, cm, _, _ = _build_cm(store, parts[1])
            from app.core.identifiability import diagnose_identifiability

            diag = diagnose_identifiability(cm.labels, cm.matrix, cm.col_support)
            self._send_json(diag)
            return

        # /api/datasets/<id>/merges
        if len(parts) == 3 and parts[0] == "datasets" and parts[2] == "merges":
            self._send_json({"merges": store.list_merges(parts[1])})
            return

        raise ApiError(404, f"未知接口: {path}")

    def _route_write(self, path: str):
        store = self.store
        parts = [p for p in path.split("/") if p][1:]  # 去掉前导 "api"
        method = self.command

        if path == "/api/validation" and method == "POST":
            data = self._read_json()
            name = (data.get("name") or "").strip()
            content = data.get("content")
            filename = data.get("filename", "upload.csv")
            if not name:
                raise ApiError(400, "缺少数据集名称 name")
            if not isinstance(content, str):
                raise ApiError(400, "缺少文件文本内容 content")
            parsed = parse_validation(content, filename)
            labels = sorted(set(parsed["y_true"]) | set(parsed["y_pred"]))
            rows = [
                {"y_true": t, "y_pred": p}
                for t, p in zip(parsed["y_true"], parsed["y_pred"])
            ]
            record = store.create_validation(name, rows, labels, filename)
            store.save_raw_upload(record["id"], filename, content.encode("utf-8"))
            self._send_json(record, 201)
            return

        if path == "/api/synthetic" and method == "POST":
            data = self._read_json()
            name = (data.get("name") or "虚构演示场景").strip()
            payload = generate_synthetic_scenario(
                seed=int(data.get("seed", 20261123))
            )
            ds_id = store.create_synthetic(name, payload)
            self._send_json({"id": ds_id, "name": name}, 201)
            return

        # /api/datasets/<id>/analyze
        if len(parts) == 3 and parts[0] == "datasets" and parts[2] == "analyze":
            if method != "POST":
                raise ApiError(405, "仅支持 POST")
            self._analyze(store, parts[1], merged=False)
            return

        # /api/datasets/<id>/merges  POST 创建
        if len(parts) == 3 and parts[0] == "datasets" and parts[2] == "merges":
            if method != "POST":
                raise ApiError(405, "仅支持 POST")
            data = self._read_json()
            ds_id = parts[1]
            groups = data.get("groups")
            merge_name = (data.get("name") or "合并视图").strip()
            if not isinstance(groups, list) or not groups:
                raise ApiError(400, "groups 必须是非空的二维数组")
            _, cm, _, _ = _build_cm(store, ds_id)
            _, plan = merge_confusion(cm, groups)
            merge_id = store.save_merge(ds_id, merge_name, plan)
            self._send_json({"id": merge_id, "plan": plan}, 201)
            return

        # /api/merges/<id>/analyze
        if len(parts) == 3 and parts[0] == "merges" and parts[2] == "analyze":
            if method != "POST":
                raise ApiError(405, "仅支持 POST")
            self._analyze_merge(store, parts[1])
            return

        if len(parts) == 2 and parts[0] == "datasets" and method == "DELETE":
            store.delete_dataset(parts[1])
            self._send_json({"deleted": parts[1]})
            return

        if len(parts) == 2 and parts[0] == "merges" and method == "DELETE":
            store.delete_merge(parts[1])
            self._send_json({"deleted": parts[1]})
            return

        raise ApiError(404, f"未知接口: {path}")

    def _analyze(self, store: Store, ds_id: str, merged: bool):
        data = self._read_json()
        record, cm, y_true, y_pred = _build_cm(store, ds_id)

        scenario_payload = None
        if data.get("scenario") and record["kind"] == "synthetic":
            scenarios = store.load_scenarios(ds_id)
            key = data["scenario"]
            if key not in ("scenario_a", "scenario_b"):
                raise ApiError(400, "scenario 只能是 scenario_a 或 scenario_b")
            scenario_payload = scenarios[key]
            data = dict(data)
            data["pred_counts"] = scenario_payload["pred_counts"]

        batch = self._extract_batch(data, cm.labels)
        kwargs = self._analysis_kwargs(data, y_true, y_pred, record)
        if scenario_payload is not None:
            kwargs["batch_true_counts"] = scenario_payload["true_counts"]
            kwargs["batch_y_true"] = scenario_payload["y_true"]
            kwargs["batch_y_pred"] = scenario_payload["y_pred"]
        result = run_analysis(cm, batch["pred_counts"], **kwargs)
        result["dataset"] = {"id": ds_id, "name": record["name"]}
        if scenario_payload is not None:
            result["scenario_meta"] = {
                "name": scenario_payload["name"],
                "description": scenario_payload["description"],
                "class_behavior_changed": scenario_payload[
                    "class_behavior_changed"
                ],
                "true_proportions": scenario_payload["true_proportions"],
            }
        self._send_json(result)

    def _analyze_merge(self, store: Store, merge_id: str):
        merges = [
            m
            for ds in store.list_datasets()
            for m in store.list_merges(ds["id"])
        ]
        merge = next((m for m in merges if m["id"] == merge_id), None)
        if merge is None:
            raise ApiError(404, f"合并方案不存在: {merge_id}")
        ds_id = merge["dataset_id"]
        record, cm, y_true, y_pred = _build_cm(store, ds_id)
        merged_cm, _ = merge_confusion(cm, merge["plan"]["groups"])

        data = self._read_json()
        scenario_payload = None
        if data.get("scenario") and record["kind"] == "synthetic":
            scenarios = store.load_scenarios(ds_id)
            key = data["scenario"]
            if key not in ("scenario_a", "scenario_b"):
                raise ApiError(400, "scenario 只能是 scenario_a 或 scenario_b")
            scenario_payload = scenarios[key]
            data = dict(data)
            data["pred_counts"] = scenario_payload["pred_counts"]

        batch_raw = self._extract_batch(data, cm.labels)
        # 把原始标签计数按合并映射聚合
        mapping = merge["plan"]["mapping"]
        merged_counts: dict[str, int] = {}
        for label, count in batch_raw["pred_counts"].items():
            key = mapping[label]
            merged_counts[key] = merged_counts.get(key, 0) + count

        # 合并视图下重采样行的标签也要映射
        y_true_m = [mapping[v] for v in y_true]
        y_pred_m = [mapping[v] for v in y_pred]
        kwargs = self._analysis_kwargs(data, y_true_m, y_pred_m, record)
        if scenario_payload is not None:
            merged_true_counts: dict[str, int] = {}
            for label, count in scenario_payload["true_counts"].items():
                key = mapping[label]
                merged_true_counts[key] = (
                    merged_true_counts.get(key, 0) + count
                )
            kwargs["batch_true_counts"] = merged_true_counts
            kwargs["batch_y_true"] = [
                mapping[v] for v in scenario_payload["y_true"]
            ]
            kwargs["batch_y_pred"] = [
                mapping[v] for v in scenario_payload["y_pred"]
            ]
        result = run_analysis(merged_cm, merged_counts, **kwargs)
        result["merge_plan"] = merge["plan"]
        result["dataset"] = {"id": ds_id, "name": record["name"]}
        self._send_json(result)

    @staticmethod
    def _extract_batch(data: dict, labels: list[str]) -> dict:
        if "batch_content" in data and data["batch_content"]:
            parsed = parse_batch(data["batch_content"], data.get("batch_filename", ""))
            return {"pred_counts": parsed["pred_counts"]}
        if "pred_counts" in data and isinstance(data["pred_counts"], dict):
            counts = {str(k): int(v) for k, v in data["pred_counts"].items()}
            unknown = set(counts) - set(labels)
            if unknown:
                raise ApiError(400, f"批次出现校验集之外的类别: {sorted(unknown)}")
            return {"pred_counts": counts}
        if "scenario" in data:  # 虚构场景快捷分析
            return {"scenario": data["scenario"]}
        raise ApiError(400, "需要提供 pred_counts、batch_content 或 scenario")

    @staticmethod
    def _analysis_kwargs(data: dict, y_true, y_pred, record: dict) -> dict:
        kwargs = {
            "y_true": y_true,
            "y_pred": y_pred,
            "n_bootstrap": int(data.get("n_bootstrap", 300)),
            "n_null": int(data.get("n_null", 300)),
            "seed": int(data.get("seed", 20261123)),
            "similarity_threshold": float(data.get("similarity_threshold", 0.98)),
        }
        return kwargs


def make_server(data_dir: str, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    store = Store(data_dir)
    Handler.store = store
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.store = store
    return httpd
