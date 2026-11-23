"""SQLite 元数据存储 + 数据集目录管理。

每个校验集占用 data_dir 下独立子目录，原始上传内容与解析后的行分开存放；
数据库只存元数据与路径，不把大段样本塞进库。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- validation | batch | synthetic
    name TEXT NOT NULL,
    labels TEXT NOT NULL,            -- JSON array
    n_rows INTEGER NOT NULL,
    file_path TEXT,
    created_at REAL NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS merges (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    name TEXT NOT NULL,
    plan TEXT NOT NULL,              -- JSON 合并方案（保留合并前定义）
    created_at REAL NOT NULL,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id)
);
"""


class Store:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "app.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # --- 校验集 ---
    def create_validation(
        self, name: str, rows: list[dict], labels: list[str], raw_name: str
    ) -> dict:
        ds_id = uuid.uuid4().hex[:12]
        ds_dir = self.data_dir / ds_id
        ds_dir.mkdir(parents=True, exist_ok=False)
        rows_path = ds_dir / "rows.json"
        raw_path = ds_dir / ("upload" + Path(raw_name).suffix)
        with open(rows_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        record = {
            "id": ds_id,
            "kind": "validation",
            "name": name,
            "labels": labels,
            "n_rows": len(rows),
            "file_path": str(rows_path.relative_to(self.data_dir)),
            "raw_path": str(raw_path.relative_to(self.data_dir)),
            "created_at": time.time(),
            "meta": {
                "raw_filename": raw_name,
                "raw_path": str(raw_path.relative_to(self.data_dir)),
            },
        }
        self._insert_dataset(record)
        return record

    def save_raw_upload(self, ds_id: str, raw_name: str, content: bytes) -> None:
        record = self.get_dataset(ds_id)
        raw_path = self.data_dir / record["meta"]["raw_path"]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "wb") as f:
            f.write(content)

    def create_synthetic(self, name: str, payload: dict) -> str:
        ds_id = uuid.uuid4().hex[:12]
        ds_dir = self.data_dir / ds_id
        ds_dir.mkdir(parents=True, exist_ok=False)
        rows = [
            {"y_true": t, "y_pred": p}
            for t, p in zip(payload["validation"]["y_true"],
                            payload["validation"]["y_pred"])
        ]
        rows_path = ds_dir / "rows.json"
        with open(rows_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        with open(ds_dir / "scenarios.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scenario_a": payload["scenario_a"],
                    "scenario_b": payload["scenario_b"],
                    "baseline_matrix": payload["baseline_matrix"],
                    "seed": payload["seed"],
                },
                f,
                ensure_ascii=False,
            )
        record = {
            "id": ds_id,
            "kind": "synthetic",
            "name": name,
            "labels": payload["labels"],
            "n_rows": len(rows),
            "file_path": str(rows_path.relative_to(self.data_dir)),
            "created_at": time.time(),
            "meta": {"has_scenarios": True},
        }
        self._insert_dataset(record)
        return ds_id

    def _insert_dataset(self, record: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO datasets (id, kind, name, labels, n_rows, file_path,"
                " created_at, meta) VALUES (?,?,?,?,?,?,?,?)",
                (
                    record["id"],
                    record["kind"],
                    record["name"],
                    json.dumps(record["labels"], ensure_ascii=False),
                    record["n_rows"],
                    record["file_path"],
                    record["created_at"],
                    json.dumps(record.get("meta", {}), ensure_ascii=False),
                ),
            )

    def list_datasets(self, kind: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if kind:
                cur = conn.execute(
                    "SELECT * FROM datasets WHERE kind=? ORDER BY created_at DESC",
                    (kind,),
                )
            else:
                cur = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC")
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def get_dataset(self, ds_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE id=?", (ds_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"数据集不存在: {ds_id}")
        return self._row_to_dict(row)

    def delete_dataset(self, ds_id: str) -> None:
        record = self.get_dataset(ds_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM merges WHERE dataset_id=?", (ds_id,))
            conn.execute("DELETE FROM datasets WHERE id=?", (ds_id,))
        ds_dir = self.data_dir / ds_id
        if ds_dir.exists():
            for child in ds_dir.iterdir():
                child.unlink()
            ds_dir.rmdir()

    def load_rows(self, ds_id: str) -> list[dict]:
        record = self.get_dataset(ds_id)
        with open(self.data_dir / record["file_path"], encoding="utf-8") as f:
            return json.load(f)

    def load_scenarios(self, ds_id: str) -> dict:
        record = self.get_dataset(ds_id)
        path = self.data_dir / record["id"] / "scenarios.json"
        if not path.exists():
            raise KeyError("该数据集没有附带虚构场景")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # --- 合并方案（保留合并前定义） ---
    def save_merge(self, ds_id: str, name: str, plan: dict) -> str:
        self.get_dataset(ds_id)
        merge_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO merges (id, dataset_id, name, plan, created_at)"
                " VALUES (?,?,?,?,?)",
                (
                    merge_id,
                    ds_id,
                    name,
                    json.dumps(plan, ensure_ascii=False),
                    time.time(),
                ),
            )
        return merge_id

    def list_merges(self, ds_id: str) -> list[dict]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM merges WHERE dataset_id=? ORDER BY created_at DESC",
                (ds_id,),
            )
            out = []
            for row in cur.fetchall():
                d = dict(row)
                d["plan"] = json.loads(d["plan"])
                out.append(d)
            return out

    def delete_merge(self, merge_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM merges WHERE id=?", (merge_id,))

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["labels"] = json.loads(d["labels"])
        d["meta"] = json.loads(d["meta"])
        return d
