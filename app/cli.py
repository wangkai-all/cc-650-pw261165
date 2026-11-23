"""命令行入口：启动服务 / 生成内置虚构演示数据。

用法：
  python -I -m app.cli serve --data-dir ./data --port 8000
  python -I -m app.cli seed  --data-dir ./data
"""

from __future__ import annotations

import argparse
import json
import os

from app.core.storage import Store
from app.core.synthetic import generate_synthetic_scenario
from app.web.server import make_server


def cmd_serve(args) -> None:
    httpd = make_server(args.data_dir, host=args.host, port=args.port)
    print(f"客服来信类别占比分析工具已启动: http://{args.host}:{args.port}/")
    print(f"数据目录: {os.path.abspath(args.data_dir)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
        httpd.shutdown()


def cmd_seed(args) -> None:
    store = Store(args.data_dir)
    payload = generate_synthetic_scenario(seed=args.seed)
    ds_id = store.create_synthetic("内置虚构演示（A/B 两组流量）", payload)
    print(json.dumps({"created": ds_id, "kind": "synthetic"}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="客服来信类别占比分析工具")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动 Web 服务")
    serve.add_argument("--data-dir", default="./data")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    seed = sub.add_parser("seed", help="生成内置虚构演示数据")
    seed.add_argument("--data-dir", default="./data")
    seed.add_argument("--seed", type=int, default=20261123)
    seed.set_defaults(func=cmd_seed)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
