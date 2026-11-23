#!/usr/bin/env bash
# 启动 Web 服务。以 -I 隔离模式运行解释器，避免从数据目录加载任何植入模块。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi
cd "$ROOT"
export APP_ROOT="$ROOT"
exec "$PY" -I "$ROOT/scripts/_entry.py" "$@"
