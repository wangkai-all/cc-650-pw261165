"""隔离模式入口：先把项目根注入 sys.path，再交给 CLI。"""

import os
import sys

sys.path.insert(0, os.environ.get("APP_ROOT", os.getcwd()))

from app.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
