from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sickle_project.legacy import run_legacy_main


def main() -> int:
    return run_legacy_main("traditional_ml")


if __name__ == "__main__":
    raise SystemExit(main())
