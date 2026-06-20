from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sickle_project.legacy import default_output_dir
from sickle_project.legacy import run_legacy_main


def main() -> int:
    args = sys.argv[1:]
    if "--out-dir" not in args:
        out_dir = default_output_dir("option1")
        args = [*args, "--out-dir", str(out_dir)]
    return run_legacy_main("train_cnn", args)


if __name__ == "__main__":
    raise SystemExit(main())
