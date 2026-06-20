from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sickle_project.legacy import run_legacy_main


TOOLS = {
    "gradcam": "tf_grad_cam",
    "lime": "tf_lime",
    "shap": "tf_shap",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1].lower() not in TOOLS:
        valid = ", ".join(sorted(TOOLS))
        print(f"Usage: python scripts/run_explainability.py <tool> [args]\nAvailable tools: {valid}")
        return 1

    tool = sys.argv[1].lower()
    return run_legacy_main(TOOLS[tool], sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
