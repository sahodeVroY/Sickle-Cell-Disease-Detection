from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .project import PROJECT_ROOT


def ensure_project_root_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@contextmanager
def patched_argv(module_name: str, args: Iterable[str] | None = None):
    previous = sys.argv[:]
    sys.argv = [module_name, *(list(args or []))]
    try:
        yield
    finally:
        sys.argv = previous


def run_legacy_main(module_name: str, args: Iterable[str] | None = None) -> int:
    ensure_project_root_on_path()
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise RuntimeError(f"Module {module_name} does not define a main() function.")

    with patched_argv(module_name, args):
        result = main()
    return 0 if result is None else int(result)


def default_output_dir(name: str) -> Path:
    out_dir = PROJECT_ROOT / "outputs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

