"""Compatibility entry for the Qwen smoke check."""
from pathlib import Path
import runpy

_implementation = runpy.run_path(str(Path(__file__).parent / "qwen" / "gpu_smoke.py"))
main = _implementation["main"]

if __name__ == "__main__":
    main()
