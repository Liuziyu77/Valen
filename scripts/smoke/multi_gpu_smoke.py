"""Compatibility entry for the Qwen smoke check."""
from pathlib import Path
import runpy

_implementation = runpy.run_path(str(Path(__file__).parent / "qwen" / "multi_gpu_smoke.py"))
main = _implementation["main"]
weight_digest = _implementation["weight_digest"]

if __name__ == "__main__":
    main()
