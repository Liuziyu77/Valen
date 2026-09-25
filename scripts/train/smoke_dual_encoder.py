"""Compatibility entry for the dual-encoder smoke check."""
from pathlib import Path
import runpy

_implementation = runpy.run_path(str(Path(__file__).resolve().parents[1] / "smoke/dual_encoder/gpu_smoke.py"))
main = _implementation["main"]
random_encoders = _implementation["random_encoders"]

if __name__ == "__main__":
    main()
