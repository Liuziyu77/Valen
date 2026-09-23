#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
VJ_PYTHON="${VJ_PYTHON:-python3}"
"$VJ_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 required"'
"$VJ_PYTHON" -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple --upgrade pip
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
.venv/bin/python -m pip install --index-url https://pypi.org/simple -e '.[test]'
