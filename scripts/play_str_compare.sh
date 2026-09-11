#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/abao/play2perfect/.venv_isaacsim/bin/python}"
exec "${python_bin}" "$repo/scripts/play_str_compare.py" "$@"

