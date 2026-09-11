#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /home/abao/play2perfect/.venv_isaacsim/bin/python "$repo/scripts/play_str_compare.py" "$@"
