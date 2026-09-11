#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SIMTOOLREAL_CONFIG_NAME=simtoolreal_state_teacher
export PYTHON_BIN="${PYTHON_BIN:-/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python}"
exec "${repo_root}/scripts/run_simtoolreal.sh" "$@"
