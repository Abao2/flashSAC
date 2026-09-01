#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
port="${TB_PORT:-6006}"

exec "${python_bin}" -m tensorboard.main --logdir "${repo_root}/runs" --host 127.0.0.1 --port "${port}"
