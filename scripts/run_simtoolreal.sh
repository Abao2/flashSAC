#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
simtoolreal_root="${SIMTOOLREAL_ROOT:-${repo_root}/third_party/simtoolreal}"
export SIMTOOLREAL_ROOT="${simtoolreal_root}"
python_bin="${PYTHON_BIN:-python}"
config_name="${SIMTOOLREAL_CONFIG_NAME:-simtoolreal_flashsac}"

if [[ ! -d "${simtoolreal_root}/isaacsimenvs" ]]; then
    echo "SIMTOOLREAL_ROOT does not contain isaacsimenvs: ${simtoolreal_root}" >&2
    exit 2
fi

export PYTHONPATH="${simtoolreal_root}:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export FLASH_SAC_OUTPUT_ROOT="${FLASH_SAC_OUTPUT_ROOT:-${repo_root}}"

# SimToolReal's official task config contains asset paths relative to its repo.
cd "${simtoolreal_root}"
exec "${python_bin}" "${repo_root}/train.py" \
    --config_path "${repo_root}/configs" \
    --config_name "${config_name}" \
    "$@"

