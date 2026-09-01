#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dex4d_root="${DEX4D_ROOT:-${repo_root}/../Dex4D/Dex4D-Simulation}"
python_bin="${PYTHON_BIN:-python}"
config_name="${DEX4D_CONFIG:-dex4d_flashsac}"

if [[ ! -d "${dex4d_root}/dex4d_policy/assets" ]]; then
    echo "DEX4D_ROOT does not contain dex4d_policy/assets: ${dex4d_root}" >&2
    exit 2
fi

export DEX4D_ROOT="$(cd "${dex4d_root}" && pwd)"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

cd "${repo_root}"
exec "${python_bin}" play_isaaclab.py \
    --config_path "${repo_root}/configs" \
    --config_name "${config_name}" \
    --overrides +env.task_cfg_overrides.show_goal_marker=true \
    --overrides agent.buffer_max_length=1 \
    --overrides agent.buffer_min_length=1 \
    --overrides agent.sample_batch_size=1 \
    "$@"
