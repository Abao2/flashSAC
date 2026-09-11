#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python}"
memory_limit_mib="${GPU_QUEUE_MEMORY_LIMIT_MIB:-8192}"
status_file="${repo_root}/logs/dex4d_m6_wuji.status"

mkdir -p "${repo_root}/logs"

set_status() {
    printf '%s %s\n' "$(date -Is)" "$*" >"${status_file}"
    printf '[Dex4D-M6-Wuji] %s %s\n' "$(date -Is)" "$*"
}

trap 'set_status "FAILED line=${LINENO}"' ERR

set_status "WAITING_FOR_GPU used_memory_below=${memory_limit_mib}MiB"
gpu=""
polls=0
while [[ -z "${gpu}" ]]; do
    gpu="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F, -v limit="${memory_limit_mib}" '$2 + 0 < limit {gsub(/ /, "", $1); print $1; exit}')"
    if [[ -z "${gpu}" ]]; then
        if ((polls % 10 == 0)); then
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        fi
        polls=$((polls + 1))
        sleep 30
    fi
done

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHON_BIN="${python_bin}"
set_status "ONE_ENV_SMOKE gpu=${gpu}"
"${repo_root}/scripts/smoke_dex4d_m6_wuji.sh" \
    --overrides exp_name=wuji_m6_one_env_smoke \
    --overrides num_env_steps=8 \
    --overrides num_train_envs=1 \
    --overrides updates_per_interaction_step=0.0 \
    --overrides agent.buffer_max_length=256 \
    --overrides agent.buffer_min_length=128 \
    --overrides agent.sample_batch_size=64 \
    --overrides metrics_per_interaction_step=1 \
    --overrides logging_per_interaction_step=1 \
    --overrides save_buffer_per_interaction_step=null

set_status "FINITE_UPDATE_SMOKE gpu=${gpu}"
"${repo_root}/scripts/smoke_dex4d_m6_wuji.sh" \
    --overrides exp_name=wuji_m6_finite_update_smoke \
    --overrides save_buffer_per_interaction_step=null

set_status "FORMAL_RUNNING gpu=${gpu} envs=1024 transitions=50000896"
"${repo_root}/scripts/run_dex4d_m6_wuji_stage12.sh" \
    --overrides exp_name=wuji_m6_stage12_1024_seed0
set_status "FORMAL_COMPLETE gpu=${gpu}"
