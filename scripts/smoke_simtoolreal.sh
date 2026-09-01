#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${repo_root}/scripts/run_simtoolreal.sh" \
    --overrides group_name=smoke \
    --overrides exp_name=simtoolreal_adapter \
    --overrides num_env_steps=8192 \
    --overrides num_train_envs=64 \
    --overrides +env.task_cfg_overrides.assets.num_assets_per_type=1 \
    --overrides num_eval_episodes=0 \
    --overrides evaluation_per_interaction_step=null \
    --overrides recording_per_interaction_step=null \
    --overrides save_checkpoint_per_interaction_step=null \
    --overrides agent.buffer_max_length=65_536 \
    --overrides agent.buffer_min_length=1024 \
    --overrides agent.sample_batch_size=512 \
    --overrides agent.use_compile=false \
    --overrides agent.use_amp=false \
    --overrides updates_per_interaction_step=0.125 \
    "$@"
