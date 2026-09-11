#!/usr/bin/env bash
# One native Isaac GUI at a time. No training and no replay allocation.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
str_root="${SIMTOOLREAL_ROOT:-${repo}/third_party/simtoolreal}"
play_python="${STR_PLAY_PYTHON:-/home/abao/play2perfect/.venv_isaacsim/bin/python}"
mode="${1:-}"
if [[ "$mode" != best && "$mode" != official ]]; then
    echo '用法: bash scripts/play_str_behavior.sh best|official [--print-command]' >&2
    exit 2
fi
if [[ "${2:-}" != '' && "${2:-}" != --print-command ]]; then
    echo '第二个参数只能为 --print-command。' >&2
    exit 2
fi
[[ -x "$play_python" ]] || { echo "缺少 Python: $play_python" >&2; exit 2; }
export PYTHONPATH="$str_root/rl_games:$repo:$str_root${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
cd "$str_root"
if [[ "$mode" == best ]]; then
    checkpoint="$repo/diagnostics/credit_nstep3_continue_20260910/candidate/models/seed0/step195320"
    [[ -f "$checkpoint/actor.pt" ]] || { echo "缺少 checkpoint: $checkpoint" >&2; exit 2; }
    echo 'FlashSAC: full-STR n3 500M；本轮100M/500M同起点对照中暂选，不代表已收敛或全checkpoint最优。'
    echo 'KUKA+Sharpa，随机初始状态/目标链，无DR，arm EMA=1 / hand EMA=0.1；固定 tolerance=0.02905653603374958。'
    echo '确定性策略、1个环境；不是完整50目标成功保证。关闭窗口即可停止。'
    command=("$play_python" "$repo/play_isaaclab.py"
        --config_path "$repo/configs" --config_name simtoolreal_full_arm1_nstep3
        --checkpoint_path "$checkpoint" --num_envs 1 --num_episodes 1000 --real_time
        --overrides agent.buffer_max_length=1 --overrides agent.buffer_min_length=1
        --overrides agent.sample_batch_size=1 --overrides agent.use_compile=false
        --overrides agent.load_optimizer=false --overrides agent.load_reward_normalizer=false
        --overrides ++env.task_cfg_overrides.termination.success_tolerance=0.02905653603374958
        --overrides ++env.task_cfg_overrides.termination.target_success_tolerance=0.02905653603374958
        --overrides ++env.task_cfg_overrides.termination.eval_success_tolerance=null
        --overrides ++env.task_cfg_overrides.termination.tolerance_curriculum_interval=1152921504606846976)
else
    [[ -f "$str_root/pretrained_policy/model.pth" ]] || { echo '缺少官方 checkpoint。' >&2; exit 2; }
    echo '官方 STR pretrained: 原生 PPO/SAPG RlPlayer（保留LSTM/归一化），claw_hammer / swing_down 完整轨迹。'
    echo '使用现有 Isaac Lab DexToolBench 播放入口，不是 legacy Gym；与 Flash 随机训练目标不是同协议成绩对比。'
    echo '确定性策略、1个环境；关闭窗口即可停止。'
    command=("$play_python" "$str_root/dextoolbench/eval_isaacsim.py"
        --object_category hammer --object_name claw_hammer --task_name swing_down
        --config_path "$str_root/pretrained_policy/config.yaml"
        --checkpoint_path "$str_root/pretrained_policy/model.pth"
        --num_episodes 1000 --play)
fi
if [[ "${2:-}" == --print-command ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo 'NVIDIA 驱动不可用，未启动 Isaac。请先运行 nvidia-smi 查看原因。' >&2
    echo '2026-09-11 已发现系统自动升级后的驱动/库版本冲突；脚本不会自动重启或修改驱动。' >&2
    exit 3
fi
if ! "$play_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA 不可用"'; then
    echo 'CUDA 预检失败，未启动 Isaac。' >&2
    exit 3
fi
if [[ -z "${DISPLAY:-}" ]]; then
    echo 'DISPLAY 未设置。请在 abao 本机图形桌面的终端运行。' >&2
    exit 3
fi
exec "${command[@]}"



