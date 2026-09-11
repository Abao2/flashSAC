# SimToolReal 论文复现记录

本文记录《SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool
Manipulation》（arXiv:2602.16863）的复现边界、当前机器环境、可执行命令和已完成的
checkpoint 单任务/完整批评测及训练 smoke 结果。除非另有说明，命令均从仓库根目录运行。

记录日期：2026-08-10（Asia/Shanghai）。

## 1. 先区分三个复现对象

| 对象 | 固定标识 | 应当如何解释 |
| --- | --- | --- |
| 论文原始实验 | 论文正文及附录 | 论文结果使用 **Isaac Gym** 训练；真实机器人总结果来自 24 个任务、每任务 5 次，共 120 次试验。论文没有使用后来加入仓库的 Isaac Sim 后端。 |
| 论文时代公开代码 | `664f2d4cdca081b456c4af1129af97f4786e3aa2`（2026-02-24） | 这是接近论文发布时的公开仓库快照，适合审计论文时代的代码和默认值；它不是“论文训练作业一定由此 commit 启动”的 provenance 证明。该版本的评测入口名为 `dextoolbench/eval.py`。 |
| 本次本地复现 | `84058661e297576f0849a25782c6fc49d611338d`（当前 HEAD） | 已包含 Isaac Sim 后端和后续维护改动。本文已经跑通的结果使用其中的 legacy Isaac Gym 入口 `dextoolbench/eval_isaacgym.py`，不是 Isaac Sim。 |

严格复现论文训练时，优先在独立 clone/worktree 中固定到 `664f2d4c`，不要把当前
HEAD 的推荐 Isaac Sim 流程当作论文原始流程。当前 HEAD 与论文时代快照之间至少有以下
会影响解释的变化：

- `isaacgymenvs/launch_training.py` 中 `expl_reward_coef_scale` 从 `0.005`
  改成了 `0.002`；论文 Table I、`664f2d4c` 和下载 checkpoint 的配置均为
  `0.005`。
- 当前 HEAD 为已知 DexToolBench 物体改用离线分解的 collision URDF；论文时代代码
  使用原 URDF，并在需要时走 VHACD。因而本文的当前 HEAD 仿真结果不能声称是旧物理资产
  的逐位复现。
- 当前 HEAD 修改了
  `brush/red_brush/sweep_right` 和
  `screwdriver/long_screwdriver/spin_vertical` 两条轨迹。跑完整 24 项时需一并固定
  commit。
- 当前 HEAD 给程序化训练物体增加了 `numAssetsPerType` 和
  `randomizeAssetOrder`；其默认值 `100`/`True` 保持旧训练行为。

可用以下只读命令复核版本差异：

```bash
git show -s --format='%H %aI %s' 664f2d4c HEAD
git diff 664f2d4c..HEAD -- \
  isaacgymenvs/launch_training.py \
  isaacgymenvs/cfg/task/SimToolReal.yaml \
  isaacgymenvs/tasks/simtoolreal/env.py \
  dextoolbench/trajectories
```

## 2. 固定输入与校验值

`pretrained_policy/` 被 `.gitignore` 排除，不随 commit 固定。开始评测前必须校验实际
下载到本机的两个文件：

```text
4cb0d09833326de6878ee5681c2b14521823fa616e01b135214a51ffe41c63b9  pretrained_policy/model.pth
147717cfdf1d19c06d5d443e428d84f8c2adb0b00abc7c159f6a3fa1ea7becea  pretrained_policy/config.yaml
```

```bash
sha256sum pretrained_policy/model.pth pretrained_policy/config.yaml
```

本机 Isaac Gym 也是被忽略的手动下载内容。用于本次运行的关键二进制指纹为：

```text
447166f3a11439b39c71284f9ba6c3a40e34e29a9f556f549469a439ff2717a5  isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so
5be89032c228c19b3cb0a8e60b90f2524862453afa4d80ebc043bbd2657d73c5  .cache/torch_extensions/gymtorch/gymtorch.so
```

这些 checksum 比仅写“Preview 4”更能约束本机输入，但不能代替 NVIDIA 原始压缩包的
来源记录。新机器的 `gymtorch.so` 是本地编译产物，若编译器或 PyTorch ABI 变化，hash
通常也会变化。

## 3. 本次环境快照

| 项目 | 本次值 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS，kernel `6.8.0-136-generic`，x86_64 |
| CPU / 内存 | AMD Ryzen 9 9950X（16 核 / 32 线程），60 GiB RAM |
| GPU / 驱动 | NVIDIA GeForce RTX 4090，49140 MiB；驱动 `580.173.02` |
| 虚拟环境 | 仓库内 `.venv`，由 uv `0.12.3` 创建 |
| Python | CPython `3.8.20`，Clang `18.1.8` 构建 |
| Isaac Gym | Preview 4，本地 editable 包 `isaacgym==1.0rc4` |
| PyTorch | `torch==2.4.1+cu121`，`torchvision==0.19.1+cu121`；本次审计中 CUDA available 为 `True` |
| RL | 仓库内 editable `rl-games==1.6.1` |
| 数值/配置 | `numpy==1.23.0`，`scipy==1.10.1`，`gym==0.23.1`，`hydra-core==1.3.5`，`omegaconf==2.3.1` |
| 其他关键包 | `wandb==0.12.21`，`trimesh==3.23.5`，`urdfpy==0.0.22`，`warp-lang==0.10.1`，`pytorch3d==0.3.0` |

根目录没有锁定全部传递依赖的 lockfile，且 `pyproject.toml` 中 `torch`、`scipy` 等
依赖没有精确 pin。因此 `uv pip install -e .` 能重建一个可运行环境，但不能保证自动得到
上述逐包相同的环境。安装流程见 [Isaac Gym 安装说明](isaacgym_installation.md)；论文复现
应使用这里的 Python 3.8 / Isaac Gym 环境，而不是 `.venv_isaacsim`。

无需导入 CUDA 库即可检查主要发行包版本：

```bash
.venv/bin/python - <<'PY'
from importlib.metadata import version

for name in (
    "isaacgym", "rl-games", "torch", "torchvision", "numpy", "scipy",
    "gym", "hydra-core", "omegaconf", "wandb", "trimesh", "urdfpy",
    "warp-lang", "pytorch3d",
):
    print(f"{name}=={version(name)}")
PY
```

## 4. 论文训练协议与代码默认值

论文 Table I 的主要设置和仓库配置能够对应如下：

| 项目 | 论文 / 论文时代目标 | 仓库中的对应项 |
| --- | --- | --- |
| 物理 / 控制频率 | 120 Hz / 60 Hz | `sim.dt=1/60`、`substeps=2`，每个 60 Hz 仿真步包含两个物理子步；`controlFrequencyInv=1` |
| 并行环境 | 24576 | launcher 显式覆盖 `task.env.numEnvs=24576`；基础 YAML 的 fallback 只是 8192，不能单独代表论文运行 |
| episode | 600 control steps（10 s） | `episodeLength=600` |
| actor | LSTM 1024 + MLP `[1024,1024,512,512]` | `SimToolRealLSTMAsymmetricPPO.yaml` |
| asymmetric critic | MLP `[1024,1024,512,512]` | `central_value_config.network.mlp` |
| learning rate | `1e-4` | actor 和 central value 均为 `1e-4` |
| rollout / recurrent sequence | 16 / 16 | `horizon_length=16`，`seq_length=16` |
| minibatch / mini-epochs | 98304 / 2 | launcher 和 LSTM asymmetric 配置 |
| SAPG 分块 | `4096 x 6` | 24576 env / `num_blocks=6`，`expl_coef_block_size=4096` |
| discount / GAE | `gamma=0.99` / `tau=0.95` | `SimToolRealPPO.yaml` |
| PPO clip | `0.1` | `e_clip=0.1` |
| entropy exploration scale | `0.005` | `664f2d4c` 和 checkpoint 配置为 `0.005`；当前 HEAD launcher 默认 **`0.002`** |

下载的 `pretrained_policy/config.yaml` 是 checkpoint 随附的训练配置快照，不应被当作
“从零训练唯一配方”。例如它记录 `forceScale=2`，而论文时代和当前 launcher 默认均为
`20`；它还保留了指向作者机器上前序 checkpoint 的绝对 `checkpoint` 路径，表明该文件
描述的是一次具体 finetune 运行。其他可见差异包括：

| 设置 | 基础/launcher | 下载 checkpoint 配置 | 本文单任务评测 |
| --- | --- | --- | --- |
| `forceScale` | `20` | `2` | 覆盖为 `0` |
| `torqueScale` | `2` | 旧配置未记录该键 | 合并当前默认后覆盖为 `0` |
| `resetWhenDropped` | 基础值 `False` | `True` | 覆盖为 `False` |
| `jointVelocityObsNoiseStd` | 基础值 `0.1` | `0.01` | 保存后的 `env_cfg.yaml` 为 `0.01` |
| 训练目标区域 | `[-.35,-.2,.6]` 至 `[.35,.2,.95]` | `[-.35,-.1,.68]` 至 `[.35,.2,1.05]` | 使用固定任务轨迹，区域采样不参与 |
| `expl_reward_coef_scale` | `664f2d4c`: `0.005`；HEAD: `0.002` | `0.005` | 只推理，不参与策略更新 |

论文完整训练曲线约为 **120B environment steps**。按 `24576 * 16 = 393216`
environment steps/update 粗略换算，是约 30.5 万次 update，而不是一次短时 smoke test。
论文消融实验的量级约为每个设置 9B steps、5 个 seeds。复现报告应记录实际停止步数、seed、
GPU 数量和 wall-clock time，不能仅以“训练脚本成功启动”宣称复现训练结果。

若要审计论文时代 launcher，可在独立工作树中运行：

```bash
git worktree add ../simtoolreal-paper-era 664f2d4c
cd ../simtoolreal-paper-era
# 按该提交的 docs/installation.md 建立 Python 3.8 + Isaac Gym Preview 4 环境，
# 再准备相同 checksum 的 checkpoint（若做 finetune）并设置自己的 W&B entity。
python isaacgymenvs/launch_training.py \
  --custom-experiment-name paper_reproduction \
  --wandb-entity YOUR_ENTITY
```

上述命令会创建一个长期训练作业；它没有自动在论文曲线的 120B steps 处停止。输出目录由
launcher 组织在 `train_dir/<project>/<group>/<experiment>/` 下。

## 5. 已跑通：当前 HEAD 的单任务 checkpoint 评测

### 5.1 运行命令

本次使用的是当前 HEAD、Isaac Gym、`claw_hammer/swing_down`、10 episodes。环境变量和
实际命令如下；未设置 `DISPLAY`，没有额外 `timeout`，并保留 evaluator 的默认
`force_table_urdf=True`。

```bash
export PATH="$PWD/.venv/bin:$PATH"
SIMTOOLREAL_PYTHON_LIBDIR="$($PWD/.venv/bin/python -c \
  'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export LD_LIBRARY_PATH="$SIMTOOLREAL_PYTHON_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions"

.venv/bin/python dextoolbench/eval_isaacgym.py \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down \
  --config-path pretrained_policy/config.yaml \
  --checkpoint-path pretrained_policy/model.pth \
  --output-dir evals/reproduction/claw_hammer/swing_down \
  --num-episodes 10 \
  --policy-name pretrained_policy
```

重跑会覆盖该目录中的 JSON/YAML；如需保留本次证据，应先复制目录或换一个新的
`--output-dir`。

### 5.2 本次结果

结果文件：

```text
evals/reproduction/claw_hammer/swing_down/eval.json
evals/reproduction/claw_hammer/swing_down/policy_config.yaml
evals/reproduction/claw_hammer/swing_down/env_cfg.yaml
```

汇总值：

```text
episodes:       10
avg_goal_pct:   100.0
avg_time_sec:   9.931666666666667
episode_steps:  [620, 601, 562, 607, 614, 589, 563, 556, 638, 609]
```

当前保存产物的 SHA-256（重跑产生不同 rollout 时，`eval.json` hash 可能不同）：

```text
ccbb3a37483aa5305c0b60aec0ccbbd592ecb7956f1f00e68894bd463c3b9fd1  eval.json
547d7e63168ffa3a285450b7a38d1d85f1c869e9265b0ccbbc65553dacc60297  policy_config.yaml
6f224b6d4bd3f6c9ce7183131f4becec757ce309a37b46f482ddfcbab8b03e89  env_cfg.yaml
```

```bash
sha256sum evals/reproduction/claw_hammer/swing_down/*
```

`env_cfg.yaml` 是合并 checkpoint 配置、当前 HEAD 默认值和 evaluator overrides 后的最终
环境配置；它比只查看 `pretrained_policy/config.yaml` 更准确地描述本次 rollout。

### 5.3 这个 100% 实际表示什么

评测脚本的 `avg_goal_pct` 是每个 episode 按顺序到达的轨迹 waypoint 数占总 waypoint
数的比例，不是“真实任务二元成功率”。本次 evaluator 还做了以下设置：

- 单环境、deterministic policy action、固定起始姿态和固定目标轨迹；
- 关闭 reset pose 随机化、action/observation/object-state delay，以及物体尺度噪声；
- 外力、力矩和线/角速度 impulse 全部置零；
- `resetWhenDropped=False`，`successSteps=1`；
- `evalSuccessTolerance=0.01`、`keypointScale=1.5`，代码以固定尺寸物体的“最远
  keypoint 距离”不超过二者乘积作为 waypoint 判据；
- 默认 `force_table_urdf=True`，没有按 hammer/marker/spatula 类别切换 nail、
  whiteboard、plate 等 task fixture。

因此该结果证明的是：在本机当前 HEAD 的 legacy Isaac Gym 环境中，给定 checkpoint 能
完成这一条无扰动固定轨迹。它不是论文真实机器人任务成功率的单项复刻。

Isaac Gym 有时会在 `eval.json` 已完整写出后于进程 teardown 阶段 segfault（exit 139 或
-11）。完整批处理脚本已按“结果文件存在”判断这类情况；单任务运行也应先检查上述三个
文件，再判断是否需要重跑。

## 6. 当前 HEAD 的完整仿真批处理

```bash
.venv/bin/python dextoolbench/run_all_evals_isaacgym.py
```

当前脚本枚举 6 类工具 × 2 个物体 × 2 个任务 = 24 个组合，每个组合
`NUM_EPISODES=10`，共 240 个**仿真** episodes，输出到：

```text
evals/<timestamp>/<category>/<object>/<task>/pretrained_policy/eval.json
```

`evals/` 被 `.gitignore` 排除，应另外归档结果、最终 `env_cfg.yaml`、checkpoint checksum、
commit、driver 和随机 seed。若目标是旧快照的 24 项结果，应从 `664f2d4c` 使用当时入口
`dextoolbench/run_all_evals.py`，不要混用当前两条已修改轨迹和新 collision assets。

### 6.1 本次完整批评测结果

本次运行目录为 `evals/2026-08-10_12-07-47`。端到端 wall-clock 时间为
`1:04:34`；24 个组合全部产生可解析结果，共 240 episodes：

```bash
.venv/bin/python -m dextoolbench.summarize_evals \
  evals/2026-08-10_12-07-47
```

```text
Expected: 24 | OK: 24 | Missing: 0 | Failed: 0
```

| 类别 | 完成组合 | 平均轨迹进度 (%) | 平均 episode 仿真时长 (s) |
| --- | ---: | ---: | ---: |
| hammer | 4/4 | 97.188 | 10.361 |
| spatula | 4/4 | 97.750 | 15.268 |
| eraser | 4/4 | 93.831 | 11.575 |
| screwdriver | 4/4 | 92.599 | 27.300 |
| marker | 4/4 | 100.000 | 7.519 |
| brush | 4/4 | 94.405 | 21.358 |
| **总体** | **24/24** | **95.962** | **15.563** |

这里的 `avg_time_sec` 是 episode 步数除以 60 Hz 控制频率，不是程序 wall-clock 时间。
汇总工具会逐个校验 `eval.json` 的数值范围、episode 数组长度和目标进度均值；缺失或损坏
结果默认返回非零退出码。它还会打印全部 24 项明细，本表只保留类别与总体均值。

## 7. 已跑通：从零训练的一轮 GPU smoke

这个 smoke 只验证环境创建、SAPG 数据路径、LSTM actor、非对称 critic、两个 optimizer、
checkpoint 保存和有限训练退出；它不用于衡量策略质量，也不能替代论文约 120B steps 的训练。
本次在当前 HEAD 上使用 6 个环境、每类 1 个程序化资产、rollout/sequence length 16：

```bash
source .venv/bin/activate
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export WANDB_MODE=disabled

python -m isaacgymenvs.train \
  task=SimToolRealLSTMAsymmetric \
  checkpoint='' \
  headless=True force_render=False capture_video=False \
  pipeline=gpu sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 \
  multi_gpu=False \
  ++task.env.useSparseReward=False \
  task.env.numEnvs=6 task.env.numAssetsPerType=1 \
  task.env.capture_video=False task.env.goodResetBoundary=0 \
  task.env.objectScaleNoiseMultiplierRange='[0.9,1.1]' \
  task.env.forceConsecutiveNearGoalSteps=True \
  task.env.forceScale=20 task.env.torqueScale=2.0 \
  task.env.objectAngVelPenaltyScale=0.0 \
  train.params.config.max_epochs=1 \
  train.params.config.horizon_length=16 \
  train.params.config.seq_length=16 \
  train.params.config.minibatch_size=96 \
  train.params.config.mini_epochs=1 \
  train.params.config.central_value_config.minibatch_size=96 \
  train.params.config.central_value_config.mini_epochs=1 \
  train.params.config.good_reset_boundary=0 \
  train.params.config.use_others_experience=lf \
  train.params.config.off_policy_ratio=1.0 \
  train.params.config.expl_type=mixed_expl_learn_param \
  train.params.config.expl_reward_type=entropy \
  train.params.config.expl_coef_block_size=1 \
  train.params.config.expl_reward_coef_scale=0.005 \
  train.params.network.space.continuous.fixed_sigma=coef_cond \
  wandb_activate=False seed=0 \
  experiment=00_smoke_sapg_verify_gpu \
  hydra.run.dir=./train_dir/smoke_sapg_verify_gpu
```

6 个环境除以 6 个 SAPG blocks 得到 block size 1；on-policy batch 为
`6 * 16 = 96`，所以 actor 和 central critic 的 minibatch 均设为 96。实测结果：

```text
GPU device count:       1
actor minibatches:      1
play / update / epoch:  0.731 s / 0.123 s / 0.854 s
epoch:                  1 / 1
process exit code:      0
checkpoint:             train_dir/smoke_sapg_verify_gpu/runs/
                        00_smoke_sapg_verify_gpu/nn/
                        last_00_smoke_sapg_verify_gpu_ep_1_rew_-inf.pth
```

rollout 只有 16 步，而 episode 长 600 步，所以
`Max epochs reached before any env terminated at least once` 是预期警告。当前 HEAD 原训练入口
还会把 rl_games 正常完成时返回的 `(last_mean_reward, epoch)` 元组误当作 PBT 的
`(new_cfg, vec_env)` 重启请求，在 checkpoint 已写出后触发 `AttributeError`。本地
`isaacgymenvs/train.py` 已改为仅在元组首项为 `DictConfig` 时重启；上述 smoke 验证修复后
正常退出。顶层 `max_iterations` 在这条代码路径中不起停止作用，有限 smoke 必须覆盖
`train.params.config.max_epochs`。

## 8. 为什么不能把仿真结果直接与论文 79.26% 对比

论文报告的总均值 `79.26%` 来自真实机器人上的 24 任务 × 5 trials，论文文字使用 2 cm
判据。当前仓库的 Isaac Gym 批评测则是 24 × 10 仿真 episodes，并采用
`evalSuccessTolerance=0.01` 再结合 `keypointScale=1.5` 的最远 keypoint 判据；此外它
默认使用普通桌面、固定轨迹、零外扰和不同的 collision 资产。二者在平台、重复次数、
成功定义和场景交互上均不一致。

合理的验收层级是：

1. **环境复现**：版本和 checksum 匹配，Isaac Gym 能创建环境并加载 checkpoint。
2. **checkpoint 仿真复现**：固定 commit，跑完 24 × 10，报告每任务轨迹进度及统计波动。
3. **论文训练复现**：按论文时代代码和 `0.005` exploration scale 完成长训练预算，并报告
   多 seed 曲线。
4. **论文真实评测复现**：搭建论文机器人、感知和任务 fixture，严格执行 24 × 5 及论文
   2 cm 判据后，才与 `79.26%` 做同口径比较。

GPU PhysX、驱动和并行归约并不保证跨机器逐位确定；因此“数值接近及置信区间”比要求每个
episode 的步数完全相同更合适。
