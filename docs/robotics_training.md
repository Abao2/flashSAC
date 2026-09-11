# FlashSAC robotics training

This fork keeps the official FlashSAC implementation as `upstream` and adds
task adapters in this repository. The SimToolReal source/assets stay in their
own repository; no reward, curriculum, observation, action, or controller code
is copied or modified.

## What is ready

The SimToolReal Isaac Lab backend is end-to-end tested. Its data path is:

```text
train.py
  -> env=simtoolreal
  -> external Gym registration import (isaacsimenvs)
  -> registered task YAML overlay
  -> IsaacLabVectorEnv
       actor:  policy observation, 140 dimensions
       critic: privileged critic state, 162 dimensions
       replay: policy + critic state, 302 dimensions
       action: normalized [-1, 1], 29 dimensions
  -> FlashSAC replay and updates
```

The adapter also:

- preserves task extras and writes reward terms, goal count, full-task success,
  termination causes, actor loss, and critic loss to TensorBoard;
- bootstraps Isaac Lab timeouts from the task-provided pre-reset final
  observation, while true failure terminations do not bootstrap;
- allows task packages and task YAML overlays to be selected from config, so a
  second Isaac Lab task does not require another algorithm fork;
- runs from the task repository so its relative assets resolve, while all
  FlashSAC logs/checkpoints remain under this repository.

## xug6 paths and commands

Framework:

```text
/home/lixiyuan/flashsac-robotics
```

SimToolReal task/assets:

```text
/home/lixiyuan/simtoolreal
```

Run the short, non-reproduction smoke test on physical GPU 0:

```bash
cd /home/lixiyuan/flashsac-robotics
CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python ./scripts/smoke_simtoolreal.sh
```

The smoke test uses 64 envs and only one procedural asset per type to validate
the adapter, replay, gradients, and logging quickly. It is not a result run.

Run the deterministic 1.05M-transition learning diagnostic:

```bash
cd /home/lixiyuan/flashsac-robotics
SIMTOOLREAL_CONFIG_NAME=simtoolreal_fixed_debug CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python ./scripts/run_simtoolreal.sh
```

This fixes the robot, one seed-42 eraser asset, initial state, and trajectory 0,
disables DR/delays, and keeps the paper's ratio of two updates per 1,024 new
transitions. It is a learning-chain diagnostic, not a paper reproduction.

Run the paper-scale FlashSAC configuration:

```bash
cd /home/lixiyuan/flashsac-robotics
CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python ./scripts/run_simtoolreal.sh
```

That configuration uses 1,024 envs, 50,000,896 environment transitions, a 10M
CUDA replay, batch 2,048, two updates per interaction, asymmetric observations,
AMP, and `n_step=1` as reported in the paper. The current upstream Isaac Lab
shell script instead uses `n_step=3`; select it explicitly with:

```bash
./scripts/run_simtoolreal.sh --overrides n_step=3
```

The 10M replay is expected to consume about 24 GB because each replay
observation stores the 140D policy input plus the 162D critic state. The critic
network itself consumes only the 162D state. Check GPU availability before a full run. In-process periodic evaluation
is disabled because Isaac Lab would reuse and reset the stateful training env;
evaluate saved checkpoints in a separate process.

## TensorBoard and outputs

Training events are under:

```text
/home/lixiyuan/flashsac-robotics/runs/<group>/<experiment>/<task>_seed<seed>_<timestamp>
```

Checkpoints are under:

```text
/home/lixiyuan/flashsac-robotics/models/<group>/<experiment>/<task>/seed<seed>-<timestamp>
```

From the VS Code SSH terminal on xug6:

```bash
cd /home/lixiyuan/flashsac-robotics
PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python TB_PORT=6009 ./scripts/tensorboard.sh
```

The most useful tags are:

- `episode/final/successes`: number of completed trajectory goals;
- `episode/final/all_goals_hit`: whole-task success rate;
- `episode/final/done_*`: failure/timeout composition;
- `episode/return` and `episode/cumulative/*`: reward and reward terms;
- `actor/loss`, `critic/loss`, `actor/entropy`, and `temperature/value`:
  FlashSAC optimization health.

## Adding another Isaac Lab task

Copy `configs/env/external_isaaclab_template.yaml` and set:

1. the registered Gym task ID;
2. the Python package whose import performs registration;
3. explicit normalized action bounds;
4. the registered YAML key, if the task has a YAML overlay;
5. the nested success metric path;
6. `agent.asymmetric_observation=true` if the task returns a `critic` state.

## Dex4D Isaac Lab + FlashSAC

The DirectRLEnv port is implemented under `robotics_tasks/dex4d`. It registers
the bottle-only stage-1/2 task and the all-category stage-3 task, both using the
official external Dex4D assets. Run the short adapter smoke test with:

```bash
cd /home/lixiyuan/flashsac-robotics
DEX4D_ROOT=/home/lixiyuan/Dex4D/Dex4D-Simulation CUDA_VISIBLE_DEVICES=0 \
  PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python \
  ./scripts/smoke_dex4d.sh
```

The intended training order is stage 1/2, then stage 3 initialized from the
stage-1/2 network checkpoint:

```bash
DEX4D_ROOT=/home/lixiyuan/Dex4D/Dex4D-Simulation CUDA_VISIBLE_DEVICES=0 \
  PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python \
  ./scripts/run_dex4d_stage12.sh

DEX4D_ROOT=/home/lixiyuan/Dex4D/Dex4D-Simulation CUDA_VISIBLE_DEVICES=0 \
  PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python \
  ./scripts/run_dex4d.sh --overrides agent_load_path=/absolute/path/to/stage12/step48829
```

The supplied 50,000,896-transition FlashSAC budgets are controlled algorithm
experiments; they are not reproductions of Dex4D's official PPO results or
819.2M-transition per-stage PPO budgets. See the
[Dex4D port guide](dex4d_isaaclab_port.md) for provenance, exact task contract,
asset setup, validation gates, known parity limits, checkpoint transfer, and
TensorBoard tags.

## Private GitHub setup

The local branch is `robotics-adapters`; the official repository is the
`upstream` remote. After creating an empty private repository, add it as
`origin` and push:

```bash
git remote add origin git@github.com:YOUR_ACCOUNT/flashsac-robotics.git
git push -u origin robotics-adapters
```

Keep `upstream` so future official FlashSAC changes can be fetched and reviewed
without mixing task repositories into this one.
