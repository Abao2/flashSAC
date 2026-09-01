# Dex4D on Isaac Lab with FlashSAC

This repository contains an Isaac Lab `DirectRLEnv` port of Dex4D's official
XArm6 + LEAP AP2AP teacher task and the FlashSAC configurations needed to train
it. The implementation and bounded GPU smoke gates pass on xug6. A full
1,024-environment run has not yet been benchmarked; the smoke results below
only establish functional startup and optimizer updates.

## Source and ownership

The task semantics and asset layout were ported from the official
[Dex4D-Simulation repository](https://github.com/Dex4D/Dex4D-Simulation) at:

```text
0dfccd82797e929cda6ab665003bc27a6b8cd31a
```

The main references are:

- `dex4d_policy/dex4d/tasks/xarm6_leap_hand_ap2ap.py`;
- `dex4d_policy/dex4d/cfg/xarm6_leap_hand_ap2ap_stage_1_2.yaml`;
- `dex4d_policy/dex4d/cfg/xarm6_leap_hand_ap2ap_stage_3.yaml`;
- `dex4d_policy/dex4d/cfg/train_set.yaml`;
- `dex4d_policy/assets/urdf/xarm6_leap_description/xarm6_leap_right_2023.urdf`.

Dex4D-Simulation is Apache-2.0 at that revision. The port records the source
revision in `robotics_tasks/dex4d/data.py`; keep the upstream license and
attribution when redistributing derived code. The official meshes, point-cloud
features, and robot files remain in a separate Dex4D checkout and are not
vendored into this repository.

## Implemented path

```text
train.py
  -> env=dex4d
  -> import robotics_tasks.dex4d (Gym registration)
  -> Dex4DEnv / Dex4DStage12EnvCfg
  -> IsaacLabVectorEnv
       observation: one 1,041D teacher vector for actor and critic
       action:      22 normalized dimensions
  -> FlashSAC replay, updates, checkpoints, and TensorBoard
```

The two registered tasks are:

| Stage | Gym task ID | Object set | Isaac Lab decimation |
|---|---|---|---:|
| 1/2 | `FlashSAC-Dex4D-XArm6Leap-Stage12-Direct-v0` | bottles | 4 |
| 3 | `FlashSAC-Dex4D-XArm6Leap-Direct-v0` | all training categories | 24 |

Isaac Gym used `dt=1/60` with two substeps. The port uses 120 Hz Isaac Lab
physics, so legacy control decimation 2/12 becomes 4/24 without changing the
policy control period. Each episode remains 400 policy steps. Stage 1/2 is 30
Hz; stage 3 is 5 Hz.

The port preserves the task-facing contracts that matter to learning:

- official 22-DOF legacy tensor order and XArm/LEAP position controllers;
- arm-increment and hand-absolute action mapping;
- the 1,041D observation layout, quaternion conversion, clipping, and noise;
- 128 object/goal keypoints and 64D object features;
- bottle filtering for stage 1/2 and code-stratified object/scale assignment;
- reward terms, reset distribution, goal resets, pushes, success latch, and
  the 30-consecutive-step completed-goal rule;
- curriculum and scheduled observation/action/physical randomization;
- tensor-only goal poses. The legacy goal actor was invisible and used a
  separate collision group, so omitting that PhysX body preserves dynamics
  while guaranteeing that it cannot contact the manipulated object;
- 1 mm contact offset and zero rest offset authored into the generated USD
  physics layers. This keeps geometry instanceable while avoiding ineffective
  runtime edits to USD instance proxies.

With the full official manifest, the catalog regression is 2,145 object codes
and 3,200 code-scale pairs. The bottle subset is 418 codes and 735 pairs.

## Assets and cache

Keep the official checkout outside this repository and point `DEX4D_ROOT` at
it. The scripts default to the sibling path `../Dex4D/Dex4D-Simulation`.

```bash
export DEX4D_ROOT=/home/lixiyuan/Dex4D/Dex4D-Simulation
export DEX4D_CACHE_ROOT=/home/lixiaocong/.cache/flashsac/dex4d
```

`DEX4D_ROOT` must contain the manifest listed above and these asset subsets:

- `dex4d_policy/assets/urdf`;
- `dex4d_policy/assets/meshdatav3_scaled`;
- `dex4d_policy/assets/meshdatav3_pc_feat`.

The adapter does not read `datasetv4.1`, `mjcf`, or `textures`.
The cache variable is optional; by default generated files go to
`~/.cache/flashsac/dex4d`. The port creates an importer-safe copy of the robot
URDF, converted USD files, and deterministic keypoint caches there. It never
modifies the official URDF.

The first run for a new object assignment converts its URDFs to USD and builds
deterministic keypoint caches before training begins. Later runs reuse those
files. A long first startup is preprocessing; inspect the cache and importer
logs before treating it as a hung optimizer.

Do not commit the roughly 8.7 GB official Dex4D checkout, generated USD files,
keypoint caches, `runs/`, or `models/` to the private FlashSAC repository. Keep
assets as an external path and record the upstream revision instead. If assets
must live below the worktree, add that directory to `.gitignore` before any
`git add` operation and verify with `git status --short`.

## Validation before training

Use the same Isaac Lab Python executable for tests and training. On xug6 the
known path is shown below; change it if the environment moves.

```bash
cd /home/lixiyuan/flashsac-robotics
export DEX4D_ROOT=/home/lixiyuan/Dex4D/Dex4D-Simulation
export PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python

"$PYTHON_BIN" -m unittest \
  tests.test_dex4d_data \
  tests.test_dex4d_task_math \
  tests.test_external_isaaclab_adapter

bash -n scripts/run_dex4d.sh scripts/run_dex4d_stage12.sh scripts/smoke_dex4d.sh
```

The unit suite checks catalog merging and official counts, object assignment,
feature/keypoint dimensions and caching, non-destructive URDF sanitization,
the complete observation layout, reward equations, action mapping, and the
external Isaac Lab adapter.

Then run the bounded GPU smoke test:

```bash
cd /home/lixiyuan/flashsac-robotics
DEX4D_ROOT=/home/lixiyuan/Dex4D/Dex4D-Simulation \
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python \
./scripts/smoke_dex4d.sh
```

The smoke configuration uses 64 environments, one object prototype, 8,192
transitions, a 16K replay, no AMP/compile, and 0.125 update per interaction.
It is an integration test, not a learning result. Accept it only if all of the
following are true:

1. the startup banner reports `obs=1041`, `actions=22`, and the expected stage;
2. the importer contract finds exactly 22 controlled joints plus `link6` and
   all four fingertip bodies after fixed-joint merging;
3. observations, actions, rewards, critic targets, and losses stay finite;
4. replay sampling and at least one actor/critic update occur;
5. a non-empty TensorBoard event file is written (the smoke script disables
   checkpoint saving intentionally);
6. the process exits cleanly without task-related PhysX, collision, or CUDA
   errors. Headless GLFW/display and `CUDA_VISIBLE_DEVICES` advisory warnings
   from this shared Isaac Sim installation are expected if the run completes.

Validated on xug6 on 2026-09-01:

- 16/16 data, task-math, and adapter tests passed;
- stage 3 completed 128 interactions with one env/one object and wrote finite actor, critic,
  temperature, reward, success, and goal-distance scalars;
- stage 1/2 completed 64 interactions with one env/one object and domain randomization enabled and
  wrote finite optimization/task scalars;
- stage 3 completed an eight-interaction, two-env heterogeneous-object run
  without `object_count`; the banner reported the full 3,200-spec catalog and
  both distinct official assets spawned and stepped cleanly;
- an eight-step regression loaded the baked collision properties without the
  prior instance-proxy collision-property warnings.

These are startup/update checks, not evidence that the policy has learned the
task.

For a lower-memory first check, append explicit Hydra overrides after the
script name:

```bash
./scripts/smoke_dex4d.sh \
  --overrides num_train_envs=1 \
  --overrides num_env_steps=256 \
  --overrides agent.buffer_max_length=512 \
  --overrides agent.buffer_min_length=64 \
  --overrides agent.sample_batch_size=64
```

Do not start a full run merely because the process stays alive; confirm the
environment metrics and optimization metrics are finite and changing.

## Stage 1/2 then stage 3

The intended FlashSAC sequence follows Dex4D's task staging:

```bash
cd /home/lixiyuan/flashsac-robotics
export DEX4D_ROOT=/home/lixiyuan/Dex4D/Dex4D-Simulation
export PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python
export CUDA_VISIBLE_DEVICES=0

./scripts/run_dex4d_stage12.sh
```

The default comparison run uses 1,024 environments, 50,000,896 transitions, a
1M-transition CUDA replay, batch 2,048, AMP, `n_step=1`, and two updates per
interaction. At 1,024 environments, the final interaction/checkpoint step is
48,829. The official PPO stage-1/2 curriculum matures after 15,000 of its
25,000 iterations (60%); this shorter FlashSAC comparison therefore matures
at interaction 29,297, preserving the same budget fraction. Locate the
completed checkpoint under:

```text
models/dex4d/flashsac_stage12_bottle/
  FlashSAC-Dex4D-XArm6Leap-Stage12-Direct-v0/seed0-<timestamp>/step48829
```

Both formal configs save a network/optimizer checkpoint every 4,439
interactions (ten recovery points plus the final checkpoint). The replay
buffer is saved only at step 48,829 because a 1M-entry, 1,041D replay is large.

Start stage 3 from that checkpoint:

```bash
./scripts/run_dex4d.sh \
  --overrides agent_load_path=/absolute/path/to/stage12/step48829
```

Stage 3 fails before environment startup when `agent_load_path` is omitted.
The adapter smoke test explicitly disables this guard because it tests startup
and updates rather than staged learning. Once a checkpoint is loaded, data
collection switches to that policy after the initial transition instead of
repeating the empty-buffer random-action warmup.

`configs/dex4d_flashsac.yaml` intentionally sets
`agent.load_optimizer=false` and `agent.load_reward_normalizer=false`: stage 3
loads network parameters but starts fresh optimizer and reward-normalizer
state, matching Dex4D's network-only stage transfer. For a same-stage resume,
also restore the replay buffer if it was saved and override both flags to
`true`.

## TensorBoard

Logs are written below:

```text
runs/dex4d/<experiment>/<task>_seed<seed>_<timestamp>
```

From a VS Code SSH terminal:

```bash
cd /home/lixiyuan/flashsac-robotics
PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python \
TB_PORT=6009 ./scripts/tensorboard.sh
```

For task learning, inspect:

- `episode/final/success`: fraction of ended episodes that reached the 5 cm
  keypoint-distance threshold at least once;
- `episode/final/completed_goals`: goals held within threshold for 30
  consecutive policy steps;
- `episode/final/goal_distance`, `episode/final/too_far`, and
  `episode/final/time_out`;
- `episode/cumulative/reward/*` and `episode/return`.

For optimizer health, inspect `actor/loss`, `actor/entropy`, `critic/loss`,
`critic/max_entropy_bonus`, `temperature/value`, and `temperature/loss`.
Success should rise while goal distance falls; finite losses alone only show
that the training machinery is running.

## Play a checkpoint

The Dex4D wrapper passes the external Gym registration and task overrides that
the generic Isaac Lab player needs. It also enables a visual-only goal-pose
coordinate frame. The frame has no rigid body or collision API; the policy's
goal remains tensor-only.

```bash
cd /home/lixiyuan/flashsac-robotics
DEX4D_ROOT=/home/lixiyuan/Dex4D/Dex4D-Simulation \
PYTHON_BIN=/home/lixiaocong/venvs/simtoolreal-isaacsim-py311/bin/python \
./scripts/play_dex4d.sh \
  --checkpoint_path /absolute/path/to/checkpoint \
  --num_envs 1 --num_episodes 10
```

For a stage-1/2 checkpoint, prefix the command with
`DEX4D_CONFIG=dex4d_stage12_flashsac`.

## What this experiment does not claim

This is a FlashSAC experiment on a ported Isaac Lab task. It is not a strict
reproduction of Dex4D's published PPO result:

- the official algorithm and checkpoints use PPO in legacy Isaac Gym;
- the official per-stage budget is 25,000 PPO iterations with 8 rollout steps
  and 4,096 environments, or 819.2M transitions per stage;
- the supplied FlashSAC configs use 50,000,896 transitions per stage and 1,024
  environments so algorithm behavior can be evaluated at a practical budget;
- Isaac Lab/Isaac Sim 5 PhysX is not numerically bit-identical to legacy Isaac
  Gym, even when time steps and task equations match.

Implementation boundaries also need to remain visible when interpreting
results:

1. Legacy DOF-force and fingertip-force-sensor tensors are mapped to Isaac
   Lab's applied joint torques and incoming body-joint wrenches. Their shapes
   and task roles match, but the sensor values are not promised bitwise equal.
2. Numeric LEAP joint names are renamed in a cached URDF, fixed joints are
   merged to avoid phantom rigid-body masses, and the virtual palm frame is
   reconstructed from `link6`. Runtime body/joint contract checks guard this.
3. Object surface points are sampled with deterministic seeds and cached before
   farthest-point sampling. This makes runs reproducible, but an old run that
   consumed uncontrolled legacy mesh-sampling RNG may not have identical
   keypoint coordinates.
4. The Isaac Sim converter cache does not detect changes inside URDF-referenced
   OBJ/STL files. Official frozen assets are safe; after modifying a mesh,
   remove that asset's generated USD cache or force reconversion.
5. Validation currently targets Isaac Lab 2.3.2 with Isaac Sim 5.1. Generated
   USD collision properties are stored in the converter's layered physics
   files and the adapter fails fast if that expected layout changes.
6. FlashSAC's generic Gym boundary currently converts actions and observations
   through NumPy. This is correct but introduces host/device copies; benchmark
   throughput at the full environment count before comparing wall-clock speed.

Accordingly, report these runs as “FlashSAC on the Dex4D Isaac Lab port.” A
claim of official Dex4D reproduction requires the official PPO algorithm,
legacy runtime, full two-stage budgets, official evaluation protocol, and an
official checkpoint/result comparison.

## Private GitHub workflow

Keep the framework and lightweight port code in the private repository, with
the official FlashSAC repository retained as `upstream`. After creating an
empty private repository:

```bash
cd /home/lixiyuan/flashsac-robotics
git remote add origin git@github.com:YOUR_ACCOUNT/flashsac-robotics.git
git push -u origin robotics-adapters
```

Before pushing, review `git status --short`, confirm no Dex4D asset/cache paths
are staged, and retain this document plus the exact source revision. Do not
store credentials, machine-specific SSH keys, generated USDs, replay buffers,
TensorBoard events, or checkpoints in Git.
