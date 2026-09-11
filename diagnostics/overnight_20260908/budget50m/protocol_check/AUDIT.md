# 50M hand-filter0.1: frozen protocol matrix

This preparation launched no GPU process and changed no training core. The follow-up adds a default-off precision flag to the rollout harness as documented below; its existing default policy/reset behavior is preserved. `manifest.json` is ready for main's review, with the existing entropy queue as the `--after-status` dependency. The standard hard stop remains 2026-09-09 09:00 Asia/Shanghai.

## Exact checkpoint and comparison

Checkpoint: `budget50m/models/arm1_hand01/seed0/step48830`, trained for 50,001,920 transitions from scratch. This is **not** a BC warm-start checkpoint. Arm smoothing is **1.0**, hand smoothing **0.1**.

The existing clean32/native-stochastic result is reused: 0/128 final-goal success, 128/128 sticky task-lift, 127/128 later below-reset drops, all episodes length600. This does not agree with the high late training metric reported by main, and the protocol matrix is intended to localize that discrepancy—not to assume a cause.

| Environment count | Normal clean start | Training-startup gate |
| --- | --- | --- |
| 32 | Existing seed0/128 episodes, reused | New seed0/128 episodes, max2500 steps |
| 1024 | New seed0/2048 episodes, max1300 steps | New seed0/2048 episodes, max1300 steps |

The fourth job, clean32/seed1/128 episodes, is an optional robustness check after the original three matrix jobs. Main may remove that job before launch if only the strict matrix is desired. The queue otherwise executes all listed jobs, including this one; the metadata flag does not disable execution. Two precision-bundle jobs are now appended after those unchanged first four jobs.

1024-env jobs use `--no-dataset` to avoid buffering/compressing up to1.3M transition rows. Episode JSONL, aggregate metrics, startup state/actions and initial-versus-subsequent cohort statistics remain available. This changes recording only. Startup affects the first vector batch; later automatic episode resets are ordinary. It reproduces randomized initial horizons plus the first random action, not a saved live training simulator state or ongoing optimizer updates.

## Verified on CPU

- The actor's complete tensor digest matches both `model_digest_before` and `model_digest_after` of the existing final stochastic rollout. All six checkpoint files are hashed in `validation.json`.
- The saved training config and completed evaluation metadata have **exactly equal** `env.task_cfg_overrides`: same fixed eraser, initial pose, goal, observation fields, reset-noise settings, DR flags, tolerances and filters. Environment registration, action bounds, timeout-bootstrap contract and success-path fields also match.
- Dataset has76800 rows and162 actor observation dimensions. Observed eraser dimensions are `[0.1460571438,0.0565996952,0.0499329232]` metres, matching the configured fixed-eraser construction. Initial observed progress is0. Dataset geometric columns are post-action; do not mistake their first row for the pre-action reset geometry.
- On1024 saved states from the actual failed final rollout, `.train()` versus `.eval()` with explicit `training=False` gives bitwise-identical actor means and standard deviations; weights and BN buffers are unchanged.
- On CPU, predicting the same first32 states as a32-state batch versus inside a1024-state batch gives maximum mean/std difference0. This does **not** prove CUDA or physics batch-size invariance.

## Model mode, compilation and numeric settings

`UnitBatchNorm.forward` uses its explicit `training` argument in `F.batch_norm`; it does not use `self.training`. Both native environment-action sampling and the frozen harness call `get_mean_and_std(..., training=False)`. The actor has no dropout. Therefore the harness's `.eval()` is not, by itself, a demonstrated BN-mode mismatch.

Native `Network.apply` calls the named `get_mean_and_std` method through `getattr`, rather than necessarily using the compiled `forward`. A CPU `OptimizedModule` inspection confirmed that this named method is bound to the original actor. Thus `use_compile=true` versus false does not automatically establish different compiled actor-inference computation. CUDA kernels and the compiled zeta helper were not tested here.

A real settings difference remains: `train.py` explicitly enables TF32 and requests float32 matmul precision `high`; the harness does not explicitly set those options. Existing metadata does not capture the final global flags after Isaac initialization. This is a candidate numeric-path difference to measure, **not an established explanation** for the success gap. Actor/critic training updates also use AMP, but frozen environment-action sampling does not enter that update autocast block.

### Default-off precision-bundle follow-up

The new explicit `--training-numerics` flag sets the same four settings as `train.py`, before environment creation: CUDA matmul TF32 enabled, cuDNN TF32 enabled, cuDNN benchmark enabled, float32 matmul precision `high`. It does not change compilation, actor inference `training=False`, observations, controls, reset or sampling. Without the flag, it sets none of these values.

Two additional clean-start/native-stochastic jobs use this flag: `env32_clean_tf32_seed0` (128 episodes) and `env1024_clean_tf32_seed0` (2048 episodes, no dataset). Their default counterparts are the existing clean32 seed0 result and the new clean1024 seed0 matrix cell. This is a **precision bundle intervention**, not a claim that one particular flag explains the difference.

Every new harness run, including defaults, now records actual settings under `torch_numerics_after_env_init` **after both Isaac environment reset and frozen policy creation/loading**, then under `torch_numerics_after_rollout` at the end. These are effective flags, not just requested values. Historical completed runs lack those fields and are not retroactively assigned invented values. `validation.json` preserves the old harness hash and records the intentional default-off extension's new hash, exact preservation of the first four job objects, six-job CLI/path checks, four passing CPU unit tests and the original self-test. No GPU test or launch was performed during this change.

## Asset and scene evidence, and remaining limit

Both saved logs report one generated object URDF, the same KUKA+Sharpa USD basename, the same70 adjacent self-collision filtered pairs across30 robot bodies, and environment spacing1.2m. The procedural object generator uses default seed42 with `num_per_type=1`, independent of environment count, so this is not selecting32 versus1024 different object identities. Scene origins and physics workload do differ when vector size changes.

The existing metadata does not retain a complete generated-USD hash manifest from both original processes. Therefore this audit can verify the shared config, generator and observed dimensions, but **cannot claim bitwise equality of all imported physics assets**. Relevant source/URDF hashes are recorded now, not retroactively proven at historical launch time.

All protocol argv, checkpoints, output paths and the existing queue dependency were checked on CPU. New output directories are absent and will not overwrite the original completed evaluation. Read `validation.json` for exact evidence and hashes.
