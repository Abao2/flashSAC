# Frozen rollout versus training sampling

Scope: static audit on 2026-09-08; no core edits and no GPU runs by this audit. Selected actor is `bc_std005_eigen`, with arm/hand moving averages both 0.1. Frozen success does not establish that subsequent SAC updates preserve the policy.

## What matches

- `diagnose_str_rollouts.py` native stochastic branch calls the same `_sample_flashsac_actions` used by `FlashSACAgent.sample_actions(training=True)` (`flash_rl/agents/flashSAC/agent.py:223` and `:459`). Both evaluate mean/std with `training=False` (no batch-normalization updates), and apply `tanh(mean + std * noise)` at multiplier 1. This multiplier is not entropy temperature alpha.
- Both use the configured truncated-zeta repetition distribution and one global repetition counter/duration. The Gaussian vector is independent per environment/action dimension, but refresh timing is shared across the vector environment.
- Native counter/cache are not reset on individual episode termination in either path. A newly reset environment can inherit the remaining duration/noise of the current global segment. The harness deliberately preserves this behavior.
- Actor observation slicing, action range and action pipeline are shared. The selected manifest explicitly restores arm/hand filters to 0.1/0.1.

## Important startup differences to test before interpreting warm-start loss

1. `train.py:128` calls `train_env.reset()` with default `random_start_init=True`. `flash_rl/envs/isaaclab.py:307` gets reset observations first, then randomizes `episode_length_buf`. Physical state is freshly reset; returned observation initially still has reset progress. After a step, progress becomes the randomized elapsed count. STR observes `log(episode_length_buf / 10 + 1)` (`simtoolreal/.../utils/obs_utils.py:360`), so this is an input/horizon shift, not merely asynchronous rendering. The frozen harness calls `reset(random_start_init=False)` (`diagnose_str_rollouts.py:398`). This discrepancy affects the initial vector batch; later ordinary automatic episode resets start normally. It is not yet demonstrated to cause failure.
2. `train.py:134-146` always samples a random first action when `transition is None`, even with a loaded checkpoint. The loaded BC actor starts at the second interaction. Frozen rollout acts with BC from the first interaction. The first random action also perturbs the filter's previous-target state.
3. Scratch training uses random actions until replay has its configured minimum; loaded-policy training bypasses that policy-collection condition after the first transition. Frozen evaluation has neither replay warm-up nor optimization. Successful frozen sampling therefore does not test Q initialization or the effect of initial actor/temperature updates.

No training fix is applied here. The optional frozen gate below isolates the combined startup behavior before changing training defaults.

## Interpretation of the independent-noise baseline

`--noise-repeat 1` deliberately refreshes each environment's Gaussian vector every action. It removes temporal persistence while preserving the per-action pre-tanh standard deviation. It is not native training sampling and is a diagnostic only. The custom branch does not draw a zeta duration, whereas the native helper draws candidate noise and duration on every call, even when retaining the cache. Consequently identical seed IDs do not give bitwise matched noise trajectories across these two samplers; compare episode-level statistics, not an assertion of identical random actions.

Changing seed also changes initial-state and simulator random draws. Training uses 1024 environments versus 32 here, and compiled network execution differs from this frozen noncompiled-network loader; neither path promises bitwise replay of a training trajectory. Checkpoint noise caches/counters are initialized afresh, not restored as physical rollout state.

## Prepared baseline manifest

`selected_baseline_manifest.json` contains exactly two serial diagnostic jobs: seed 1/native repetition, and seed 0/independent per-step noise. Both use 128 episodes, 32 environments, max 2500 vector steps, native standard-deviation multiplier 1, identical frozen selected checkpoint, and arm/hand filters 0.1/0.1. CPU argparse and file-path validation passed. Main owns queue/GPU execution.

## Implemented combined startup gate (default remains unchanged)

`diagnose_str_rollouts.py --training-startup` is restricted to Flash/wrapper. It mirrors the two native training resets (before agent construction and before collection), both with randomized initial horizons. Vector step 0 calls the actual batched `action_space.sample()`; actor forward and zeta cache are not advanced. All later actions use the frozen policy, and no subsequent automatic reset adds a random action.

The IsaacLab action-space Box is not explicitly seeded by its creation path. Seeding NumPy globally does not seed this separate Gym generator. The diagnostic therefore records the exact Box RNG state and first actions instead of silently changing its RNG or claiming bitwise equality to an earlier training run. It reproduces the sampling API/distribution, not every setup RNG draw of a different vector size or initialized critic.

`startup_baseline_manifest.json` prepares deterministic and native-stochastic gates, each 128 episodes/32 envs/seed 0 at filters 0.1/0.1. `metadata.json` records randomized raw initial counts, initially returned progress, first-step pre-reset progress and done masks. Dataset rows mark `startup_random_action`; `summary.json` reports `initial_batch` and `subsequent_episodes` separately, since only the first 32 episodes experience startup horizon randomization. Whole-run averages alone can dilute the startup effect.

CPU verification: three tests in `tests/test_str_rollout_startup.py` passed (no actor advance for first random action, actual adapter reset observation/count timing, default-off CLI), plus the existing rollout self-test, compilation, and both manifest argument/path validations. No GPU execution was performed while preparing the gate.
