# Labeled stochastic teacher data: memory and BC diagnostics

Dataset: `../official_filter01_labeled128/transitions.npz`, 8,139 transitions from 128 successful episodes. Initial object/robot/goal setup is fixed; teacher action sampling varies. Arm and hand moving-average coefficients are both **0.1**. This is not random-initial-state or object generalization.

`actions` are actually executed stochastic actions. `teacher_actions` are the deterministic clipped mean captured from the **same** original policy forward and RNN state, without a second forward. Their aggregate difference is large: action RMSE **0.91055**. Consequently BC uses `--bc-action-key teacher_actions`; dynamics prediction still conditions on actual executed `actions`.

All conditions: seed 0, 3,000 updates, batch 512, width 128, history window 8 (about 117 ms of past observations at 60 Hz), whole-episode split 96 train episodes / 6,139 rows and 32 validation episodes / 2,000 rows. True history and repeat-current controls have identical parameter counts. Normalizers are fit only on train rows. Prediction targets are next object-velocity delta, goal-relative keypoint delta and raw immediate reward.

## Held-out prediction

| Input | Train normalized MSE | Validation normalized MSE | Keypoint-delta RMSE | Reward RMSE |
| --- | ---: | ---: | ---: | ---: |
| Current | 0.08813 | 0.14065 | 0.443 mm | 28.92 |
| Repeat current | 0.06252 | 0.13950 | 0.435 mm | 28.15 |
| Real history | 0.04047 | 0.18557 | 0.525 mm | 26.88 |

The train-mean baseline validation normalized MSE is 0.96604. Current observations and current executed actions contain substantial predictive information. History fits training data better but worsens held-out aggregate dynamics prediction; a modest reward prediction gain is not a general history advantage. Near-object proxy rows are harder for all three models (normalized MSE 0.501 / 0.499 / 0.677); these labels are geometric proxies, not verified contact states.

This does not prove the observations are fully Markov, nor rule out longer history or another learning setup. It provides no positive evidence that missing short history is the main bottleneck here.

## Behavior cloning of teacher means

Native FlashSACActor, two residual blocks, width 128, deterministic tanh(mean) target, native UnitBatchNorm and parameter projection; no external BC input standardization. The std head remains untrained.

| Input | Parameters | Train action MSE | Validation action MSE | Validation action RMSE |
| --- | ---: | ---: | ---: | ---: |
| Current | 293,374 | 0.002614 | 0.044686 | 0.21139 |
| Repeat current | 440,794 | 0.003340 | 0.041151 | 0.20286 |
| Real history | 440,794 | 0.002974 | 0.044116 | 0.21004 |

The train-mean-action baseline validation MSE is 0.46767. These students learn useful imitation, but show a substantial episode-held-out fitting gap. History does not beat the matched-size repeat-current control. The near-goal-center proxy has the largest BC validation error (about 0.10 MSE), identifying a place to inspect closed-loop drift rather than proving a particular cause.

Checkpoints `bc_current.pt`, `bc_repeat.pt`, `bc_history.pt` are ready for deterministic closed-loop evaluation with **both filters 0.1**. Offline errors are not success rates. The smaller prior deterministic-data current-state BC lifted in 32/32 repeated fixed-start trials but later dropped in all 32 and achieved 0 goals; that separate result is not a result for these new checkpoints.

All six offline conditions were CPU-only and completed in approximately 66 seconds of fitting/evaluation. Exact metrics, phase counts, split IDs, configuration, data hash and source metadata are in `summary.json` and the checkpoints.

## Subsequent closed-loop results — independently audited

All three new checkpoints subsequently completed the actual single-goal task in **64/64 deterministic fixed-start episodes**, with both action filters 0.1. These are repeated trials of one configuration, not 64 independently sampled initial states.

| Student | Actual goal successes | Mean policy steps | Mean raw return | Mean final position error | Mean final maximum keypoint error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 64/64 | 62.00 | 1388.04 | 11.87 mm | 16.14 mm |
| Repeat current | 64/64 | 71.67 | 1394.75 | 9.01 mm | 10.20 mm |
| Real history | 64/64 | 60.00 | 1386.76 | 19.01 mm | 24.65 mm |

Sources: `../bc_current_full_eval`, `../bc_repeat_full_eval`, `../bc_history_full_eval`, each containing `summary.json`, `episodes.jsonl`, `metadata.json` and `transitions.npz`.

### Why this is actual goal success, not a geometric proxy

The harness records `step_info['episode_final']['all_goals_hit']` on episode termination. STR computes that field from `_successes >= max_consecutive_successes`, here 1. `_successes` increments after the goal counter reaches 10; the counter uses maximum fixed-size keypoint distance <= `0.02 * 1.5 = 0.03 m`. The configured counter is cumulative, not necessarily consecutive.

An independent CPU audit reconstructed reward keypoints from the saved post-action/pre-reset observation, object quaternion and fixed reward offsets, using a separate explicit rotation-matrix implementation. Every episode in all three evaluations had exactly **10** qualifying steps. All 64 terminal rows per evaluation were `terminated=True`, `truncated=False`, with `successes` observation `log(1 + 1) = 0.693147`; this observation is log-encoded and must not be read as a raw success count.

The largest terminal maximum-keypoint error was 16.21 mm (current), 11.48 mm (repeat) and 25.41 mm (history), all below 30 mm. Independent reconstructed errors agree with logged errors within 9e-10 m. Every episode received exactly **+1000 goal bonus and +300 lift bonus**. Thus neither lift flags nor near-palm proxies explain the success count.

For current-state BC, mean reward components were +26.711 fingertip progress, +57.247 pre-lift height, +300 lift bonus, +8.978 keypoint progress, -1.447 arm velocity penalty, -3.449 hand velocity penalty, +1000 goal bonus. Their sum agrees with total return within 3.7e-5 for each episode. No dropped-below-reset event occurred before success; retention after successful termination was not tested.

### Why the current student really uses only the latest 162-vector

The checkpoint declares `mode='current'`, `input_dim=162`; the loaded class is the repository's native `FlashSACActor`, with 293,374 parameters and no RNN/LSTM/GRU modules. Although the shared harness maintains an eight-frame buffer, `predict_bc` selects only `obs_history[:, -1]` in this mode.

CPU audit with the real checkpoint: replacing all seven earlier frames with unrelated random values scaled by 1000 produced **exactly zero action difference**. Supplying a one-frame `[N,1,162]` tensor reproduces all saved rollout actions with a maximum absolute CPU/GPU numerical difference of 1.49e-5. Neither parameters nor BatchNorm buffers changed during inference; the simulator evaluation also verified its model digest unchanged.

This means **no learned recurrent policy memory or stacked past observations are required for this demonstrated fixed-task solution**. It does not mean the 162-vector has no historical information: it explicitly includes previous action targets, closest-distance trackers and progress. Nor does it prove that the original noisy 140-dimensional observation, real hardware, longer trajectories or diverse initial states can dispense with recurrent memory.

### What this establishes and what it does not

Established: the exact current-state Flash actor architecture can express and execute grasp/lift-to-goal behavior in this fixed STR task when taught suitable successful actions, using filter 0.1. An absolute claim that the architecture cannot solve the task without LSTM is contradicted by this controlled result. Real-history BC does not show a success-rate advantage here; all three reached the ceiling.

Not established: FlashSAC can discover this policy from scratch, which RL learning component previously failed, performance with filter 1.0, deployment readiness, random-reset or new-object generalization, long-horizon retention, or equivalence to full STR/SAPG pretraining. This is supervised BC, not a successful from-scratch SAC training run.
