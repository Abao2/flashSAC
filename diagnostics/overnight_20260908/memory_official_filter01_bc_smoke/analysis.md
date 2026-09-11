# Successful-trajectory BC plumbing check — action filter 0.1

Source: `../official_filter01_smoke/transitions.npz`: 944 transitions, 16 successful fixed-start episodes, both arm and hand moving-average coefficients 0.1. These are repeated fixed-task episodes, not independent random initial-state generalization tests.

All models directly use the repository's `FlashSACActor`, width 128, two residual blocks, native UnitBatchNorm and parameter projection. Only deterministic tanh(mean) is supervised. The std head is **not trained**. Raw normalized executed actions are targets. No extra input normalization is inserted into BC.

Seed 0, 1,500 Adam updates, batch 512. Whole-episode split: 12 train episodes / 708 rows; four validation episodes / 236 rows. Real history contains eight observations, oldest to newest, padded with the current episode's first observation at reset. Repeat-current has the same input width and parameter count as real-history but contains no past information.

| Input | Validation action MSE | Action RMSE | Arm RMSE | Hand RMSE |
| --- | ---: | ---: | ---: | ---: |
| Current state | 0.001091 | 0.03302 | 0.05069 | 0.02490 |
| Repeat current | 0.001804 | 0.04247 | 0.06844 | 0.02979 |
| Real history | 0.000861 | 0.02935 | 0.04234 | 0.02377 |

Constant train-mean-action validation MSE is 0.43505. The current-state native Flash actor fits the teacher's actions well on the sampled state manifold. History has somewhat lower offline action error, but this small, highly repetitive dataset does not establish memory necessity or a robust performance difference.

**No closed-loop result is contained in this test.** The environment must next evaluate the saved policies with both moving-average coefficients **0.1**, matching the demonstrations. Good offline imitation does not imply robust recovery from states outside the demonstrations; failures after closed-loop drift need not imply an observation/memory deficiency.

Saved checkpoints: `bc_current.pt`, `bc_repeat.pt`, `bc_history.pt`. Call `load_bc_checkpoint` and `predict_bc` from `scripts/diagnose_str_memory.py`; history is `[N,k,162]` with `k=payload['config']['window']`. Reset history separately for every environment after every reset. Do not sample the untrained std head.

## Subsequent current-state closed-loop check

The separate `../bc_current_smoke_eval/summary.json` now records 32 deterministic fixed-start episodes with filter 0.1. All 32 crossed the task's 10 cm lift threshold and the sustained-lift/near-palm proxy, but all 32 later fell below the reset height and none completed the goal. Mean maximum object lift was 0.11737 m; final position error was 0.16752 m. Each episode reached the 600-step timeout.

This shows the native current-state feedforward actor can produce repeatable lifting behavior on this fixed setup after supervised training. It does **not** demonstrate stable grasp retention, goal completion, generalization, or successful from-scratch SAC training. The geometric proxy is not a verified contact sensor. It does weaken the claim that LSTM is intrinsically necessary for any lift in this setting.
