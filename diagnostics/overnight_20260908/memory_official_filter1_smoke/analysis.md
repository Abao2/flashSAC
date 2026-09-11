# CPU memory prediction smoke — official policy, action filter 1.0

Dataset: `official_wrapper_smoke/transitions.npz`, 3,135 transitions / 16 fixed-start episodes. These episodes did **not** reach the target; this is a dynamics/reward prediction diagnostic, not successful behavior cloning.

Train: 12 whole episodes / 2,646 rows. Validation: episode IDs 2, 3, 10, 11 / 489 rows. All three conditions use the same split, seed, batches, target scaling and current action input. Real history uses 8 frames (at 60 Hz, seven previous intervals cover about 117 ms). Repeat-current and history have equal parameter count. Current-state model is smaller.

| Updates | Input | Train normalized MSE | Validation normalized MSE | Validation keypoint-delta RMSE |
| --- | --- | ---: | ---: | ---: |
| 500 | Current | 0.3156 | 1.2143 | 12.33 mm |
| 500 | Repeat current | 0.2052 | 1.2753 | 12.67 mm |
| 500 | Real history | 0.1557 | 1.2518 | 12.87 mm |
| 1000 | Current | 0.2093 | 1.2717 | 12.47 mm |
| 1000 | Repeat current | 0.1200 | 1.3885 | 12.77 mm |
| 1000 | Real history | 0.0602 | 1.3784 | 13.11 mm |

The train-mean baseline has validation normalized MSE 1.8783. All fitted models contain predictive information, but added history does **not** beat current state on these held-out episodes. Extra fitting lowers training loss and worsens validation loss, consistent with overfitting this small dataset. History improves reward RMSE at 500 updates (17.85 to 15.23), but this benefit does not persist at 1000 updates and does not improve keypoint-delta prediction.

Limits: one seed, four validation episodes, fixed starts, no confidence interval, no success trajectories, and no closed-loop control test. This neither proves the state is Markov nor rules out memory helping another setting. It provides no basis for claiming LSTM is required.

500-update details/checkpoints: this directory's `summary.json`. Independent 1000-update rerun with the same initialization and split: `../memory_official_filter1_smoke_1000/summary.json`.
