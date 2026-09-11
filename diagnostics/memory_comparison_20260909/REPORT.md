# STR memory comparison — training windows

Updated: 2026-09-09T09:14:41.360112+00:00

Queue: stopped_by_user_after_seed0

## Interpretation

- CPU-only TensorBoard training snapshot; no evaluation or deployment evidence.
- Each statistic is an unweighted mean of logged windows in the preceding 10M transitions, not an episode-weighted overall rate.
- Lift fraction = lift bonus / 300: an episode crossed the task height threshold at least once; not stable grasping success.
- Goals/episode counts reached goals, not the fraction completing the 50-goal chain.
- Missing data are PENDING/null, not zero. Seeds and distinct run directories remain separate.
- A reuses existing full STR obs140 feed-forward runs. B uses state162 feed-forward actor.
- C uses a finite 32-frame LSTM actor with obs140 and history-Q (current state162 + 32 * 141 = 4674 inputs before actions).
- C changes actor/critic capacity and critic information; it is not an actor-only causal ablation or official full-episode STR LSTM.
- Matched references use identical window endpoints no later than either run's last observation; do not compare baseline100M with candidate10M as a win/loss.

## Latest preceding 10M windows (different progress; not a ranking)

| Group / seed / run | Window end (M) | Return | Lift event fraction | Goals/episode | Keypoint reward | Tolerance | Alpha | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A / 0 / 1 | 100.003840 | 82.0127 | 0.112109 | 0.000746486 | 1.14841 | 0.075 | 1.30736e-06 | -13.8724 |
| A / 1 / 1 | 100.003840 | 80.5696 | 0.107766 | 0.000847849 | 1.06351 | 0.075 | 9.05519e-07 | -13.5914 |
| A / 2 / 1 | 100.003840 | 79.8834 | 0.107427 | 0.000785836 | 1.09583 | 0.075 | 8.42204e-07 | -13.8669 |
| B / 0 / 1 | 100.003840 | 82.2042 | 0.10811 | 0.000492765 | 1.02272 | 0.075 | 1.43546e-06 | -13.9577 |
| B / 1 / 1 | 30.003200 | 22.7614 | 0.0991238 | 0.000436304 | 0.943976 | 0.075 | 0.000164908 | 19.6669 |
| B / 2 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| C / 0 / 1 | 100.003840 | 73.522 | 0.106637 | 0.000275043 | 1.07147 | 0.075 | 2.75554e-06 | -13.7993 |
| C / 1 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| C / 2 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Same-progress baseline references

| Group / seed / run | Window end (M) | Return | Lift event fraction | Goals/episode | Keypoint reward | Tolerance | Alpha | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A / 0 / match1 | 100.003840 | 82.0127 | 0.112109 | 0.000746486 | 1.14841 | 0.075 | 1.30736e-06 | -13.8724 |
| B / 0 / match1 | 100.003840 | 82.2042 | 0.10811 | 0.000492765 | 1.02272 | 0.075 | 1.43546e-06 | -13.9577 |
| A / 0 / match1 | 100.003840 | 82.0127 | 0.112109 | 0.000746486 | 1.14841 | 0.075 | 1.30736e-06 | -13.8724 |
| C / 0 / match1 | 100.003840 | 73.522 | 0.106637 | 0.000275043 | 1.07147 | 0.075 | 2.75554e-06 | -13.7993 |
| A / 1 / match1 | 30.003200 | 33.2532 | 0.09789 | 0.000441585 | 0.982 | 0.075 | 0.000170044 | 19.3858 |
| B / 1 / match1 | 30.003200 | 22.7614 | 0.0991238 | 0.000436304 | 0.943976 | 0.075 | 0.000164908 | 19.6669 |
| A vs C / 1 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| A vs B / 2 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| A vs C / 2 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Fixed milestones (preceding 10M windows)

| Group / seed / run | Window end (M) | Return | Lift event fraction | Goals/episode | Keypoint reward | Tolerance | Alpha | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A / 0 / 1 | 20 | 22.4812 | 0.0993094 | 0.000360009 | 0.98143 | 0.075 | 0.000624676 | 19.5019 |
| A / 0 / 1 | 40 | 49.41 | 0.100282 | 0.000491904 | 0.946954 | 0.075 | 4.7214e-05 | 17.3692 |
| A / 0 / 1 | 60 | 72.4157 | 0.102714 | 0.000698851 | 1.03525 | 0.075 | 3.9393e-06 | 7.73143 |
| A / 0 / 1 | 80 | 81.2493 | 0.110285 | 0.000525114 | 1.07107 | 0.075 | 9.20079e-07 | -13.9716 |
| A / 0 / 1 | 100 | 81.9356 | 0.111915 | 0.000754103 | 1.14948 | 0.075 | 1.30737e-06 | -13.8751 |
| A / 1 / 1 | 20 | 19.875 | 0.100777 | 0.000383571 | 0.984794 | 0.075 | 0.000644509 | 19.5425 |
| A / 1 / 1 | 40 | 53.8255 | 0.103735 | 0.000524371 | 1.00186 | 0.075 | 4.88514e-05 | 17.4419 |
| A / 1 / 1 | 60 | 70.2784 | 0.104238 | 0.00101059 | 0.983149 | 0.075 | 4.02722e-06 | 9.81515 |
| A / 1 / 1 | 80 | 77.5038 | 0.105549 | 0.000945221 | 1.04987 | 0.075 | 8.33236e-07 | -13.6912 |
| A / 1 / 1 | 100 | 80.7806 | 0.108405 | 0.000856501 | 1.0706 | 0.075 | 9.06452e-07 | -13.5795 |
| A / 2 / 1 | 20 | 27.9115 | 0.102368 | 0.000389435 | 1.06049 | 0.075 | 0.00063119 | 19.4939 |
| A / 2 / 1 | 40 | 50.6627 | 0.0993651 | 0.00049893 | 0.986701 | 0.075 | 4.7669e-05 | 18.0035 |
| A / 2 / 1 | 60 | 69.0319 | 0.103407 | 0.000545919 | 1.06763 | 0.075 | 3.91369e-06 | 11.2927 |
| A / 2 / 1 | 80 | 77.3205 | 0.106856 | 0.000841546 | 1.0818 | 0.075 | 6.4319e-07 | -13.6451 |
| A / 2 / 1 | 100 | 79.6119 | 0.107033 | 0.00064489 | 1.08712 | 0.075 | 8.42904e-07 | -13.8679 |
| B / 0 / 1 | 20 | 23.9775 | 0.101832 | 0.000614463 | 1.04681 | 0.075 | 0.000643546 | 19.4771 |
| B / 0 / 1 | 40 | 50.6209 | 0.10084 | 8.45149e-05 | 0.979412 | 0.075 | 4.87794e-05 | 17.2189 |
| B / 0 / 1 | 60 | 68.9085 | 0.103845 | 0.000608618 | 0.998591 | 0.075 | 4.03165e-06 | 9.58806 |
| B / 0 / 1 | 80 | 79.8911 | 0.105837 | 0.000540283 | 1.00037 | 0.075 | 9.54134e-07 | -14.1846 |
| B / 0 / 1 | 100 | 82.0459 | 0.10779 | 0.000497793 | 1.02353 | 0.075 | 1.43381e-06 | -13.9556 |
| B / 1 / 1 | 20 | 24.8606 | 0.101838 | 0.000714779 | 1.00423 | 0.075 | 0.00062648 | 19.5112 |
| B / 1 / 1 | 40 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B / 1 / 1 | 60 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B / 1 / 1 | 80 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| B / 1 / 1 | 100 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| C / 0 / 1 | 20 | 19.6701 | 0.0995627 | 0.000264794 | 1.01089 | 0.075 | 0.000619651 | 19.3857 |
| C / 0 / 1 | 40 | 50.6593 | 0.0982294 | 0.000320879 | 0.920071 | 0.075 | 4.73678e-05 | 15.1759 |
| C / 0 / 1 | 60 | 65.9267 | 0.105181 | 0.000134292 | 1.04058 | 0.075 | 4.56123e-06 | -6.68772 |
| C / 0 / 1 | 80 | 68.7486 | 0.102917 | 0.000417148 | 1.06289 | 0.075 | 3.7212e-06 | -13.7058 |
| C / 0 / 1 | 100 | 73.8048 | 0.107311 | 0.00027785 | 1.07776 | 0.075 | 2.75633e-06 | -13.7976 |

Missing groups have no milestones yet. Exact run paths, sample counts, actual last logged steps, queue job states, and pending values are in `summary.json`.
