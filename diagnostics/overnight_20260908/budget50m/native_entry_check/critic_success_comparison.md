# Successful native SAC checkpoint: paired CPU critic probes

## Conclusion

On the **same recorded states and shared recorded/uniform candidate actions**, the successful checkpoint has greater learned Q/action sensitivity. This accompanies demonstrated goal success, but **does not establish better Q accuracy or causality**. In particular, this is not a simple change from entropy-dominated to Q-dominated gradients: the old model already had Q/entropy raw parameter-gradient ratios17.79 and89.91; the new ratios are18.31 and81.69.

Alpha **increased** from2.06535e-5 to5.61762e-5 (2.72×). Reward-normalization divisor changed only251.597→253.166 (+0.62%). Therefore lower alpha or a large normalization-unit change cannot explain these paired measurements. The earlier BC-warmstart negative-support-floor problem is not a universal explanation for native SAC.

## Exact comparison

- Old: `budget50m/models/arm1_hand01/seed0/step48830`, frozen0/64 deterministic and0/128 native stochastic goals.
- New: `native_entry_check/training/normal/models/final/step4883`, from the same random-init native50M run plus5M cold-replay native updates; no BC. Final deterministic64/64; stochastic confirmation557/640 across three rollout/noise seeds for this same trained model.
- Failure batch: uniform512 rows from old native-stochastic rollout76,800 rows;0 terminations,1 timeout,2 raw-reward>200 events.
- Successful-support batch: uniform512 rows among8,919 rows belonging to the new rollout's110 successful episodes;2 terminations,0 timeouts,6 reward>200 events. Its18 unsuccessful episodes are deliberately excluded. This is success-conditioned support, not the entire new rollout or replay distribution.

The two models use exactly the same512 indices and order **within each batch**; the two batches remain separate. Selection seed0,32 uniform actions/state, target batches128, four raw Actor-gradient repeats on all512 current/next pairs, eight inference entropy samples/state. Indices, dataset/checkpoint hashes and per-phase results are in [critic_success_comparison.json](critic_success_comparison.json). These are new50M-failure samples, not the older10M-failure512 samples used in `critic_followup.md`.

## Fixed-state critic signals

Q values are normalized soft values. `Q(recorded)` and uniform-action spread hold queried actions fixed across models. **dQ/da is evaluated at each model's own deterministic action**, not at a common action.

|Fixed512 batch / model|Q(recorded) mean|32 uniform-action Q std|Own-mean-action dQ/da L2|Top5 atom mass|
|---|---:|---:|---:|---:|
|Failure /old|0.2892|0.005596|0.01004|0.192%|
|Failure /new|1.4620|0.029001|0.04269|4.700%|
|Successful support /old|0.9213|0.019357|0.03054|1.334%|
|Successful support /new|3.4545|0.061151|0.09152|32.194%|

Uniform-action sensitivity rises5.18× on failure support and3.16× on successful support. This is a measurable change in the learned function, not proof that its rankings match real counterfactual returns. Higher Q or more atom mass near+5 is not automatically more accurate.

## Actor gradients and entropy

These are **raw gradients at each Actor's own sampled actions**, with common random draws but different action outputs. Actor uses training cross-BN on current+next rows; Critic uses inference BN. No Adam step, AMP, unit-norm projection or live model mutation occurs.

|Fixed512 batch / model|Q-loss gradient L2|Alpha-weighted entropy gradient L2|Q/entropy norm ratio|Gradient cosine|
|---|---:|---:|---:|---:|
|Failure /old|0.04748|0.002670|17.79|-0.210|
|Failure /new|0.23554|0.012864|18.31|-0.039|
|Successful support /old|0.21254|0.002364|89.91|-0.144|
|Successful support /new|0.54924|0.006724|81.69|-0.103|

Q-gradient magnitude increases, but entropy-gradient magnitude also increases. The ratio is nearly unchanged on failure support and decreases on successful support; it cannot be used as a scalar “learning works” threshold.

True tanh entropy (nats, not pretanh Gaussian entropy) also depends strongly on state/BN mode:

|Batch / model|Inference H,8MC|Training-cross-BN H,4 gradient draws|Mean pretanh sigma|
|---|---:|---:|---:|
|Failure /old|-33.22|-13.59|0.5415|
|Failure /new|-51.24|-50.85|0.4561|
|Successful support /old|-53.01|-13.94|0.4490|
|Successful support /new|-42.82|-26.69|0.4328|

On successful support, inference entropy actually rises (becomes less negative). These mode-specific values cannot be interchanged with historical TB/replay entropy. New-versus-old deterministic normalized-action RMSE is0.7830 on failure support (arm0.8489/hand0.7608), and0.8383 on successful support (arm0.9001/hand0.8177): behavior changed substantially, not just sigma. These are pre-controller normalized actions, not physical joint-motion distances.

## TD support clipping: lower floor absent, upper pressure increased

Recomputed categorical targets use the **same recorded transitions**, each checkpoint's current next-policy/alpha/frozen normalizer and cloned target training cross-BN. Terminated transitions do not bootstrap; the single timeout uses recorded pre-reset final_obs. These are not historical replay targets or clipping frequencies.

|Batch / model|Below-−5 probability mass|Above-+5 probability mass|Mean unprojected target|Mean clipping shift|
|---|---:|---:|---:|---:|
|Failure /old|0%|0.001626%|0.2231|-0.0000102|
|Failure /new|0%|0.366553%|1.7078|-0.0031640|
|Successful support /old|0%|0.000736%|0.2829|-0.0000017|
|Successful support /new|0%|2.172653%|1.1109|-0.0081877|

Percentages are mean **categorical probability mass**, not percentages of failed episodes. New successful-support near-goal-center proxy rows(n100) have7.08% upper clipping mass and60.00% top5 atom mass; lifted-proxy rows(n194) have2.09% upper clipping. So “new policy succeeds” does not imply clipping disappeared. It did not. Lower clipping is zero in all four sampled comparisons.

Final limits: rare events were not exhaustively sampled; reset coverage is only1/3 rows and cannot support broad reset claims. Cross-support inference is partly out of distribution, and cross-BN results depend on the artificial batch. No true-return ranking/MC calibration, GPU kernels, training updates or causal intervention was performed. All six checkpoint file hashes and Actor/Critic/target tensors remained unchanged. Continued updates, changed policy/state visitation and cold replay are not disentangled by this analysis.
