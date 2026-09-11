# Native-entry check — completed

All14 main queue jobs and3 additional rollout-seed confirmation jobs completed. Initial final frozen-policy evaluation: **normal continuation deterministic64/64 goals (100%); native stochastic110/128 (85.94%)**. All192 requested episodes completed, model unchanged during rollout.

Additional confirmation uses this **same trained checkpoint**, not newly trained policies:

|Sampling / rollout seed|Goal success|Completed episodes|
|---|---:|---:|
|Native stochastic /0 (initial)|110/128 (85.94%)|128/128|
|Native stochastic /1|218/256 (85.16%)|256/256|
|Native stochastic /2|229/256 (89.45%)|256/256|
|Deterministic /1|64/64 (100%)|64/64|

Combined native stochastic descriptive count: **557/640 =87.03%**. These are rollout/noise seeds for one training seed; vector episodes are correlated, so this is not three independent training replications and no independent-IID confidence interval is claimed. Every confirmation rollout kept the model unchanged and completed its full budget. Stochastic seed2 has231 terminations but only229 goal successes: not every termination is a successful task.

## Provenance: native SAC, not BC

`budget50m/configs/arm1_hand01_seed0.json` confirms `agent_load_path=null`, `buffer_load_path=null`, `actor_bc_alpha=0`. This source run started with random networks and used the original feed-forward FlashSAC architecture—not a BC checkpoint or LSTM.

Chain: **random initialization → continuous50,001,920 transitions → full six-file checkpoint `budget50m/models/arm1_hand01/seed0/step48830` → fresh simulator/replay and5,000,192 native SAC transitions → successful final saved checkpoint**. Total collected55,002,112 transitions. Source and resume paths/hashes are recorded in [summary.json](summary.json) and [validation.json](validation.json).

Controls stayed arm filter1.0/hand0.1, fixed eraser/start/goal, DR off, state162 observations, native FlashSAC reward/termination/updates. This is a privileged simulator-state single-task diagnostic, not a deployable policy or paper-level result.

## Online windows versus saved-policy evaluation

| Checkpoint / branch | Nearest logged goal windows | Frozen deterministic goals | Frozen native-stochastic goals |
|---|---:|---:|---:|
| Original50M checkpoint |final80.21%; last10 mean52.66%|0/64|0/128|
| Frozen Actor, relative100 calls |0% before/after|0/64|0/128|
| Frozen Actor, relative1,000 calls |0% before/after|0/64|0/128|
| Frozen Actor, final5M extra |0%; all97 windows zero|0/64|0/128|
| Normal updates, relative100 calls |0% before/after|0/64|0/128|
| Normal updates, relative1,000 calls |0% before/after|0/64|0/128|
| Normal updates, final5M extra |final98.04%; last10 mean96.29%|64/64|110/128|

Relative100/1,000 snapshots are at150,528/611,328 new transitions; the table brackets them with available TB log windows, not an invented exact-time evaluation. Window averages are unweighted, not episode-weighted overall rates.

## Final budget and optimizer accounting

Both branches collected5,000,192 new transitions (4,883 vector steps ×1,024 environments), with fresh replay10M/min100k. Source native counter97,466 was preserved; both finished107,038.

| Quantity | Frozen Actor/temp | Normal updates |
|---|---:|---:|
| New Critic attempted / applied |9,572 /9,570|9,572 /9,567|
| New Actor attempted / applied |0 /0|4,786 /4,786|
| New temperature applied |0|4,786|
| Final alpha |2.06535e-5|5.61762e-5|
| Final online return window |318.33|1,350.73|
| Final frozen deterministic mean return |333.13|1,356.01|
| Final frozen stochastic mean return |317.21|1,311.33|

AMP skips explain attempted-versus-applied differences. Both full final checkpoints are finite; both normalizer counts are55,002,112. Frozen Actor parameters, BN, optimizer/scheduler and temperature stayed invariant.

Normal final deterministic episodes:64 successes,64 terminations,0 timeouts,0 drop-below-reset-after-lift proxies; mean32.11 steps. Stochastic:110 successes/terminations,18 timeouts,11 drop proxies; mean154.05 steps. All episodes triggered formal task-lift, but lift/drop proxies alone are not stable-grasp proof. The separate harness audit checks the task's actual cumulative-goal/reward/termination semantics.

## Interpretation and limits

In this matched test, **continued Actor/temperature updates were necessary for improvement**: frozen control remained zero, normal updates produced a saved policy that succeeds under an independent frozen rollout. Thus success is no longer only a live-training metric.

This does **not** establish why the improvement appeared: another5M transitions, cold replay, full reload, and different simulator/history are not separated. A fresh continuous60M multi-seed replication will test whether reload is unnecessary and whether the result repeats.

The saved reward normalizer includes per-environment discounted-return histories from the old simulator; those histories do not match new physical initial states. Both branches inherit them equally. Even the frozen branch continues Critic/target/replay/normalizer/scaler updates.

Compiled-path questions remain separate: `agent.py` explicitly compiles `actor.network.get_mean_and_std`; seeing a bound method does not rule compilation out. No CUDA parity claim is made here.

This is one training seed and one fixed task, with three stochastic rollout/noise seeds—not independent training replication,24-task paper replication, robustness across physical resets, or real-robot validation.
