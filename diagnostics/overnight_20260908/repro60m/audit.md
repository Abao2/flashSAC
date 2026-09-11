# Continuous60M — final independent audit of all three training seeds

Snapshot: 2026-09-08T20:39:40.253970+00:00. **15/15 jobs complete: three fresh continuous training runs and all twelve frozen evaluations.** Only CPU reads were used for this audit; no simulator, GPU, optimizer or queue changes.

## Verified frozen-policy outcomes

|Training seed|Budget|Deterministic goal success|Native stochastic goal success|
|---|---:|---:|---:|
|0|50,001,920|0/64|0/128|
|0|60,002,304|62/64 (96.88%)|127/128 (99.22%)|
|1|50,001,920|1/64 (1.56%)|40/128 (31.25%)|
|1|60,002,304|1/64 (1.56%)|88/128 (68.75%)|
|2|50,001,920|64/64 (100%)|125/128 (97.66%)|
|2|60,002,304|64/64 (100%)|128/128 (100%)|

All twelve evaluations used the specified checkpoint, common rollout seed0, full episode budget and unchanged model. Seed0 success at60M and seed2 success at50M/60M occurred in **uninterrupted training processes without loading or resetting replay**. Replay reset is therefore **not necessary in these trials**; this does not show that reset can never help. Native feed-forward FlashSAC can learn this fixed task without BC or LSTM, but these three seeds do not establish seed-robust deterministic success, paper-wide replication, or real-robot readiness.

![Independent frozen evaluations, not online training-window curves](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/reproducibility.png)

The figure shows raw evaluation counts, separated by sampling mode and training seed. No three-seed IID confidence interval is inferred.

Seed1 must not be described by a single mixed success rate: its stochastic policy improves substantially, while deterministic tanh-mean control remains1/64. These are different policies, not conflicting measurements. No extra policy/action probe was launched.

## Fresh native training provenance and continuity

For all three completed runs, independently rechecked in the final audit:

- Queue actual command matches the prepared manifest; one attempt, one PID/run directory, exit code0.
- Recomposition from actual launch arguments exactly matches saved resolved config.
- `agent_load_path=null`, `buffer_load_path=null`, `actor_bc_alpha=0`: fresh random native feed-forward SAC, no BC, no LSTM or checkpoint warmstart.
- Same fixed eraser/start/goal, state162 inputs, DR off, arm filter1.0/hand0.1,1,024 environments, replay10M/min100k, fixed LR decay19,532.
- All17 recorded source/config/launcher/core hashes still match preparation.
- Logs show intermediate saves at vector steps9,766/19,532/29,298/39,064/48,830 with counters19,338/38,870/58,402/77,934/97,466, and no checkpoint-load message.
- Final save print is absent from captured stdout, but all six final files independently load, are finite, and contain the correct final native counter and normalizer count. File evidence, not a missing stdout line, establishes final saving.

This configuration proof uses actual logged argv plus unchanged source; the trainer did not separately dump a live runtime config.

## Counters, normalization and wallclock

|Training seed|Whole native60M process wallclock|Final Critic calls / applied steps|Final Actor calls / applied steps|Temperature applied steps|
|---|---:|---:|---:|---:|
|0|1,718.66s (28m39s)|116,998 /116,945|58,499 /58,488|58,499|
|1|1,622.55s (27m03s)|116,998 /116,936|58,499 /58,493|58,499|
|2|1,699.64s (28m20s)|116,998 /116,931|58,499 /58,494|58,499|

The native call count excludes the first97 replay-fill vector steps: (58,596−97)×2=116,998. AMP skipped some applied steps. The global native counter—not the Network wrapper's unused stored counter—is used here.

All three50M checkpoints have normalizer count50,001,920 and native calls97,466. All three60M checkpoints have count60,002,304 and calls116,998; all budget checks pass. All36 component files were reread and verified finite, including optimizer/scaler state, with hashes recorded.

|Training seed / budget|Alpha|Reward divisor|Actor applied|Critic applied|
|---|---:|---:|---:|---:|
|0 /50M|1.35507e-5|251.954|48,727|97,420|
|0 /60M|6.66895e-5|252.881|58,488|116,945|
|1 /50M|8.30222e-5|249.070|48,730|97,413|
|1 /60M|2.66398e-4|250.893|58,493|116,936|
|2 /50M|1.67106e-4|252.504|48,728|97,414|
|2 /60M|1.31381e-4|265.750|58,494|116,931|

Wallclock comes from queue process start/end, including initialization and final close; it excludes subsequent frozen evaluation jobs.

## Online goal-window timing—not a convergence guarantee

|Training seed|First positive window|First window≥50%|First five consecutive windows≥50% begin|Final window / last10 window mean|
|---|---:|---:|---:|---:|
|0|45.056M at22m03s|45.8752M at22m26s|52.6848M at25m30s|82.31% /83.97%|
|1|33.3312M at15m29s|44.288M at20m16s|47.0528M at21m24s|44.88% /58.98%|
|2|31.488M at14m41s|33.2288M at15m31s|33.2288M at15m31s|99.23% /99.41%|

These are logging-window means, not episode-weighted rates. Early high online windows can coexist with a failing frozen50M checkpoint; they cannot replace saved-policy evaluation. Seed1's final online drop also demonstrates non-monotonic learning.

## Original50M seed0 versus this seed0 prefix

The source50M Actor and new continuous60M seed0 Actor at50M have different SHA256 hashes (recorded in `audit.json`): they are **not bitwise-replayed prefixes**. Equal configured seed0 and algorithm recipe do not promise that. Native `train.py` seeds Python/NumPy/Torch, but the wrapper creates Gymnasium Box spaces without explicitly seeding the action space, then native random replay-fill actions use `action_space.sample()`. GPU simulation can also be nondeterministic. This audit changed neither behavior.

Do not attribute prefix differences solely to the60M budget or use them as a deterministic counterfactual. The supported conclusion is narrower: fresh continuous native SAC runs can reach successful frozen policies without a replay reset; seed dependence and deterministic/stochastic differences remain.

Full source hashes, checkpoint hashes, applied counters, goal-window event timestamps and evaluation counts are in [audit.json](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/audit.json). Final all-seed results are in [summary.json](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/summary.json) and [analysis.md](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/analysis.md). The figure is reproducible with [plot_reproducibility.py](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/repro60m/plot_reproducibility.py).
