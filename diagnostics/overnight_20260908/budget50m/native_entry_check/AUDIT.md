# Native training-entry check — CPU preparation only

Two branches load the same full `step48830` checkpoint: native normal updates versus Critic updates with Actor/temperature frozen throughout. Each collects 5,000,192 new transitions (4,883 vector steps × 1,024 environments). No GPU was launched by this preparation.

The isolated wrapper now treats gate/snapshot counts as **new calls relative to the loaded counter**. It never resets the native counter. Source counter is 97,466; expected additional Critic calls are 9,572, final native counter 107,038. Normal Actor/temperature have 4,786 opportunities; frozen branch has zero. AMP can skip attempted optimizer steps, so read actual audit/optimizer counters as well.

Replay is fresh: capacity10M, minimum100k, first update at vector step98 after100,352 transitions. Snapshot `update0` is immediately before that first update, not a byte-identical copy of the original six files: reward normalization has already seen new data. This is **cold-replay resume, not uninterrupted continuation**. Saved per-environment running-return histories are also restored, while the physical simulator is fresh; both branches inherit this equally.

CPU validation:

- 11 tests pass: fresh BC behavior retained; nonzero and odd native counters; relative snapshots; unfinished long gate verifies Actor/temperature remain frozen at final close; reset-reward wrapper regressions.
- Actual native full checkpoint load preserves all four model state dictionaries, three optimizer/scheduler dictionaries, native counter, gradient scaler, and every reward-normalizer field. CPU-only load used tiny replay and eager compiled-name wrappers; this does not test CUDA kernels.
- Source LR is already1.5e-4; Actor/temperature scheduler epoch48,733, Critic97,466. Loaded scheduler and original19,532-step schedule are preserved, not restarted.
- Both resolved agent configs match the source50M recipe exactly. Task/environment fields match except the explicitly shortened environment budget.
- Previous wrapper preserved in `source_backup/train_str_warmstart_before_relative.py`; old/new hashes and all six source checkpoint hashes are in `validation.json`. No core/environment/harness edits.

Main launch requires the existing queue runner with this `manifest.json`, explicit `--status` pointing to this directory's `queue_status.json`, and `--after-status` pointing to `budget50m/reset_reward_probe/queue_status.json`. The hard deadline remains09:00 local. Manifest dependency metadata alone does not enforce waiting.

Primary question: does the frozen Actor recover the late-online goal-success behavior when used through the native training entry? Compare native goal/lift logs to matched frozen evaluations at relative100/1000/final. If it does not, cold replay/new simulator history still prevent identifying which element of uninterrupted training mattered. Do not call a zero-success evaluation a loader/algorithm failure without that distinction.
