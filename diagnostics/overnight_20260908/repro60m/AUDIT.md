# Fresh continuous60M reproducibility — prepared, not launched

Three independent native feed-forward FlashSAC trainings (seeds0/1/2), each from random initialization with **no checkpoint, BC, replay or warmstart loading**. Fixed eraser/start/goal, simulator-state162 inputs, DR off, arm filter1.0/hand0.1.

Each native `train.py` process runs continuously for60,002,304 transitions =58,596 vector steps ×1,024 environments. Replay10M/min100k remains live throughout; snapshots every9,766 steps (approximately10M transitions) do not restart the simulator. Evaluate the50M (`step48830`) and60M (`step58596`) snapshots only after training finishes.

The queue has15 serial jobs:3 trainings +12 frozen evaluations (each deterministic64 episodes / native stochastic128 episodes). All evaluations use common seed0,32 environments, max2,500 vector steps; training seeds remain independent. Partial/missing results are not zero success.

CPU validation resolved all three Hydra configurations and compared every field against `budget50m/configs/arm1_hand01_seed0.json`. Differences are restricted to training budget and derived counters, seed, and output/run names. **All agent recipe fields besides seed remain identical**, especially fixed LR decay19,532 updates (not recalculated from60M), initial/peak3e-4, final1.5e-4. All task/controller fields match. Source/config/launcher/core hashes are in `validation.json`.

Expected per run: nominal budget117,192 update slots, actual116,998 Critic calls after initial replay fill,58,499 Actor/temperature opportunities. AMP may skip some applied optimizer steps; read saved counters instead of assuming every call updated parameters. First replay-ready vector step is98.

This tests whether the single-task SAC success can repeat with continuous budget and no reload. It does not alone isolate extra budget from the earlier cold-replay/full-reload effect. Three seeds are bounded reproducibility evidence, not a population estimate or paper24-task/real-robot result.

Parent-only launch: existing `run_str_diagnostic_queue.py` with this `manifest.json`, explicit `--status repro60m/queue_status.json`, and `--after-status budget50m/compiled_forward_probe/queue_status.json`. That prior status did not yet exist during preparation; do not fabricate it. Parent verifies its prerequisite/confirmation ordering. The runner enforces09:00local hard stop; each training additionally has5,400-second timeout. `depends_on` metadata is descriptive, not an execution gate.

No source files were changed for this preparation; existing native launchers and rollout harness are reused. Shell syntax, all training/evaluation argument layouts, expected checkpoint multiples, EULA/Python paths and resolved configs passed CPU checks. No GPU or queue was launched.
