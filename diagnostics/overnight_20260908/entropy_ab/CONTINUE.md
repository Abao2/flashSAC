# Initial alpha × critic-only warm-up: bounded retention diagnostic

Prepared on CPU only. Main must review and launch; this preparation did not start training or GPU jobs. Six configurations are queued **serially**, not concurrently. The queue stops at **2026-09-09 09:00 Asia/Shanghai (01:00 UTC)** and may stop before every job finishes.

## Question

The preceding selected BC policy succeeds when frozen, but the earlier alpha=0.01 retention test loses its skill after native SAC updates. At critic warm-up end, the preceding diagnostic measured strong lower-support saturation. This experiment asks whether reducing the **initial adaptive alpha** changes critic saturation and short-horizon skill retention, with and without critic-only warm-up.

This is **not a pure actor entropy-term intervention**. Alpha changes both the actor maximum-entropy objective and the critic soft TD target. Native temperature adaptation resumes with actor updates; alpha is not held constant. The result cannot alone distinguish every source of learning instability or establish long-budget training success.

## Six branches

| Initial alpha | Critic-only calls | Branch |
| --- | ---: | --- |
| 0.01 | 0 | `alpha_1e-2_gate0` |
| 0.01 | 2000 | `alpha_1e-2_gate2000` |
| 0.0001 | 0 | `alpha_1e-4_gate0` |
| 0.0001 | 2000 | `alpha_1e-4_gate2000` |
| 0.000001 | 0 | `alpha_1e-6_gate0` |
| 0.000001 | 2000 | `alpha_1e-6_gate2000` |

All use seed 0, the selected `bc_std005_eigen` actor, controls arm/hand=0.1/0.1, native Flash architecture/update rules, original STR task reward/reset/observations, 1024 environments, replay capacity 10M, replay minimum 100k, and fresh optimizer states (`agent.load_optimizer=false`). The previous alpha=0.01 experiment is **rerun** at this exact short budget; its longer run is not substituted as the endpoint control.

Each branch collects **2,000,896 transitions = 1954 vector steps**. Replay becomes eligible at vector step 98, giving an expected **3714 native network calls** at UTD=2. The nominal config counter without replay warm-up is 3908, not the number of actual updates. Expected actor/temperature opportunities are 1857 for gate0 and 857 for gate2000. Actual optimizer steps and skipped AMP updates must be read from the audits. All learning-rate decay horizons remain 19532 updates; the warm-up branch still has fewer actor/temperature scheduler steps at equal data budget.

## Exact checkpoint intervention

`scripts/clone_str_initial_alpha.py` clones the six native checkpoint files into `initial/alpha_1e-*`. It modifies only:

`temperature.pt → network_state_dict → _orig_mod.log_temp = log(initial_alpha)`

It does not rely solely on a config override: native loading would replace the config-initialized temperature with the saved value. The config value is also set to the matching alpha so the resolved config is truthful. Actor, critic, target critic, reward normalizer and agent state are byte-identical to the selected source. Other temperature payload fields are recursively equal. The alpha=0.01 clone is byte-identical across all six files. Each `alpha_initialization.json` records source/output hashes; the source is preserved.

The loaded actor and stochastic exploration standard deviations are identical initially across alphas. Frozen rollout sampling does not multiply actions by SAC alpha. Therefore existing selected-actor update0 frozen results are reused, while each branch still saves its full native update0 snapshot. No critic or normalizer from previous training is transferred.

## What is recorded

- Snapshots after native network calls: `0,10,100,1000,2000,2001,2002,2010,2100,3000`.
- Gate0 rollout checkpoints: update10, update100, update1000, final.
- Gate2000 rollout checkpoints: update2000, update2010, update2100, final.
- Each rollout checkpoint: deterministic 64 episodes / max1300 vector steps; native stochastic 128 episodes / max2500 steps, 32 environments, seed0 and filters0.1/0.1.
- CPU critic probes at update2000 and final for every branch, using the **same fixed expert dataset**. For gate0, update2000 is a matched reference, not a warm-up end. These are off-policy probes, not training replay or calibrated true returns.
- Final native checkpoint: `runs/<branch>/models/final/step1954/`.
- Warm-up audit: `runs/<branch>/warmup_audit.json`; this is incremental, not the queue completion marker.

Final success rates must use each summary's actual completed episode count. A missing/partial result is not zero success. Compare initial alpha, its learned trajectory, lower/upper support probability mass, return, physical lift proxies and final-goal success together. The startup random action and randomized initial horizon remain the native training behavior; separate frozen startup gates test that mismatch. Gym Box random-action RNG is not explicitly seeded by the native IsaacLab path, so a shared seed label is not a promise of identical first random actions.

## Files and launch boundary

`manifest.json` contains 66 jobs: 6 training, 48 rollout evaluations, 12 CPU critic probes. `configs/` holds six fully resolved native configs. `config_assertions.json` records the allowed config differences, verified initial hashes, budgets, paths, and source hashes. `tests/test_str_initial_alpha.py` passed on CPU, and clone/hash/config validations passed without CUDA initialization.

Launch only after reviewing the manifest. The existing startup queue status is:

`/home/abao/flashsac-robotics/diagnostics/overnight_20260908/warmstart/startup_baseline_status.json`

The existing runner accepts this as `--after-status` and waits for completion; a failed deadline/interrupted dependency stops the new queue. Use the standard hard deadline, diagnostic GPU free-memory guard 12GiB and training guard 36GiB. It never kills unrelated processes. Main owns the launch command and service creation.

If code changes before launch, compare `config_assertions.json`'s `source_sha256` against the current file hashes and review differences. Do not silently reuse the old validation claim. Existing nonempty training output directories are refused rather than overwritten; inspect a partial run before any manual recovery. This plan does not authorize a new full training campaign or real-robot deployment.
