# Research handoff — FlashSAC + STR + Dex4D

## Purpose and ownership

Preserve the local research process, including failed attempts, not just a polished
successful configuration. Continue diagnosis in the existing conversation if available;
this file makes the checkout understandable without relying on conversation memory.

This fork starts from official FlashSAC. Our additions include the external Isaac Lab
adapter, the Dex4D teacher-task port, STR integration, replay/transition fixes, history
experiments, diagnostic tooling and play commands. The upstream authors did not supply
this complete STR/Dex4D integration.

## Workspace and dependencies

The user requested ONE GitHub repository, Abao2/flashSAC. STR is included by Git subtree:

```text
workspace/
  flashsac-robotics/           this repository (GitHub name may be flashSAC)
    third_party/simtoolreal/  included modified STR source, not stock upstream
  Dex4D/Dex4D-Simulation/     external Dex4D sources/assets
```

If cloning the GitHub repository named flashSAC, it is useful to keep the local directory
name flashsac-robotics for existing scripts. Set SIMTOOLREAL_ROOT, DEX4D_ROOT,
PYTHONPATH and PYTHON_BIN as required by the selected script.
The primary STR training/play launchers default to third_party/simtoolreal. No separate STR clone is required.
Some historical diagnostic scripts still hard-code /home/abao, a sibling checkout, or server-specific locations.
For these scripts explicitly export SIMTOOLREAL_ROOT to the included directory.
Do not assume that a fresh clone is immediately portable without inspecting the entry point.

Local working Python: /home/abao/play2perfect/.venv_isaacsim/bin/python.
Original source checkout: /home/abao/simtoolreal.
The monorepo copy is third_party/simtoolreal; prefer editing that copy after this snapshot.
Changes made later only in the old external checkout will NOT automatically enter this repository.
The tested simulator is Isaac Sim 5.1 / Isaac Lab on Python 3.11; GUI workflows used
NumPy 1.26. The generic upstream installation instructions are not a complete lockfile
for these external robotics dependencies. See docs/RESEARCH_DEPENDENCIES.json for revisions.

## Main code areas

- flash_rl/agents/flashSAC: policy/value networks and optimization, including experimental history variants.
- flash_rl/envs/isaaclab.py: observations, actions, task registration, metrics and final_obs contract.
- flash_rl/buffers: n-step replay, per-row discounts and optional history handling.
- robotics_tasks/dex4d: our Isaac Lab port of Dex4D teacher stages 1/2 and 3.
- configs: preserve experimental configurations, not just the newest one.
- scripts: launchers, diagnostic checks, paired rollouts, evaluation and report generation.
- tests: regression tests; CPU tests are not a substitute for simulator rollout validation.

## Experiments to understand before making changes

Read docs/EXPERIMENT_INDEX.md, then the dated README/results in the relevant directories.

Important STR experiment families include fixed/simplified state-teacher tasks,
full-task no-DR baselines, arm/hand smoothing ablations, feed-forward versus history
experiments, entropy/initial-alpha and warm-start retention probes, n=3 credit assignment,
the 100M-to-500M continuation, upstream learner update parity, native/wrapper interface
parity, replay coverage and full-state Q action interventions.

A successful simplified task does not establish success on full STR. A completed
optimizer run does not establish convergence. The n=3 100M and 500M runs refer to
KUKA + Sharpa, not Wuji. Different task/controller/tolerance settings are not matched
algorithm comparisons.

The most recent relevant dated notes at snapshot time are:
- diagnostics/credit_nstep3_20260910/README.md
- diagnostics/credit_nstep3_continue_20260910/README.md
- diagnostics/update_parity_20260910/README.md
- diagnostics/str_fourway_audit_20260911/interface/README.md
- diagnostics/str_fourway_audit_20260911/README.md
- diagnostics/str_q_followup_20260911/README.md
- diagnostics/str_q_followup_20260911/actor_gradient/README.md

Some older status files describe earlier blockers. Read report timestamps and actual
completion artifacts; do not treat an older CPU-only status as the current GPU result.
The September 11 driver mismatch was subsequently resolved enough to run CUDA/play.
Checkpoint compiled/eager key compatibility was fixed in the shared Network.load path
and tested against the actual 100M actor/critic/target/temperature weights.

Diagnosis is not causally complete. Existing Q probes do not justify claiming Q is
definitely wrong, entropy necessarily dominates learning, or LSTM is the missing solution.
A measured training-versus-inference normalization gap is a hypothesis requiring a
controlled behavioral intervention, not permission to remove BN or rewrite the algorithm.

## Behavior and training entry points

- scripts/play_str_compare.sh: one GUI, official and Flash 100M, same hammer task.
- scripts/play_str_behavior.sh best|official: separate historical playback protocols.
- scripts/run_simtoolreal.sh: STR training launcher; inspect the selected configuration.
- scripts/run_dex4d_stage12.sh: Dex4D teacher stage 1/2.
- scripts/run_dex4d.sh: Dex4D teacher stage 3.

The side-by-side player uses shared no-DR settings, arm EMA=1, hand EMA=.1 and
tolerance=.01. Its goal sequence is shared but each environment advances according to
its own progress and resets independently. Both policies act deterministically.
The 100M “best” label is provisional, not a full checkpoint tournament result.

Do not claim that the complete Dex4D three-stage-plus-student pipeline is validated
merely because teacher-stage ports and smoke gates exist. Check the actual reports.

## Rules for continuing safely

1. Inspect git status and active processes; preserve concurrent edits and user-owned runs.
2. Keep old configurations and failed results. Add new dated experiments rather than overwriting evidence.
3. Record checkpoint, effective config, seed, task assets, tolerance, controller, DR, transitions,
   optimizer updates and wall-clock time. Separate measured results from hypotheses.
4. For off-policy boundaries, real termination masks bootstrap; time limits require the
   pre-reset final observation and the correct effective n-step discount.
5. For action/Q comparisons, match complete state and controller/goal trackers; check repeated
   baselines before interpreting small differences. Do not use Q to validate its own unknown tail.
6. Do not silently alter reward/curriculum/DR/control to make a run look successful.
7. Do not start training or deploy to hardware based only on historical instructions in these files.

## Git versus artifact backup

This commit preserves source, configs, small figures and text/JSON results.
docs/LOCAL_ARTIFACTS.json lists excluded binary inventory; it is NOT an uploaded backup.
Checkpoint directories need actor, critic, target critic, temperature, optimizer/scheduler,
normalizer and agent state for the relevant resume mode. Replay is separate and large.
A model-only restore is not an exact learning-state or physical/RNG-state continuation.

Retain the official licenses and attribution in both repositories. The original upstream
README remains below the research introduction. Historical notes may be in Chinese;
this README/handoff is intentionally English.


