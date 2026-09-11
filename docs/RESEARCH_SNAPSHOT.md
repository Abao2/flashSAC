# Research snapshot: STR companion repository

## Ownership and scope

This is a working research snapshot, including unsuccessful experiments. Upstream STR
provides the original task and assets; our local edits support FlashSAC integration,
transition semantics diagnostics, goal/reset handling and interactive evaluation.
Do not attribute our integration code or results to the original authors.

The enclosing `flashsac-robotics` monorepo contains the algorithm and diagnostic history.
This source is included at `third_party/simtoolreal/` by Git subtree (not a submodule).
Set `SIMTOOLREAL_ROOT` to that directory and add it to `PYTHONPATH` when a historical script requires it.
Use an Isaac Lab/Isaac Sim compatible Python environment; the local tested setup uses
Python 3.11 and Isaac Sim 5.1. Preserve upstream setup instructions below the fork README.

## Important code areas

- `isaacsimenvs/tasks/simtoolreal/`: Lab environment and action/observation/reward/reset utilities.
- `isaacsimenvs/cfg/task/SimToolReal.yaml`: registered task defaults; sparse FlashSAC overrides depend on these.
- `dextoolbench/`: task trajectories, native pretrained player and local functional diagnostics.
- `isaacsimenvs/tests/`, `tests/`, `scripts/`: local checks and experiment utilities.
- `docs/reproduction.md`: historical reproduction evidence and protocol limitations.

## Continuation rules for Codex and humans

1. Treat this as an evolving checkout. Inspect `git status` before editing; preserve concurrent work.
2. Read the companion `docs/RESEARCH_HANDOFF.md` and dated diagnostic reports before repeating tests.
3. Do not infer convergence from startup, finite losses, goal counts, or a single successful clip.
4. Keep true termination separate from time limits; timeout bootstrap needs pre-reset final observation.
5. Do not silently change reward, curriculum, control smoothing or task distribution to obtain success.
6. Do not start training, modify a remote machine or stop another process merely because old notes mention it.

## External files

`pretrained_policy/`, downloaded DexToolBench data, simulator binaries, caches, videos and
large model/replay arrays are excluded from new commits. They are not restored by `git clone`.
Use the upstream downloader for the official checkpoint and retain checksums for local artifacts.
The companion artifact inventory records what existed locally, not an uploaded backup URL.

Some historical notes contain machine-specific paths or Chinese text; they are retained as
experimental evidence. This handoff and the new README are intentionally in English.

