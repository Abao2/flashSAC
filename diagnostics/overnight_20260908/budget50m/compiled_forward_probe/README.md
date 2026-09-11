# CUDA/Inductor forward probe — prepared, not yet run

CPU preparation and fixture checks completed. `summary.json` does not exist until the actual CUDA job finishes; preparation is not evidence of GPU equivalence.

- One existing-queue `diagnostic` job: 12GiB free-memory gate, 900s timeout, hard stop 2026-09-09 09:00 Asia/Shanghai.
- Parent must launch with `--after-status` pointing to `budget50m/native_entry_check/queue_status.json`; manifest dependency text alone does not enforce ordering.
- No Isaac, physics, replay insertion, network update, or post-load normalization. Original full checkpoint/native optimizer state is read, not changed.
- Actual native `create_agent(use_compile=true, compile_mode=auto)` and manually compiled `get_mean_and_std`, default CUDA Inductor; comparison invokes the existing `FrozenPolicy` eager loader.
- Same training TF32/high-matmul bundle on both sides. Frozen inference uses explicit `training=False`, without autocast, as native collection does.
- Recorded teacher/native reset, lifted-proxy and near-goal-center-proxy states; 32/1024 batches, contiguous162/strided324. Low-cardinality selections repeat rows and disclose uniqueness.
- Twelve actual native/Frozen cached-noise calls per batch, common RNG/cache states, effective GPU input strides recorded. No claim of independent sequence rollouts.
- Finite values and unchanged parameter/BN digests are hard checks. Max/RMSE/relative differences and error-threshold fractions are descriptive; `complete` does not mean differences must be zero or cannot amplify in contact dynamics.

CPU-only regenerate/check:

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/compiled_forward_probe/run_probe.py --prepare
```

Do not run without `--prepare` before the parent queues it: that starts the actual CUDA diagnostic. Prepared source hashes are checked before and after execution. If a source changes, inspect it and re-prepare deliberately.
