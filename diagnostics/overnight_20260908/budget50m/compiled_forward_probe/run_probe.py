"""Prepare on CPU; parent may later run actual CUDA/Inductor inference parity (no Isaac)."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
BUDGET = ROOT.parent
REPO = BUDGET.parents[2]
PYTHON = "/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python"
CHECKPOINT = BUDGET / "models/arm1_hand01/seed0/step48830"
CONFIG = BUDGET / "configs/arm1_hand01_seed0.json"
DEADLINE = "2026-09-09T01:00:00Z"
DATASETS = [BUDGET.parent / "official_filter01_labeled128/transitions.npz",
            BUDGET / "evaluations/arm1_hand01_seed0/step48830/deterministic/transitions.npz"]
PHASES = ["reset", "lifted_proxy", "near_goal_center_proxy"]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = a - b
    return {"exact_equal": bool(np.array_equal(a, b)), "max_abs": float(np.abs(delta).max()),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-12)),
            "fractions_abs_gt": {str(t): float((np.abs(delta) > t).mean()) for t in [1e-5, 1e-4, 1e-3]}}


def state_selection():
    rng = np.random.default_rng(9919)
    result = {}
    for path in DATASETS:
        with np.load(path, allow_pickle=False) as data:
            assert data["obs"].shape[1] == 162
            by_phase = {}
            for phase in PHASES:
                candidates = np.flatnonzero(data["phase"] == phase)
                assert len(candidates)
                rows = np.resize(rng.permutation(candidates), 512)  # Repeats disclosed for small reset pools.
                by_phase[phase] = {"rows": rows.tolist(), "available": len(candidates), "unique_selected": len(np.unique(rows))}
            result[str(path)] = by_phase
    return result


def prepare():
    import torch
    assert not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA"
    selected = state_selection()
    assert compare([1., 2.], [1., 2.])["max_abs"] == 0
    assert compare([1., 2.], [1., 3.])["rmse"] == np.sqrt(.5)
    for phase in PHASES:
        combined = np.stack([selected[str(path)][phase]["rows"] for path in DATASETS], axis=1).reshape(-1)
        assert len(combined) == 1024 and len(combined[:32]) == 32
    (ROOT / "state_selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    source_paths = [Path(__file__), CONFIG, *DATASETS, ROOT / "state_selection.json",
        REPO / "flash_rl/agents/flashSAC/agent.py", REPO / "flash_rl/agents/flashSAC/network.py",
        REPO / "flash_rl/agents/flashSAC/layer.py", REPO / "flash_rl/agents/utils/network.py",
        REPO / "scripts/diagnose_str_rollouts.py", *sorted(CHECKPOINT.glob("*.pt"))]
    manifest = {"phase": "actual_cuda_inductor_frozen_forward_parity", "prepared_only": True,
        "launch_authority": "Parent only; --prepare runs CPU selection/assertions only and never launches GPU.",
        "hard_deadline_utc": DEADLINE, "required_prior_status": str(BUDGET / "native_entry_check/queue_status.json"),
        "launch_guard": "Parent must pass required_prior_status to queue --after-status; metadata alone is not enforced.",
        "checkpoint": str(CHECKPOINT), "requested_compile_mode": json.loads(CONFIG.read_text())["agent"]["compile_mode"],
        "expected_source_sha256": {str(p): sha(p) for p in source_paths},
        "limits": ["No simulator, optimizer update, replay insertion, BN update or normalization after checkpoint load.",
            "Native original use_compile=true and original auto mode; real default Inductor backend including manually compiled get_mean_and_std.",
            "Full checkpoint loaded by native agent.load; actual FrozenPolicy class actor-only eager loading on comparison side.",
            "Small replay capacity1/batch1 to avoid original10M allocation; not a training test.",
            "Three recorded phases,32 and1024 batches, contiguous162 and strided324 row storage; low-cardinality state repetition disclosed.",
            "12 native cached-noise calls per batch using actual NumPy324 collector inputs, paired CPU/CUDA RNG and cache states.",
            "Finite/digest invariants are hard checks; numerical differences are measured, not arbitrarily required to be exactly0.",
            "Expected small floating-point differences can still amplify in contact dynamics; no rollout-success inference."],
        "jobs": [{"id": "cuda_inductor_forward_parity", "kind": "diagnostic",
            "argv": [PYTHON, str(Path(__file__).resolve())], "cwd": str(REPO),
            "env": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": f"{REPO}:{REPO.parent / 'simtoolreal'}",
                    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TORCHINDUCTOR_COMPILE_THREADS": "2"},
            "log": str(ROOT / "logs/probe.log"), "timeout_seconds": 900, "gpu_index": 0,
            "expected_completion_file": str(ROOT / "summary.json")} ]}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "validation.json").write_text(json.dumps({"status": "cpu_preparation_checked_gpu_not_run",
        "cuda_initialized": torch.cuda.is_initialized(), "jobs": 1, "queue_free_memory_gate_mib": 12288,
        "timeout_seconds": 900, "hard_deadline_utc": DEADLINE, "selected_phase_rows": 3072,
        "selection_note": "Each phase alternates512 teacher and512 native rows; first32 has16 of each. Near-goal is center-proximity phase, not true pose success."}, indent=2) + "\n")
    print("Prepared one serial CUDA diagnostic; GPU NOT launched.")


def canonical_state(model):
    return {key.removeprefix("_orig_mod."): value.detach().cpu().contiguous() for key, value in model.state_dict().items()}


def digests(model):
    state = canonical_state(model)
    result = {}
    for group, selected in [("all_parameters_and_buffers", state),
                            ("batchnorm_buffers", {k: v for k, v in state.items() if "running_" in k})]:
        digest = hashlib.sha256()
        for key, value in sorted(selected.items()):
            digest.update(key.encode()); digest.update(value.numpy().tobytes())
        result[group] = {"sha256": digest.hexdigest(), "tensors": len(selected), "finite": all(bool(v.isfinite().all()) for v in selected.values())}
    return result


def cpu_array(value):
    return value.detach().float().cpu().numpy().copy()


def run_gpu():
    import gymnasium as gym
    import torch
    from omegaconf import OmegaConf
    from flash_rl.agents import create_agent
    from diagnose_str_rollouts import FrozenPolicy
    from torch._dynamo.utils import counters

    started = time.monotonic()
    assert datetime.now(timezone.utc) < datetime.fromisoformat(DEADLINE.replace("Z", "+00:00"))
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert all(sha(path) == expected for path, expected in manifest["expected_source_sha256"].items()), "Prepared sources changed; re-audit before launch"
    assert os.environ.get("TORCH_COMPILE_DISABLE", "0") not in ("1", "True", "true")
    assert os.environ.get("TORCHDYNAMO_DISABLE", "0") not in ("1", "True", "true")
    assert torch.cuda.is_available(), "This test requires actual CUDA, not CPU/eager fallback"
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = False
    torch.cuda.set_per_process_memory_fraction(min(11 * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, .9))
    torch.cuda.reset_peak_memory_stats()
    cfg = OmegaConf.create(json.loads(CONFIG.read_text()))
    cfg.agent.buffer_max_length = cfg.agent.buffer_min_length = cfg.agent.sample_batch_size = 1
    assert cfg.agent.use_compile and cfg.agent.load_optimizer and cfg.agent.load_reward_normalizer
    fake = SimpleNamespace(device="cuda:0", observation_space=gym.spaces.Box(-np.inf, np.inf, (1024, 324), np.float32),
                           action_space=gym.spaces.Box(-1., 1., (1024, 29), np.float32))
    info = {"actor_observation_size": (162,), "critic_observation_size": (162,), "critic_observation_offset": 162}
    native = create_agent(fake.observation_space, fake.action_space, info, cfg.agent)
    native.load(str(CHECKPOINT))
    assert native._update_step == 97466
    method = native._actor.network.get_mean_and_std
    assert hasattr(method, "_torchdynamo_orig_callable"), "Manual get_mean_and_std compilation missing"
    frozen_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    frozen_cfg.agent.use_compile = frozen_cfg.agent.load_optimizer = frozen_cfg.agent.load_reward_normalizer = False
    args = SimpleNamespace(policy="flash", checkpoint=CHECKPOINT, num_envs=1024,
                           sampling="stochastic", noise_multiplier=1., noise_repeat=0)
    frozen = FrozenPolicy(args, frozen_cfg, fake, info, {}, [])
    state_n, state_f = canonical_state(native._actor.network), canonical_state(frozen.model)
    assert state_n.keys() == state_f.keys() and all(torch.equal(state_n[k], state_f[k]) for k in state_n)
    models = {"native_" + name: getattr(native, "_" + name).network for name in ["actor", "critic", "target_critic", "temperature"]}
    models["frozen_actor"] = frozen.model
    before = {key: digests(value) for key, value in models.items()}
    selected = json.loads((ROOT / "state_selection.json").read_text())
    observations = {}
    for phase in PHASES:
        parts = []
        for path in DATASETS:
            with np.load(path, allow_pickle=False) as data:
                parts.append(data["obs"][selected[str(path)][phase]["rows"]])
        observations[phase] = np.stack(parts, axis=1).reshape(1024, 162).copy()
    report = {"status": "running_cuda_forward_measurement", "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0), "requested_compile_mode": str(cfg.agent.compile_mode),
        "resolved_compile_mode": native._cfg.compile_mode, "backend": "actual default Inductor; no monkeypatch/backend=eager",
        "manual_get_mean_and_std_is_torchdynamo_wrapped": True, "loaded_native_update_step": native._update_step,
        "inference_training_argument": False, "autocast_enabled": False,
        "numerics": {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                     "matmul_precision": torch.get_float32_matmul_precision(), "cudnn_benchmark": torch.backends.cudnn.benchmark},
        "sources": manifest["expected_source_sha256"], "selection_file": str(ROOT / "state_selection.json"),
        "loaded_actor_tensors_exactly_equal": True, "model_digests_before": before, "cases": [], "sampler_sequences": []}
    counter_before = {name: dict(value) for name, value in counters.items()}
    with torch.no_grad():
        for phase, array in observations.items():
            for batch in [32, 1024]:
                epsilon = torch.randn((batch, 29), generator=torch.Generator().manual_seed(7300 + batch)).cuda()
                full = torch.from_numpy(np.concatenate([array[:batch], array[:batch]], axis=-1)).cuda()
                for layout in ["strided324", "contiguous162"]:
                    x = full[:, :162] if layout == "strided324" else full[:, :162].contiguous()
                    # This apply invokes the MANUALLY COMPILED method, not the compiled forward namespace.
                    mu, sigma = native._actor.apply("get_mean_and_std", observations=x, training=False)
                    mu, sigma = mu.clone(), sigma.clone()
                    fmu, fsigma = frozen.agent._actor.apply("get_mean_and_std", observations=x, training=False)
                    result = {"phase": phase, "batch": batch, "layout": layout, "input_stride": list(x.stride()),
                        "mean": compare(cpu_array(mu), cpu_array(fmu)), "std": compare(cpu_array(sigma), cpu_array(fsigma)),
                        "deterministic_tanh_mean": compare(cpu_array(mu.tanh()), cpu_array(fmu.tanh())),
                        "paired_fixed_epsilon_actions": compare(cpu_array((mu + sigma * epsilon).tanh()), cpu_array((fmu + fsigma * epsilon).tanh()))}
                    report["cases"].append(result)
                    print("FORWARD_CASE " + json.dumps(result), flush=True)
        # Native train copies324 NumPy cols toGPU before slicing; FrozenPolicy copies its162 slice.
        # Use these ACTUAL APIs and record the effective strides instead of assuming they match.
        calls = {"native": [], "frozen": []}
        for label, agent in [("native", native), ("frozen", frozen.agent)]:
            original_apply = agent._actor.apply
            def traced(method, *pos, _label=label, _original=original_apply, **kwargs):
                if method == "get_mean_and_std":
                    calls[_label].append(list(kwargs["observations"].stride()))
                return _original(method, *pos, **kwargs)
            agent._actor.apply = traced
        for batch in [32, 1024]:
            def reset_cache(agent):
                agent._cached_noise = torch.zeros((batch, 29), device="cuda")
                agent._cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device="cuda")
                agent._cur_noise_repeat_n = torch.tensor(1, dtype=torch.int32, device="cuda")
            for agent in [native, frozen.agent]:
                reset_cache(agent)
            # Warm compile helper RNG functions first; then start both from identical caches/RNG.
            warm = np.concatenate([observations["reset"][:batch]] * 2, axis=-1)
            native.sample_actions(0, {"next_observation": warm}, training=True)
            frozen.actions(warm, 0)
            for agent in [native, frozen.agent]:
                reset_cache(agent)
            torch.manual_seed(8310 + batch)
            for step in range(12):
                phase = PHASES[step % len(PHASES)]
                full_np = np.concatenate([observations[phase][:batch]] * 2, axis=-1)
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                na = native.sample_actions(step + 1, {"next_observation": full_np}, training=True)
                after_cpu, after_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
                fa = frozen.actions(full_np, step + 1)
                rng_equal = torch.equal(after_cpu, torch.get_rng_state()) and torch.equal(after_cuda, torch.cuda.get_rng_state())
                caches_equal = all(torch.equal(getattr(native, key), getattr(frozen.agent, key))
                    for key in ["_cached_noise", "_cur_noise_repeat_count", "_cur_noise_repeat_n"])
                assert rng_equal and caches_equal, "Noise/RNG not paired; action deltas would be confounded"
                report["sampler_sequences"].append({"batch": batch, "step": step + 1, "phase": phase,
                    "native_input_stride": calls["native"][-1], "frozen_input_stride": calls["frozen"][-1],
                    "rng_equal": rng_equal, "noise_cache_count_repeat_equal": caches_equal,
                    "noise_count": native._cur_noise_repeat_count.item(), "repeat_n": native._cur_noise_repeat_n.item(),
                    "actions": compare(na, fa)})
    after = {key: digests(value) for key, value in models.items()}
    assert before == after and all(value["all_parameters_and_buffers"]["finite"] for value in after.values())
    assert native._update_step == 97466 and len(native._replay_buffer) == 0
    graph_delta = int(counters["stats"]["unique_graphs"]) - int(counter_before.get("stats", {}).get("unique_graphs", 0))
    assert graph_delta > 0, "No new Dynamo graphs from the actual method calls; cannot certify compiled execution"
    assert all(sha(path) == expected for path, expected in manifest["expected_source_sha256"].items())
    report.update(status="complete", wallclock_seconds=time.monotonic() - started, model_digests_after=after,
        model_parameters_and_BN_unchanged=True, optimizer_calls=0, replay_insertions=0,
        dynamo_counters_before_forward=counter_before, dynamo_counters_after={name: dict(value) for name, value in counters.items()},
        new_dynamo_graphs_during_forward_probe=graph_delta,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        interpretation="Descriptive finite forward-difference measurement, not pass==zero and not a rollout-success test.",
        limitations=manifest["limits"])
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Actual CUDA/Inductor forward parity", "", "Measured differences; no arbitrary exact-zero pass criterion. No physics or optimizer updates.", "",
        "| phase | batch | stride | mu max/RMSE | sigma max/RMSE | tanh mean max | fixed-noise action max |", "| --- | ---: | --- | --- | --- | ---: | ---: |"]
    for item in report["cases"]:
        lines.append(f"| {item['phase']} | {item['batch']} | {item['layout']} | {item['mean']['max_abs']:.6g}/{item['mean']['rmse']:.6g} | {item['std']['max_abs']:.6g}/{item['std']['rmse']:.6g} | {item['deterministic_tanh_mean']['max_abs']:.6g} | {item['paired_fixed_epsilon_actions']['max_abs']:.6g} |")
    lines += ["", f"24 sampler comparisons (12 calls ×2 batches), same RNG/noise caches. Max action difference: {max(x['actions']['max_abs'] for x in report['sampler_sequences']):.6g}.",
        "Actor/native critic/target/temp parameters and BN buffers unchanged and finite. JSON retains actual input strides, compiler counters, all source hashes and descriptive tolerances."]
    (ROOT / "analysis.md").write_text("\n".join(lines) + "\n")
    print("Completed actual CUDA forward measurement; inspect summary differences, not a zero-equality pass label.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="CPU validation and one-job manifest ONLY; never initialize CUDA")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run_gpu()
