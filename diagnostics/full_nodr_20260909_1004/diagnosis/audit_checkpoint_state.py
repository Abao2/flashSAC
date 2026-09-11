"""CPU-only saved-state audit. No observation query, simulator, or model mutation.

Prints JSON; output may be saved separately. Predictor biases are parameters,
not observed action standard deviations or categorical probability masses.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from summarize_str_control_ab import checkpoint_audit, sha256


def stats(value):
    x = value.detach().double().reshape(-1)
    return {"mean": x.mean().item(), "std": x.std(correction=0).item(),
            "min": x.min().item(), "max": x.max().item()}


def weights(path, name):
    state = torch.load(path / f"{name}.pt", map_location="cpu", weights_only=True)
    return {k.removeprefix("_orig_mod."): v for k, v in state["network_state_dict"].items()}


def row_diversity(weight):
    matrices = weight.unsqueeze(0) if weight.ndim == 2 else weight
    result = []
    for matrix in matrices.double():
        unit = torch.nn.functional.normalize(matrix, dim=-1)
        cosine = unit @ unit.T
        off = cosine[~torch.eye(matrix.shape[0], dtype=torch.bool)]
        singular = torch.linalg.svdvals(matrix)
        centered = torch.linalg.svdvals(matrix - matrix.mean(dim=0, keepdim=True))
        result.append({"off_diagonal_cosine": stats(off),
            "adjacent_row_cosine_mean": cosine.diag(1).mean().item(),
            "stable_rank": (singular.square().sum() / singular[0].square()).item(),
            "largest_singular_value": singular[0].item(),
            "row_centered_stable_rank": (centered.square().sum() / centered[0].square().clamp_min(1e-30)).item(),
            "row_centered_frobenius_norm": centered.norm().item(),
            "row_centered_largest_singular_value": centered[0].item()})
    return result


def network_stats(state):
    bn = {}
    for key, var in state.items():
        if not key.endswith("running_var"):
            continue
        prefix = key.removesuffix("running_var")
        gain = state[prefix + "weight"] / (var + 1e-5).sqrt()
        top = gain.abs().flatten().topk(min(5, gain.numel()))
        bn[prefix.rstrip(".")] = {"running_mean": stats(state[prefix + "running_mean"]),
            "running_var": stats(var), "negative_var_count": int((var < 0).sum()),
            "var_below_1e-8_count": int((var < 1e-8).sum()), "inference_abs_gain": stats(gain.abs()),
            "top_abs_gain_flat_indices": top.indices.tolist(), "top_abs_gains": top.values.tolist()}
    linear = {k: stats(v.norm(dim=-1)) for k, v in state.items()
              if k.endswith("weight") and v.ndim >= 2 and "norm" not in k}
    predictor = {k: stats(v) for k, v in state.items() if k.startswith("predictor.") and "bin_values" not in k}
    diversity = {k: row_diversity(v) for k, v in state.items()
                 if k.startswith("predictor.") and k.endswith("weight")}
    return {"batch_norm": bn, "linear_row_norms": linear, "predictor_parameters": predictor,
            "predictor_row_diversity": diversity,
            "post_norm_scale": stats(state["post_norm.weight"])}


def differences(before, after, buffer_only=False):
    keys = [k for k in before if before[k].is_floating_point() and
            (k.endswith("running_mean") or k.endswith("running_var")) == buffer_only and k != "predictor.bin_values"]
    delta = torch.cat([(after[k] - before[k]).flatten().double() for k in keys])
    base = torch.cat([before[k].flatten().double() for k in keys])
    return {"elements": delta.numel(), "rmse": delta.square().mean().sqrt().item(),
            "l2": delta.norm().item(), "relative_l2": (delta.norm() / base.norm().clamp_min(1e-30)).item()}


def audit_one(path, num_envs, updates_per_step, g_max):
    checked = checkpoint_audit(path)
    assert checked["all_finite"], path
    actor, critic, target = [weights(path, name) for name in ("actor", "critic", "target_critic")]
    norm = torch.load(path / "reward_normalizer.pt", map_location="cpu", weights_only=True)
    alpha = float(weights(path, "temperature")["log_temp"].exp())
    step = int(path.name.removeprefix("step"))
    expected = (step - (100000 - 1) // num_envs) * updates_per_step
    calls = checked["files"]["agent_state.pt"]["stored_update_step"]
    assert calls == expected and float(norm["G_rms_count"]) == step * num_envs
    fields = {"sha256", "all_finite", "optimizer_applied_steps", "scheduler_last_epoch", "stored_update_step", "grad_scaler"}
    files = {name: {k: v for k, v in data.items() if k in fields} for name, data in checked["files"].items()}
    applied = {}
    for name, opportunities in (("actor", calls // 2), ("critic", calls), ("temperature", calls // 2)):
        payload = torch.load(path / f"{name}.pt", map_location="cpu", weights_only=True)
        steps = files[f"{name}.pt"]["optimizer_applied_steps"]
        assert len(steps) == 1
        applied[name] = {"opportunities": opportunities, "applied_steps": steps[0],
            "skipped_steps": opportunities - steps[0], "skipped_fraction": 1 - steps[0] / opportunities,
            "optimizer_lrs": [p["lr"] for p in payload["optimizer_state_dict"]["param_groups"]]}
    sd, mx = float(norm["G_rms_var"].sqrt()), float(norm["G_r_max"])
    denominator = max(math.sqrt(float(norm["G_rms_var"]) + 1e-8), mx / g_max)
    bins = critic["predictor.bin_values"].flatten()
    return {"path": str(path), "transitions": step * num_envs, "all_finite": True,
        "actor_obs_dim": actor["embedder.w.w.weight"].shape[-1],
        "critic_obs_dim": critic["embedder.w.weight"].shape[-1] - 29,
        "critic_support": [float(bins[0]), float(bins[-1]), bins.numel()],
        "native_update_calls": calls, "normalizer_count": float(norm["G_rms_count"]),
        "alpha": alpha, "optimizer": applied, "files": files,
        "reward_normalization": {"denominator": denominator, "return_std": sd,
            "historical_max_abs_return": mx, "max_constraint_denominator": mx / g_max,
            "dominant_branch": "historical_max" if mx / g_max >= sd else "variance",
            "return_running_mean": float(norm["G_rms_mean"]), "last_return_vector": stats(norm["G_r"]),
            "raw_300_scaled": 300 / denominator, "raw_1000_scaled": 1000 / denominator},
        "actor": network_stats(actor), "critic": network_stats(critic),
        "actor_std_bias_vector": actor["predictor.std_bias"].tolist(),
        "critic_target_parameter_gap": differences(critic, target),
        "critic_target_bn_gap": differences(critic, target, buffer_only=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.self_test:
        assert stats(torch.tensor([1., 3.]))["mean"] == 2
        a = {"w": torch.tensor([1., 2.]), "x.running_mean": torch.tensor([1.])}
        b = {"w": torch.tensor([2., 3.]), "x.running_mean": torch.tensor([3.])}
        assert differences(a, b)["rmse"] == 1 and differences(a, b, True)["rmse"] == 2
        test = row_diversity(torch.eye(2))[0]
        assert test["stable_rank"] == 2 and test["off_diagonal_cosine"]["max"] == 0
        print("CPU self-test passed")
        return
    root = Path(__file__).resolve().parent.parent
    old = REPO / "diagnostics/overnight_20260908/repro60m"
    full_cfg = json.loads((root / "preflight_gpu.json").read_text())["resolved_training_config"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "cpu_threads": 2,
        "scope": "Saved-state inspection only; no forward inference, rollout, evaluation, training, or optimizer changes.",
        "source_hashes": {str(p): sha256(p) for p in [Path(__file__), root / "preflight_gpu.json",
            REPO / "flash_rl/agents/flashSAC/layer.py", REPO / "flash_rl/agents/utils/reward_normalization.py"]},
        "full": {}, "simple_reference": {}, "full_20m_to100m_parameter_changes": {},
        "limitations": ["No full-task observation or replay arrays saved under this run; preflight contains summary/config only.",
            "No dummy observations are queried. Q ranking, atom probability/clipping, actual policy entropy/std and state-dependent BN mismatch are not identified from weights alone.",
            "Actor140 full-task and actor162 simple-task have different information and task distributions; parameter differences are descriptive, not a controlled causal test.",
            "Std-head bias is not log_std: learned hidden features contribute, followed by bounded tanh mapping [-10,2].",
            "Critic output row norms are constrained by native parameter normalization; healthy norms do not establish accurate Q values."]}
    for seed in range(3):
        key = str(seed)
        report["full"][key] = {str(step): audit_one(root / "models" / f"seed{seed}" / f"step{step}", 2048, 4,
            full_cfg["agent"]["normalized_G_max"]) for step in [9766, 19532, 29298, 39064, 48830]}
        report["simple_reference"][key] = audit_one(old / "models" / f"seed{seed}" / "step58596", 1024, 2, 5.)
        change = {}
        for model in ("actor", "critic"):
            a = weights(root / "models" / f"seed{seed}" / "step9766", model)
            b = weights(root / "models" / f"seed{seed}" / "step48830", model)
            change[model] = {"parameters": differences(a, b), "bn_buffers": differences(a, b, True)}
        report["full_20m_to100m_parameter_changes"][key] = change
    print(json.dumps(report, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
