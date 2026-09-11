"""CPU-only calibration of the BC Actor's std head; mean policy/BN stay frozen."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def calibrate(source, dataset_path, output, target_std, updates=1500, batch_size=512,
              learning_rate=.003, seed=0, log_every=250, method="adam"):
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo), str(repo / "scripts")]
    import numpy as np
    import torch
    from diagnose_str_memory import episode_split, load_bc_checkpoint

    source, dataset_path, output = Path(source).resolve(strict=True), Path(dataset_path).resolve(strict=True), Path(output).resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite calibration: {output}")
    if not math.isfinite(target_std) or not math.exp(-10) < target_std < math.exp(2):
        raise ValueError("Target std must be strictly inside native exp([-10,2]) range")
    if updates < 1 or batch_size < 1 or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("Positive updates, batch size and finite learning rate required")
    if method not in ("adam", "min-variance"):
        raise ValueError("Unknown std calibration method")
    model, payload = load_bc_checkpoint(source, device="cpu")
    if payload["config"]["mode"] != "current" or payload["config"]["input_dim"] != 162:
        raise ValueError("Std calibration currently requires the current-state 162-dim Actor")
    with np.load(dataset_path, allow_pickle=False) as data:
        observations = np.asarray(data["obs"], dtype=np.float32)
        episode = np.asarray(data["episode_id"])
    if observations.ndim != 2 or observations.shape[1] != 162 or len(episode) != len(observations):
        raise ValueError("Expected obs[M,162] and episode_id[M]")
    if not np.isfinite(observations).all():
        raise ValueError("Nonfinite calibration observations")
    # Reuse BC's whole-episode holdout, rather than accidentally fitting on it.
    if payload.get("train_episode_ids") and payload.get("validation_episode_ids"):
        train = np.flatnonzero(np.isin(episode, payload["train_episode_ids"]))
        validation = np.flatnonzero(np.isin(episode, payload["validation_episode_ids"]))
        split_source = "original_BC_whole_episode_split"
    else:
        train, validation = episode_split(episode, seed, .25)
        split_source = "seeded_whole_episode_25_percent_holdout"
    if not len(train) or not len(validation) or np.intersect1d(episode[train], episode[validation]).size:
        raise ValueError("Need nonempty disjoint train/held-out episodes")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.eval().requires_grad_(False)
    model.predictor.std_w.requires_grad_(True)
    model.predictor.std_bias.requires_grad_(True)
    allowed = {"predictor.std_w.w.weight", "predictor.std_bias"}
    assert {name for name, value in model.named_parameters() if value.requires_grad} == allowed
    before = {name: value.clone() for name, value in model.state_dict().items()}
    tensor = torch.from_numpy(observations)

    @torch.no_grad()
    def all_predictions():
        means, deviations = [], []
        for start in range(0, len(tensor), batch_size):
            mean, std = model.get_mean_and_std(tensor[start:start + batch_size], training=False)
            means.append(mean)
            deviations.append(std)
        return torch.cat(means), torch.cat(deviations)

    def stats(std, rows):
        values = std[rows]
        return {"rows": len(rows), "episodes": int(len(np.unique(episode[rows]))),
                "std_min": float(values.min()), "std_mean": float(values.mean()), "std_max": float(values.max()),
                "std_mse": float((values - target_std).square().mean()),
                "log_std_mse": float((values.log() - math.log(target_std)).square().mean())}

    mean_before, std_before = all_predictions()
    started = time.monotonic()
    curve = []
    eigen_fit = None
    if method == "min-variance":
        with torch.no_grad():
            features = []
            for start in range(0, len(train), batch_size):
                x = model.embedder(tensor[train[start:start + batch_size]], training=False)
                for block in model.encoder:
                    x = block(x, training=False)
                features.append(model.post_norm(x).double())
            features = torch.cat(features)
            center = features.mean(dim=0)
            centered = features - center
            covariance = centered.T @ centered / (len(features) - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            direction = eigenvectors[:, 0]
            direction = direction / direction.norm()
            model.predictor.std_w.w.weight.copy_(direction.float().expand(29, -1))
            model.predictor.std_w.normalize_parameters()
            direction = model.predictor.std_w.w.weight[0].double()
            unit_interval = 2 * (math.log(target_std) - model.predictor.log_std_min) / (
                model.predictor.log_std_max - model.predictor.log_std_min) - 1
            raw_target = math.atanh(unit_interval)
            bias = raw_target - float(center @ direction)
            model.predictor.std_bias.fill_(bias)
            eigen_fit = {"feature_rows": len(features), "feature_dim": features.shape[1],
                         "covariance_dtype": "float64", "fit_uses_train_episodes_only": True,
                         "minimum_eigenvalue": float(eigenvalues[0]), "covariance_trace": float(covariance.trace()),
                         "projected_train_variance": float(direction @ covariance @ direction),
                         "raw_log_std_target_before_tanh_map": raw_target,
                         "centered_bias": bias, "unit_direction": direction.tolist(),
                         "all_29_std_rows_share_direction": True}
    else:
        optimizer = torch.optim.Adam([*model.predictor.std_w.parameters(), model.predictor.std_bias], lr=learning_rate)
        for step in range(1, updates + 1):
            rows = rng.choice(train, size=batch_size, replace=True)
            _, std = model.get_mean_and_std(tensor[rows], training=False)
            loss = (std.log() - math.log(target_std)).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite std calibration objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.predictor.std_w.normalize_parameters()
            if step == 1 or step % log_every == 0 or step == updates:
                record = {"step": step, "sampled_train_log_std_mse": float(loss.detach())}
                curve.append(record)
                print("STD_CALIBRATION " + json.dumps(record), flush=True)
    mean_after, std_after = all_predictions()
    assert torch.equal(mean_before, mean_after), "Calibration changed deterministic means"
    assert torch.equal(mean_before.tanh(), mean_after.tanh()), "Calibration changed deterministic actions"
    for name, value in model.state_dict().items():
        assert torch.isfinite(value).all(), name
        if name not in allowed:
            assert torch.equal(before[name], value), f"Frozen weight/BN modified: {name}"
    norms = model.predictor.std_w.w.weight.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0)
    assert not torch.cuda.is_initialized()
    report = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_bc_checkpoint": str(source), "source_bc_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "dataset": str(dataset_path), "target_pre_tanh_std": target_std,
              "method": method, "updates": updates if method == "adam" else 0,
              "batch_size": batch_size, "learning_rate": learning_rate if method == "adam" else None, "seed": seed,
              "eigen_fit": eigen_fit,
              "split_source": split_source, "train_episode_ids": np.unique(episode[train]).tolist(),
              "validation_episode_ids": np.unique(episode[validation]).tolist(),
              "before": {"train": stats(std_before, train), "validation": stats(std_before, validation)},
              "after": {"train": stats(std_after, train), "validation": stats(std_after, validation)},
              "validation": {"deterministic_means_exact": True, "deterministic_actions_exact": True,
                             "all_non_std_parameters_and_BN_exact": True,
                             "std_weight_row_norm_min_max": [float(norms.min()), float(norms.max())],
                             "cuda_initialized": False},
              "changed_keys": [name for name, value in model.state_dict().items() if not torch.equal(value, before[name])],
              "curve": curve, "wallclock_seconds": time.monotonic() - started,
              "notes": ["Only predictor.std_w.w.weight and predictor.std_bias were fitted. All means/torso/BN remain bitwise unchanged.",
                        ("Train-only frozen post_norm feature covariance in float64; minimum-variance unit direction and centered bias map to the requested log std. No gradient updates."
                         if method == "min-variance" else
                         "Objective: mean squared error in log pre-tanh std; unit-row projection after every Adam step."),
                        "Target is a fitted std on expert observations, NOT an enforced global constant or SAC entropy coefficient.",
                        "No SAC, Q, temperature or reward-normalizer training. Native stochastic rollout is still required."]}
    payload["model_state_dict"] = model.state_dict()
    payload["std_calibration"] = report
    payload["config"] = {**payload["config"], "std_head_calibrated": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("STD_CALIBRATION_RESULT " + json.dumps({"output": str(output), "after": report["after"],
                                                 "validation": report["validation"]}, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-std", type=float, required=True)
    parser.add_argument("--method", choices=["adam", "min-variance"], default="adam")
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    calibrate(args.bc_checkpoint, args.dataset, args.output, args.target_std,
              args.updates, args.batch_size, args.learning_rate, args.seed, method=args.method)
