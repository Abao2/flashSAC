"""Episode-held-out memory / behavior-cloning probes; no simulator is imported.

Input NPZ: obs,next_obs [M,162], actions [M,29], rewards, terminated,
truncated, episode_id, step_in_episode. Optional teacher_actions, phase, success and metadata_json
(a JSON string). next_obs must describe the actual post-action, pre-reset state;
terminal rows are excluded from prediction unless metadata confirms that contract.

BC uses the real FlashSACActor, not a stand-in MLP. Only its deterministic mean
is fitted; its exploration/std head is not trained. BC MSE is NOT closed-loop
success. Predictability from history is NOT evidence that LSTM is necessary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flash_rl.agents.flashSAC.network import FlashSACActor


# Exact state162_eraser layout, verified against STR obs_utils. No Isaac import.
OBS_FIELDS = [
    "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot",
    "palm_vel", "object_rot", "object_vel", "fingertip_pos_rel_palm",
    "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
    "closest_keypoint_max_dist", "closest_fingertip_dist", "lifted_object",
    "progress", "successes", "reward",
]
TARGET_COLUMNS = list(range(104, 110)) + list(range(137, 149))


def load_dataset(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata_raw = data.pop("metadata_json", data.pop("metadata", np.array("{}")))
    metadata = json.loads(str(metadata_raw.item()))
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    n = len(data["obs"])
    for name, shape in (("obs", (n, 162)), ("next_obs", (n, 162)), ("actions", (n, 29))):
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f"{name}: expected finite array {shape}")
        data[name] = data[name].astype(np.float32, copy=False)
    for name in ("rewards", "terminated", "truncated", "episode_id", "step_in_episode"):
        data[name] = np.asarray(data[name]).reshape(-1)
        if data[name].shape != (n,) or not np.isfinite(data[name]).all():
            raise ValueError(f"{name}: expected finite vector of length {n}")
    for name in ("episode_id", "step_in_episode"):
        if np.any(data[name] != data[name].astype(np.int64)):
            raise ValueError(f"{name}: noninteger IDs/steps")
        data[name] = data[name].astype(np.int64)
    if np.any(data["step_in_episode"] < 0):
        raise ValueError("negative step_in_episode")
    for name in ("terminated", "truncated"):
        if not np.isin(data[name], [0, 1]).all():
            raise ValueError(f"{name} must contain booleans")
        data[name] = data[name].astype(bool)
    if np.abs(data["actions"]).max() > 1.00001:
        raise ValueError("BC actions must be executed normalized/clipped actions in [-1,1]")
    if "teacher_actions" in data:
        labels = data["teacher_actions"]
        if labels.shape != (n, 29) or not np.isfinite(labels).all() or np.abs(labels).max() > 1.00001:
            raise ValueError("teacher_actions: expected finite normalized/clipped labels [M,29] in [-1,1]")
        data["teacher_actions"] = labels.astype(np.float32, copy=False)
    for field_key in ("obs_fields", "state_fields"):
        if field_key in metadata and metadata[field_key] != OBS_FIELDS:
            raise ValueError(f"Unexpected {field_key} layout; prediction columns would be invalid")
    if "phase" in data and np.asarray(data["phase"]).shape != (n,):
        raise ValueError("phase must have one entry per transition")
    return data, metadata


def history_indices(data: dict, window: int) -> np.ndarray:
    """Oldest -> current; pad with the first available state in each contiguous segment.

    Supports env-interleaved storage. Never looks ahead, crosses an episode, or
    crosses a missing-step gap. Reused IDs after termination are rejected.
    """
    if window < 1:
        raise ValueError("window must be positive")
    episode, step = data["episode_id"], data["step_in_episode"]
    order = np.lexsort((step, episode))
    result = np.empty((len(order), window), dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(episode[order])) + 1, len(order)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        rows = order[lo:hi]
        diffs = np.diff(step[rows])
        if (diffs == 0).any():
            raise ValueError("Duplicate (episode_id, step_in_episode)")
        if (data["terminated"][rows[:-1]] | data["truncated"][rows[:-1]]).any():
            raise ValueError("Episode ID reused after terminal/truncated transition")
        starts = np.maximum.accumulate(np.r_[0, np.where(diffs != 1, np.arange(1, len(rows)), 0)])
        offsets = np.arange(len(rows))[:, None] - np.arange(window - 1, -1, -1)[None, :]
        result[rows] = rows[np.maximum(offsets, starts[:, None])]
    return result


def episode_split(episode: np.ndarray, seed: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(episode)
    if len(ids) < 2 or not 0 < fraction < 1:
        raise ValueError("Need >=2 episodes and 0 < validation fraction < 1")
    ids = np.random.default_rng(seed).permutation(ids)
    count = min(len(ids) - 1, max(1, int(round(len(ids) * fraction))))
    val = np.isin(episode, ids[:count])
    return np.flatnonzero(~val), np.flatnonzero(val)


def selected_inputs(data: dict, history: np.ndarray, rows: np.ndarray, mode: str) -> np.ndarray:
    if mode == "current":
        return data["obs"][rows]
    if mode == "repeat":
        return np.tile(data["obs"][rows], (1, history.shape[1]))
    return data["obs"][history[rows]].reshape(len(rows), -1)


def successful_ids(data: dict, metadata: dict) -> np.ndarray:
    ids = list(metadata.get("successful_episode_ids", []))
    if "success" in data:
        flags = np.asarray(data["success"]).reshape(-1)
        if len(flags) != len(data["obs"]) or not np.isin(flags, [0, 1]).all():
            raise ValueError("success must be a boolean per-transition marker")
        ids.extend(data["episode_id"][flags.astype(bool)].tolist())
    return np.unique(np.asarray(ids, dtype=np.int64))


def create_model(config: dict) -> nn.Module:
    if config["task"] == "bc":
        return FlashSACActor(config["num_blocks"], config["input_dim"], config["width"], 29)
    return nn.Sequential(
        nn.Linear(config["input_dim"], config["width"]), nn.ReLU(),
        nn.Linear(config["width"], config["width"]), nn.ReLU(),
        nn.Linear(config["width"], len(TARGET_COLUMNS) + 1),
    )


@torch.no_grad()
def normalize_flash(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "normalize_parameters"):
            module.normalize_parameters()


def load_bc_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load only locally produced probe checkpoints; caller supplies per-env history."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["format"] != "str_memory_probe_v1" or payload["config"]["task"] != "bc":
        raise ValueError("Not a behavior-cloning probe checkpoint")
    model = create_model(payload["config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def predict_bc(model: nn.Module, payload: dict, obs_history: torch.Tensor | np.ndarray) -> torch.Tensor:
    """history [N,k,162], oldest -> newest, left-padded using THIS episode's first obs.

    Reinitialize the calling env's history on EVERY reset. Deterministic only:
    the saved policy std head was not optimized by BC.
    """
    cfg = payload["config"]
    obs_history = torch.as_tensor(obs_history, dtype=torch.float32, device=next(model.parameters()).device)
    if obs_history.ndim != 3 or obs_history.shape[-1] != 162:
        raise ValueError("Expected obs_history [N,k,162]")
    if cfg["mode"] == "current":
        x = obs_history[:, -1]
    elif cfg["mode"] == "repeat":
        x = obs_history[:, -1].repeat(1, cfg["window"])
    else:
        if obs_history.shape[1] != cfg["window"]:
            raise ValueError("History window does not match checkpoint")
        x = obs_history.flatten(1)
    x = x.to(next(model.parameters()).device, dtype=torch.float32)
    mean, _ = model.get_mean_and_std(x, training=False)
    return mean.tanh()


def run_probe(args, data: dict, metadata: dict, history: np.ndarray, task: str, mode: str) -> dict:
    good_ids = successful_ids(data, metadata)
    excluded = 0
    eligible = np.arange(len(data["obs"]))
    if task == "bc" and args.bc_only_success:
        eligible = eligible[np.isin(data["episode_id"], good_ids)]
    if len(np.unique(data["episode_id"][eligible])) < 2:
        return {"task": task, "mode": mode, "status": "skipped", "reason": "Need >=2 eligible whole episodes"}
    train_local, val_local = episode_split(data["episode_id"][eligible], args.seed, args.validation_fraction)
    train, val = eligible[train_local], eligible[val_local]
    if task == "prediction" and not metadata.get("next_obs_is_pre_reset", False):
        valid = ~(data["terminated"] | data["truncated"])
        excluded = int((~valid).sum())
        train, val = train[valid[train]], val[valid[val]]
    if len(train) < 2 or not len(val):
        return {"task": task, "mode": mode, "status": "skipped", "reason": "Insufficient episode-held-out rows after filtering"}

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # Statistics use train rows ONLY. For BC preserve the original Actor's native BN.
    obs_mean = data["obs"][train].mean(0) if task == "prediction" else np.zeros(162, np.float32)
    obs_std = np.maximum(data["obs"][train].std(0), 1e-4) if task == "prediction" else np.ones(162, np.float32)
    act_mean, act_std = data["actions"][train].mean(0), np.maximum(data["actions"][train].std(0), 1e-4)
    if task == "bc":
        if args.bc_action_key not in data:
            raise ValueError(f"Missing BC label array {args.bc_action_key!r}; refusing to substitute noisy executed actions")
        target = data[args.bc_action_key]
        target_mean, target_std = np.zeros(29, np.float32), np.ones(29, np.float32)
    else:
        delta = data["next_obs"][:, TARGET_COLUMNS] - data["obs"][:, TARGET_COLUMNS]
        target = np.column_stack((delta, data["rewards"])).astype(np.float32)
        target_mean, target_std = target[train].mean(0), np.maximum(target[train].std(0), 1e-4)

    baseline_mean = target[train].mean(0)
    config = {"task": task, "mode": mode, "window": args.window, "width": args.width,
              "num_blocks": args.num_blocks, "input_dim": 162 * (1 if mode == "current" else args.window) + (29 if task == "prediction" else 0),
              "obs_dim": 162, "action_dim": 29, "target_columns": TARGET_COLUMNS,
              "target_names": ["object_velocity_delta[6]", "goal_keypoint_delta[12]", "raw_step_reward"] if task == "prediction" else [f"{args.bc_action_key}[29]"],
              "bc_action_key": args.bc_action_key if task == "bc" else None,
              "prediction_action_key": "actions" if task == "prediction" else None,
              "normalization": "identity_input_native_UnitBatchNorm" if task == "bc" else "train_only_mean_std",
              "deterministic_only": task == "bc", "seed": args.seed}
    model = create_model(config).to(device)
    if task == "bc":
        normalize_flash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    repeat = 1 if mode == "current" else args.window
    xmean, xstd = np.tile(obs_mean, repeat), np.tile(obs_std, repeat)

    def batch(rows):
        x = (selected_inputs(data, history, rows, mode) - xmean) / xstd
        if task == "prediction":
            x = np.column_stack((x, (data["actions"][rows] - act_mean) / act_std))
        y = (target[rows] - target_mean) / target_std
        return torch.as_tensor(x, dtype=torch.float32, device=device), torch.as_tensor(y, dtype=torch.float32, device=device)

    def forward(x, training):
        return model.get_mean_and_std(x, training)[0].tanh() if task == "bc" else model(x)

    @torch.no_grad()
    def metrics(rows):
        # Same deterministic selection for each architecture; avoid huge eval tensors.
        if len(rows) > args.max_eval_rows:
            rows = np.random.default_rng(args.seed + 1).choice(rows, args.max_eval_rows, replace=False)
        residual, baseline = [], []
        for start in range(0, len(rows), args.batch_size):
            selected = rows[start:start + args.batch_size]
            x, y = batch(selected)
            residual.append((forward(x, False) - y).cpu().numpy())
            baseline.append((baseline_mean - target[selected]) / target_std)
        error, base = np.concatenate(residual), np.concatenate(baseline)
        raw_error = error * target_std
        answer = {"rows": int(len(rows)), "normalized_mse": float(np.square(error).mean()),
                  "train_mean_baseline_normalized_mse": float(np.square(base).mean()),
                  "raw_rmse": float(np.sqrt(np.square(raw_error).mean()))}
        if task == "prediction":
            answer.update(object_velocity_delta_rmse=float(np.sqrt(np.square(raw_error[:, :6]).mean())),
                          goal_keypoint_delta_rmse_m=float(np.sqrt(np.square(raw_error[:, 6:18]).mean())),
                          step_reward_rmse=float(np.sqrt(np.square(raw_error[:, -1]).mean())))
        else:
            answer.update(arm_action_rmse=float(np.sqrt(np.square(raw_error[:, :7]).mean())),
                          hand_action_rmse=float(np.sqrt(np.square(raw_error[:, 7:]).mean())))
        phase = data.get("phase", np.where(data["obs"][:, 158] > 0.5, "lifted_flag", "not_lifted_flag"))[rows]
        answer["by_phase"] = {str(p): {"rows": int((phase == p).sum()), "normalized_mse": float(np.square(error[phase == p]).mean())} for p in np.unique(phase)}
        return answer

    started = time.monotonic()
    curve = []
    for update in range(1, args.updates + 1):
        rows = rng.choice(train, size=args.batch_size, replace=True)
        x, y = batch(rows)
        loss = (forward(x, True) - y).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite {task}/{mode} loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if task == "bc":
            normalize_flash(model)
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            point = {"update": update, "batch_loss": float(loss.detach().cpu())}
            curve.append(point)
            print(json.dumps({"task": task, "mode": mode, **point}), flush=True)
    result = {"task": task, "mode": mode, "status": "complete", "config": config,
              "parameters": sum(p.numel() for p in model.parameters()), "updates": args.updates,
              "train_rows": int(len(train)), "validation_rows": int(len(val)),
              "train_episodes": np.unique(data["episode_id"][train]).tolist(),
              "validation_episodes": np.unique(data["episode_id"][val]).tolist(),
              "excluded_terminal_prediction_rows": excluded,
              "successful_episodes_in_dataset": good_ids.tolist(), "bc_success_only": args.bc_only_success,
              "claim": "Offline behavior imitation only; closed-loop success NOT measured" if task == "bc" else "History predictability only; does NOT establish LSTM necessity",
              "train": metrics(train), "validation": metrics(val), "curve": curve,
              "seconds": time.monotonic() - started}
    checkpoint = args.output / f"{task}_{mode}.pt"
    torch.save({"format": "str_memory_probe_v1", "config": config,
                "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "scaler": {name: torch.as_tensor(value) for name, value in {
                    "obs_mean": obs_mean, "obs_std": obs_std, "action_mean": act_mean,
                    "action_std": act_std, "target_mean": target_mean, "target_std": target_std}.items()},
                "dataset_path": str(args.dataset.resolve()), "dataset_metadata": metadata,
                "train_episode_ids": result["train_episodes"], "validation_episode_ids": result["validation_episodes"],
                "metrics": result}, checkpoint)
    result["checkpoint"] = str(checkpoint)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", choices=["bc", "prediction"], default=["prediction", "bc"])
    parser.add_argument("--modes", nargs="+", choices=["current", "repeat", "history"], default=["current", "repeat", "history"])
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--max-eval-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--bc-only-success", action="store_true")
    parser.add_argument("--bc-action-key", choices=["actions", "teacher_actions"], default="actions",
                        help="BC targets only; prediction always conditions on actually executed actions")
    args = parser.parse_args()
    if min(args.updates, args.width, args.window, args.max_eval_rows, args.log_every, args.threads) < 1 or args.batch_size < 2:
        parser.error("positive sizes/updates required; batch size >=2 for native BN")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be new or empty; refusing to overwrite prior diagnostics")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    data, metadata = load_dataset(args.dataset)
    history = history_indices(data, args.window)
    with args.dataset.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    summary = {"dataset": str(args.dataset.resolve()), "sha256": digest,
               "metadata": metadata, "rows": len(data["obs"]),
               "episodes": int(len(np.unique(data["episode_id"]))),
               "history_padding": "repeat earliest available state within same contiguous episode segment",
               "split": "whole episode IDs, seeded random split; not a generalization test on fixed starts",
               "warnings": ["Prediction targets assume the declared STR state162 field order.",
                            "History-vs-repeat matches parameter count; current baseline has fewer input weights.",
                            "BC std head is untrained; evaluate deterministic actions only.",
                            "History can help dynamics prediction without being necessary for control.",
                            "Offline loss does not prove a grasp/lift skill; requires closed-loop evaluation."],
               "probes": []}
    for task in args.tasks:
        for mode in args.modes:
            summary["probes"].append(run_probe(args, data, metadata, history, task, mode))
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(args.output / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
