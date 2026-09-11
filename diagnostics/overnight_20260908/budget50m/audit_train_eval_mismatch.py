"""CPU-only train/frozen config, logger and recorded-trajectory audit; no Isaac import."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
STR = REPO.parent / "simtoolreal"
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from eval_str_state_teacher import pose_errors
from flash_rl.common.logger import AverageMeterDict, TensorboardTrainerLogger
from flash_rl.envs.isaaclab import IsaacLabVectorEnv
from flash_rl.agents.flashSAC.network import FlashSACActor
from summarize_str_control_ab import sha256


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [item for key in sorted(a.keys() | b.keys())
                for item in differences(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}.{key}")]
    return [] if a == b else [{"field": path.lstrip("."), "train": a, "eval": b}]


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max()),
            "mean": float(values.mean())}


def logger_check():
    path = STR / "isaacsimenvs/tasks/simtoolreal/utils/logging_utils.py"
    spec = importlib.util.spec_from_file_location("task_logging_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = SimpleNamespace(cfg=SimpleNamespace(termination=SimpleNamespace(max_consecutive_successes=1)),
        _successes=torch.tensor([1, 0, 1, 0]), _termination_reasons={"max_successes": torch.tensor([1, 0, 1, 0], dtype=torch.bool)},
        _reward_terms={"bonus_rew": torch.tensor([100., 0., 100., 0.])},
        _prev_episode_successes=torch.full((4,), 9), _current_success_tolerance=.02, extras={})
    module.log_step_metrics(raw)
    raw._successes.zero_()  # Mimic in-place auto-reset AFTER extras publication.
    assert raw.extras["episode_final"]["all_goals_hit"].tolist() == [1., 0., 1., 0.]
    wrapper = IsaacLabVectorEnv.__new__(IsaacLabVectorEnv)
    wrapper.num_envs, wrapper.device = 4, "cpu"
    wrapper._episode_returns, wrapper._episode_lengths = torch.zeros(4), torch.zeros(4)
    wrapper._episode_cumulative = {}
    first = wrapper._collect_episode_info(raw.extras, torch.zeros(4), torch.tensor([1, 0, 0, 1], dtype=torch.bool))
    raw.extras["episode_final"]["all_goals_hit"] = torch.tensor([0., 1., 0., 0.])
    second = wrapper._collect_episode_info(raw.extras, torch.zeros(4), torch.tensor([0, 1, 0, 0], dtype=torch.bool))
    logger = SimpleNamespace(average_meter_dict=AverageMeterDict(), media_dict={})
    for value in [first, second]:
        TensorboardTrainerLogger.update_metric(logger, **value)
    result = logger.average_meter_dict.averages()["episode/final/all_goals_hit"]
    assert abs(result - 2 / 3) < 1e-7 and result != .75
    return {"pre_reset_bool_to_float_snapshot_survives_success_reset": True,
            "ignores_stale_prev_episode_successes_value_9": True,
            "done_weighted_fixture": {"step1": [.5, 2], "step2": [1., 1], "result": result}}


def main():
    torch.set_num_threads(2)
    train = json.loads((ROOT / "configs/arm1_hand01_seed0.json").read_text())
    path = ROOT / "evaluations/arm1_hand01_seed0/step48830/deterministic"
    metadata = json.loads((path / "metadata.json").read_text())
    dataset = np.load(path / "transitions.npz", allow_pickle=False)
    task = train["env"]["task_cfg_overrides"]
    assert task == metadata["resolved_config"]["env"]["task_cfg_overrides"]
    base = yaml.safe_load((STR / "isaacsimenvs/cfg/task/SimToolReal.yaml").read_text())
    tb_files = list((ROOT / "runs/budget50m/arm1_hand01").rglob("events.out.tfevents*"))
    assert len(tb_files) == 1
    tb = EventAccumulator(str(tb_files[0]), size_guidance={"scalars": 0, "tensors": 0}).Reload()
    tb_cfg = yaml.safe_load(tb.Tensors("config/text_summary")[0].tensor_proto.string_val[0])
    assert not differences(train, tb_cfg)
    tolerance = [s.value for s in tb.Scalars("task/current_success_tolerance")]
    fields = {name: slice(*bounds) for name, bounds in metadata["state_field_slices"].items()}
    goal = np.asarray(metadata["fixed_goal_pose_wxyz"])
    offsets = np.asarray([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]) * np.asarray(base["reward"]["fixed_size"]) * base["reward"]["keypoint_scale"] / 2
    dims = np.unique(dataset["obs"][:, fields["object_scales"]] * metadata["object_base_size_m"], axis=0)
    assert len(dims) == 1
    episodes, representative = [], None
    completed = {row["episode_id"]: row for row in map(json.loads, (path / "episodes.jsonl").read_text().splitlines())}
    for episode_id in np.unique(dataset["episode_id"]):
        indices = np.flatnonzero(dataset["episode_id"] == episode_id)
        indices = indices[np.argsort(dataset["step_in_episode"][indices])]
        assert np.array_equal(dataset["step_in_episode"][indices], np.arange(len(indices)))
        state = dataset["next_obs"][indices]
        errors = np.array([pose_errors(row, fields, goal, offsets, base["obs"]["clamp_abs_observations"]) for row in state])
        height = dataset["object_z"][indices] - metadata["initial_object_z_m"]
        lifted = dataset["task_lift_flag"][indices]
        drops = np.flatnonzero(lifted & (height < 0))
        near = errors[:, 2] <= metadata["thresholds"]["success_keypoint_m"]
        row = completed[int(episode_id)]
        assert np.isclose(errors[-1, 2], row["keypoint_error_m"], atol=1e-7)
        assert bool(len(drops)) == row["dropped_below_reset_after_task_lift"]
        item = {"episode_id": int(episode_id), "steps": len(indices), "near_goal_keypoint_frames": int(near.sum()),
                "first_task_lift_step": int(np.flatnonzero(lifted)[0] + 1) if lifted.any() else None,
                "first_below_reset_after_lift_step": int(drops[0] + 1) if len(drops) else None,
                "minimum_center_error_m": float(errors[:, 0].min()), "minimum_keypoint_error_m": float(errors[:, 2].min()),
                "minimum_keypoint_step": int(errors[:, 2].argmin() + 1), "final_keypoint_error_m": float(errors[-1, 2]),
                "maximum_height_m": float(height.max()), "initial_reward_observation": float(dataset["obs"][indices[0], fields["reward"]][0]),
                "success": row["success"], "terminated": row["terminated"], "truncated": row["truncated"]}
        episodes.append(item)
        if int(episode_id) == 0:  # Selection fixed by ID, not cherry-picked after looking at performance.
            representative = (indices, errors, height)
    indices, errors, height = representative
    reset_obs = dataset["obs"][dataset["step_in_episode"] == 0]
    model_path = ROOT / "models/arm1_hand01/seed0/step48830/actor.pt"
    saved = torch.load(model_path, map_location="cpu", weights_only=True)["network_state_dict"]
    actor = FlashSACActor(num_blocks=2, input_dim=162, hidden_dim=128, action_dim=29)
    actor.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in saved.items()}, strict=True)
    actor.eval()
    sensitivity = []
    with torch.no_grad():
        reference = actor.get_mean_and_std(torch.from_numpy(reset_obs), training=False)[0].tanh()
        for field, value in [("reward", 0.), ("reward", 1.), ("progress", float(np.log(31)))]:
            changed = reset_obs.copy()
            changed[:, fields[field]] = value
            delta = (actor.get_mean_and_std(torch.from_numpy(changed), training=False)[0].tanh() - reference).abs()
            sensitivity.append({"field": field, "counterfactual_value": value, "reset_states": len(reset_obs),
                "mean_absolute_action_delta": float(delta.mean()), "max_absolute_action_delta": float(delta.max()),
                "arm_mean_absolute_delta": float(delta[:, :7].mean()), "hand_mean_absolute_delta": float(delta[:, 7:].mean())})
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu", "config_task_overrides_equal": True,
        "saved_train_config_equals_tb_config": True, "full_config_differences": differences(train, metadata["resolved_config"]),
        "logger_fixture": logger_check(), "tb_tolerance": stats(tolerance),
        "tb_final_goal_window": {"transition": tb.Scalars("episode/final/all_goals_hit")[-1].step,
                                 "fraction": tb.Scalars("episode/final/all_goals_hit")[-1].value,
                                 "denominator": None, "note": "TB stores weighted mean, not its episode count; cannot reconstruct exact numerator/denominator uniquely."},
        "runtime_eval_metadata": {key: metadata[key] for key in ["arguments", "training_startup", "dt_seconds", "table_top_m", "initial_object_z_m", "fixed_goal_pose_wxyz", "thresholds"]},
        "asset_actual_dimensions_from_eval_obs_m": dims[0].tolist(), "reward_fixed_dimensions_m": base["reward"]["fixed_size"],
        "table_and_timing_sources": {"registered_yaml_dt": base["sim"]["dt"], "registered_yaml_decimation": base["decimation"],
                                    "reset_when_dropped": base["termination"]["reset_when_dropped"]},
        "episodes": episodes, "all_episode_minimum_keypoint_error_m": stats([x["minimum_keypoint_error_m"] for x in episodes]),
        "all_episode_minimum_center_error_m": stats([x["minimum_center_error_m"] for x in episodes]),
        "drop_count": sum(x["first_below_reset_after_lift_step"] is not None for x in episodes),
        "total_keypoint_near_goal_frames": sum(x["near_goal_keypoint_frames"] for x in episodes),
        "counterfactual_reset_input_sensitivity": sensitivity,
        "source_sha256": {str(p): sha256(p) for p in [model_path, path / "metadata.json", path / "transitions.npz", tb_files[0],
            REPO / "train.py", REPO / "flash_rl/common/logger.py", REPO / "flash_rl/envs/isaaclab.py",
            STR / "isaacsimenvs/cfg/task/SimToolReal.yaml", STR / "isaacsimenvs/tasks/simtoolreal/utils/logging_utils.py",
            STR / "isaacsimenvs/tasks/simtoolreal/utils/reset_utils.py", STR / "isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py"]},
        "limitations": ["Training has no transition dataset/runtime geometry dump; source/config agreement is not direct proof of identical physical rollout states.",
            "TB completed-episode mean is over a moving policy and short logging window, not an independent evaluation of the final checkpoint.",
            "Reset-input counterfactual is CPU action sensitivity only; it does not establish the cause or a successful policy intervention.",
            "Object height and palm proximity do not prove force closure, verified contacts or stable grasp."]}
    (ROOT / "train_eval_mismatch_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_trace(dataset, indices, errors, height, metadata, episodes[0])
    print(json.dumps({key: report[key] for key in ["logger_fixture", "all_episode_minimum_keypoint_error_m", "drop_count", "total_keypoint_near_goal_frames", "counterfactual_reset_input_sensitivity"]}))


def plot_trace(dataset, indices, errors, height, metadata, episode):
    time = (dataset["step_in_episode"][indices] + 1) * metadata["dt_seconds"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(time, height * 100, label="Object center above reset")
    ax.axhline(10, color=".4", ls="--", label="Task lift threshold")
    ax.axhline(0, color=".6", lw=.8)
    ax.set_ylabel("Height (cm)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(time, errors[:, 0] * 100, label="Object-center error")
    ax.plot(time, errors[:, 2] * 100, label="True reward-keypoint error")
    ax.axhline(3, color=".4", ls="--", label="Goal threshold (keypoints)")
    ax.scatter([time[errors[:, 2].argmin()]], [errors[:, 2].min() * 100], color="C1", s=20)
    ax.set_ylabel("Distance to goal (cm)")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(time, dataset["palm_object_distance_m"][indices] * 100, label="Palm–object center")
    ax.plot(time, dataset["tip_box_distance_m"][indices] * 100, label="Nearest tip–box distance")
    ax.set_ylabel("Geometric distance (cm)")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(time, dataset["object_pos"][indices, 0] * 100, label="Object X")
    ax.plot(time, dataset["object_pos"][indices, 1] * 100, label="Object Y")
    ax.plot(time, dataset["palm_pos"][indices, 0] * 100, color="C0", ls=":", label="Palm X")
    ax.plot(time, dataset["palm_pos"][indices, 1] * 100, color="C1", ls=":", label="Palm Y")
    ax.axhline(5, color="C0", ls="--", alpha=.5)
    ax.axhline(0, color="C1", ls="--", alpha=.5)
    ax.set_ylabel("Env-local position (cm)")
    ax.legend(fontsize=8, ncol=2)
    for ax in axes.flat:
        ax.set_xlabel("Time after policy takeover (s)")
        ax.grid(alpha=.2)
        ax.set_xlim(0, time[-1])
    fig.suptitle("Frozen 50M policy, hand filter 0.1 · episode 0 · goal not reached", fontsize=13)
    fig.supxlabel("Recorded post-step states only; goal needs 10 keypoint-near frames (observed: 0). Proximity is not verified grasp.", fontsize=9)
    fig.savefig(ROOT / "frozen50m_episode0_trace.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
