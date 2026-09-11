"""CPU-only geometric motion/dwell audit; never joins differences across resets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OVERNIGHT = ROOT.parents[1]
SOURCES = {
    "native_det_seed0": ROOT / "evaluations/normal/relativefinal/deterministic",
    "native_stoch_seed0": ROOT / "evaluations/normal/relativefinal/stochastic",
    "native_stoch_seed1": ROOT / "confirmation/stochastic_seed1",
    "native_stoch_seed2": ROOT / "confirmation/stochastic_seed2",
    "bc_det_context": OVERNIGHT / "bc_current_full_eval",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stats(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    return {"n": len(values), "min": float(values.min()), "mean": float(values.mean()),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9)), "max": float(values.max())}


def runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def poses(states, fields, goal, offsets):
    relative = states[:, fields["keypoints_rel_goal"]].astype(np.float64).reshape(-1, 4, 3)
    assert np.abs(relative).max() < 9.999
    position = goal[:3] + relative.mean(axis=1)
    quat = states[:, fields["object_rot"]].astype(np.float64)  # xyzw, not wxyz.
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    goal_q = goal[[4, 5, 6, 3]]
    goal_q /= np.linalg.norm(goal_q)
    def rotate(q):
        xyz, w = q[..., None, :3], q[..., None, 3:4]
        return offsets + 2 * np.cross(xyz, np.cross(xyz, offsets) + w * offsets)
    keypoint_error = np.linalg.norm(position[:, None] - goal[:3] + rotate(quat) - rotate(goal_q), axis=-1).max(axis=1)
    return position, quat, keypoint_error


def angular_speed(previous, following, dt):
    # Shortest SO(3) rotation, quaternion sign-invariant; atan2 avoids acos loss near zero.
    imaginary = previous[:, 3:4] * following[:, :3] - following[:, 3:4] * previous[:, :3] - np.cross(previous[:, :3], following[:, :3])
    dot = np.abs((previous * following).sum(axis=-1))
    return 2 * np.arctan2(np.linalg.norm(imaginary, axis=-1), dot) / dt


def aggregate(rows):
    if not rows:
        return {"episodes": 0}
    numeric = sorted({key for row in rows for key, value in row.items() if isinstance(value, (float, int)) and not isinstance(value, bool) and key != "episode_id"})
    return {"episodes": len(rows), "with_consecutive10_near_samples": sum(row["longest_near_run_samples"] >= 10 for row in rows),
            "with_no_near_samples": sum(row["near_samples"] == 0 for row in rows),
            "with_drop_below_initial_after_lift": sum(row["below_initial_after_lift"] for row in rows),
            "metrics": {key: stats([row[key] for row in rows if row[key] is not None]) for key in numeric}}


def main():
    validation_path = ROOT / "success_validation.json"
    criterion = json.loads(validation_path.read_text())["criterion"]
    offsets = np.asarray([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]) * np.asarray(criterion["reward_fixed_size_m"]) * criterion["keypoint_scale"] / 2
    assert runs([False, True, True, False, True]) == [(1, 3), (4, 5)]
    identity = np.asarray([[0., 0., 0., 1.]])
    assert angular_speed(identity, -identity, .1)[0] == 0
    quarter_turn = np.asarray([[0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    assert np.isclose(angular_speed(identity, quarter_turn, .1)[0], np.pi / .2)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu", "criterion": criterion,
        "derivative_method": "Per-transition difference obs_t to pre-reset next_obs_t; position displacement norm/dt and shortest quaternion rotation/dt, float64. Not instantaneous PhysX velocity.",
        "near_dwell_method": "A run of k consecutive near samples spans (k-1)*dt between samples. k*dt is only sampled-occupancy proxy, not verified continuous dwell.",
        "last_near_window": "Last5 near-goal samples, not necessarily consecutive; each speed is for the interval ending at that sample.",
        "source_sha256": {str(validation_path): sha(validation_path)}, "conditions": {},
        "limitations": ["Goal success terminates immediately; there is zero recorded post-success stability horizon.",
            "No contact force/force-closure/slip verification. Palm proximity and height cannot certify stable grasp or deployment readiness.",
            "Motion metrics summarize successful episodes separately; failed episodes are not dropped from success denominators.",
            "Metrics requiring near samples are null for a never-near episode; aggregate n counts available values, not fabricated zeros.",
            "One training seed/fixed object/start/goal; rollout seed1/2 do not establish independent training reproducibility.",
            "BC is contextual only: its arm filter.1 differs from native1.0, so completion speed/motion are not an algorithm-only comparison.",
            "Finite differences average one policy interval (16.667ms), cannot resolve within-interval impulses/oscillation; shortest quaternion differences alias rotations above pi per interval."]}
    traces = {}
    for name, folder in SOURCES.items():
        metadata = json.loads((folder / "metadata.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        checkpoint = Path(metadata["arguments"]["checkpoint"])
        assert summary["model_unchanged"] and metadata["model_digest_before"] == summary["model_digest_after"]
        control = metadata["resolved_config"]["env"]["task_cfg_overrides"]["action"]
        if name.startswith("native_"):
            assert checkpoint == ROOT / "training/normal/models/final/step4883"
            assert metadata["model_digest_before"] == "f35ced3258d0d68dff361c477aeaaec8aec30a98053769644ca5408ed4b9bb9b"
            assert control == {"arm_moving_average": 1.0, "hand_moving_average": .1}
        actor_file = checkpoint / "actor.pt" if checkpoint.is_dir() else checkpoint
        report["source_sha256"][str(actor_file)] = sha(actor_file)
        rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
        data = np.load(folder / "transitions.npz", allow_pickle=False)
        assert all(row["complete"] for row in rows)
        assert len(rows) == summary["episodes_completed"] == summary["episodes_requested"]
        assert set(np.unique(data["episode_id"])) == {row["episode_id"] for row in rows}
        assert json.loads(str(data["metadata_json"])) == metadata
        fields = {key: slice(*value) for key, value in metadata["state_field_slices"].items()}
        dt = metadata["dt_seconds"]
        goal = np.asarray(metadata["fixed_goal_pose_wxyz"], dtype=float)
        obs, nxt = data["obs"], data["next_obs"]
        assert metadata["next_obs_is_pre_reset"] and np.isfinite(obs).all() and np.isfinite(nxt).all()
        position, quat, error = poses(nxt, fields, goal, offsets)
        prev_position, prev_quat, _ = poses(obs, fields, goal, offsets)
        linear = np.linalg.norm(position - prev_position, axis=1) / dt
        angular = angular_speed(prev_quat, quat, dt)
        assert np.allclose(position, data["object_pos"], atol=1e-6, rtol=0)
        height = position[:, 2] - metadata["initial_object_z_m"]
        near = error <= metadata["thresholds"]["success_keypoint_m"]
        assert metadata["thresholds"]["success_keypoint_m"] == criterion["threshold_m"]
        records, continuity_error = [], 0.
        for row in sorted(rows, key=lambda row: row["episode_id"]):
            ix = np.flatnonzero(data["episode_id"] == row["episode_id"])
            ix = ix[np.argsort(data["step_in_episode"][ix])]
            assert np.array_equal(data["step_in_episode"][ix], np.arange(row["length"]))
            done = data["terminated"][ix] | data["truncated"][ix]
            assert done[-1] and not done[:-1].any()
            continuity_error = max(continuity_error, float(np.abs(nxt[ix[:-1]] - obs[ix[1:]]).max(initial=0)))
            assert continuity_error == 0, "Cannot use a trajectory with cross-reset/gapped observation history"
            hit = near[ix]
            hit_indices = np.flatnonzero(hit)
            spans = runs(hit)
            cumulative = int(hit.sum())
            assert (cumulative >= 10) == row["success"] and (data["success"][ix] == row["success"]).all()
            if row["success"]:
                assert cumulative == 10 and hit_indices[-1] == len(ix) - 1
                assert row["terminated"] and not row["truncated"]
            assert np.isclose(row["keypoint_error_m"], error[ix[-1]], atol=1e-6)
            assert abs(row["reward_components"]["bonus_rew"] - 100 * cumulative) < 1e-5
            assert abs(float(data["rewards"][ix].sum(dtype=np.float64)) - row["return"]) < 1e-6
            above = height[ix] > .10
            lifted = np.maximum.accumulate(above)
            assert np.array_equal(lifted, data["task_lift_flag"][ix])
            below_after = bool((lifted & (height[ix] < 0)).any())
            assert below_after == row["dropped_below_reset_after_task_lift"]
            tail = ix[hit_indices[-5:]]
            longest = max((end - start for start, end in spans), default=0)
            lift_runs = runs(above)
            lift_longest = max((end - start for start, end in lift_runs), default=0)
            item = {"episode_id": row["episode_id"], "success": row["success"], "terminated": row["terminated"], "truncated": row["truncated"],
                "completion_or_timeout_steps": len(ix), "completion_or_timeout_s": len(ix) * dt,
                "near_samples": cumulative, "near_step_numbers": (hit_indices + 1).tolist(), "near_run_count": len(spans),
                "longest_near_run_samples": longest, "longest_near_sample_span_s": max(0, longest - 1) * dt,
                "terminal_near_run_samples": spans[-1][1] - spans[-1][0] if hit[-1] else 0,
                "first_to_last_near_sample_span_s": (hit_indices[-1] - hit_indices[0]) * dt if len(hit_indices) else None,
                "final_keypoint_error_m": float(error[ix[-1]]), "final_linear_speed_m_s": float(linear[ix[-1]]),
                "final_angular_speed_rad_s": float(angular[ix[-1]]), "final_angular_speed_deg_s": float(np.rad2deg(angular[ix[-1]])),
                "last5_near_linear_speed_mean_m_s": float(linear[tail].mean()) if len(tail) else None, "last5_near_linear_speed_max_m_s": float(linear[tail].max()) if len(tail) else None,
                "last5_near_angular_speed_mean_rad_s": float(angular[tail].mean()) if len(tail) else None, "last5_near_angular_speed_max_rad_s": float(angular[tail].max()) if len(tail) else None,
                "minimum_height_above_initial_m": float(min(height[ix].min(), prev_position[ix[0], 2] - metadata["initial_object_z_m"])),
                "maximum_height_above_initial_m": float(height[ix].max()), "final_height_above_initial_m": float(height[ix[-1]]),
                "above10cm_samples": int(above.sum()), "above10cm_sample_occupancy_s": float(above.sum() * dt),
                "longest_above10cm_sample_span_s": max(0, lift_longest - 1) * dt, "below_initial_after_lift": below_after,
                "final_palm_object_distance_m": float(data["palm_object_distance_m"][ix[-1]]),
                "last5_near_palm_object_distance_mean_m": float(data["palm_object_distance_m"][tail].mean()) if len(tail) else None,
                "minimum_palm_object_distance_m": float(data["palm_object_distance_m"][ix].min())}
            records.append(item)
            if name == "native_det_seed0":
                traces[row["episode_id"]] = {"time": (data["step_in_episode"][ix] + 1) * dt, "height": height[ix],
                    "error": error[ix], "near": hit, "linear": linear[ix], "angular": angular[ix], "dt": dt}
        successful = [item for item in records if item["success"]]
        assert len(successful) == round(summary["success"]["mean"] * len(rows))
        result = {"folder": str(folder), "checkpoint": metadata["arguments"]["checkpoint"],
            "control": metadata["resolved_config"]["env"]["task_cfg_overrides"]["action"], "dt_seconds": dt,
            "episodes_completed": len(rows), "episodes_requested": summary["episodes_requested"],
            "successes": len(successful), "success_summary": aggregate(successful),
            "failure_summary": aggregate([item for item in records if not item["success"]]), "episodes": records,
            "within_episode_obs_continuity_max_abs_error": continuity_error, "single_terminal_end_and_pre_reset_final_obs_verified": True,
            "no_post_success_frames": True, "metadata_model_unchanged": summary["model_unchanged"]}
        result["model_digest_before_and_after"] = metadata["model_digest_before"]
        report["conditions"][name] = result
        report["source_sha256"].update({str(folder / filename): sha(folder / filename) for filename in ["metadata.json", "summary.json", "episodes.jsonl", "transitions.npz"]})
    deterministic = report["conditions"]["native_det_seed0"]["episodes"]
    median_steps = np.median([row["completion_or_timeout_steps"] for row in deterministic])
    median_run = np.median([row["longest_near_run_samples"] for row in deterministic])
    representative = min(deterministic, key=lambda row: (abs(row["completion_or_timeout_steps"] - median_steps), abs(row["longest_near_run_samples"] - median_run), row["episode_id"]))
    report["representative"] = {"episode_id": representative["episode_id"], "selection": "Closest to median completion steps, then median longest-near-run, then lowest episode ID; no best-case selection.", "record": representative}
    plot(traces[representative["episode_id"]], representative["episode_id"])
    (ROOT / "success_motion_quality.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    markdown(report)
    print(json.dumps({name: {"successes": value["successes"], "episodes": value["episodes_completed"],
        "with10_consecutive": value["success_summary"]["with_consecutive10_near_samples"],
        "terminal_speed_m_s": value["success_summary"]["metrics"]["final_linear_speed_m_s"],
        "terminal_angular_deg_s": value["success_summary"]["metrics"]["final_angular_speed_deg_s"]}
        for name, value in report["conditions"].items()}))


def plot(trace, episode_id):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6), constrained_layout=True, sharex=True)
    t, near = trace["time"], trace["near"]
    ax = axes[0, 0]
    ax.plot(t, trace["height"] * 100, label="Object height")
    ax.scatter(t[near], trace["height"][near] * 100, color="C2", s=20, label="Sampled near-goal frames", zorder=3)
    ax.axhline(10, color=".5", ls="--", label="Lift threshold")
    ax.axhline(15, color=".3", ls=":", label="Goal center height")
    ax.set_ylabel("Height above initial (cm)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(t, trace["error"] * 100, color="C1")
    ax.scatter(t[near], trace["error"][near] * 100, color="C2", s=20, zorder=3)
    ax.axhline(3, color=".4", ls="--", label="Goal threshold (3 cm)")
    ax.set_ylabel("Maximum reward-keypoint error (cm)")
    ax.legend(fontsize=8)
    axes[1, 0].plot(t, trace["linear"], color="C0")
    axes[1, 0].set_ylabel("Finite-difference linear speed (m/s)")
    axes[1, 1].plot(t, np.rad2deg(trace["angular"]), color="C1")
    axes[1, 1].set_ylabel("Finite-difference angular speed (deg/s)")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.axvline(t[-1], color=".4", ls=":", lw=1)
        ax.set_xlim(0, t[-1] * 1.025)
        ax.set_xlabel("Episode time (s)")
    fig.suptitle(f"Native FlashSAC task success · representative deterministic episode {episode_id}", fontsize=12)
    fig.supxlabel(f"Frame dt = {trace['dt'] * 1000:.3f} ms; recording stops at success. Dots are sampled near frames, not post-success stability.", fontsize=8.5)
    fig.savefig(ROOT / "success_motion_trace.png", dpi=175)
    plt.close(fig)


def markdown(report):
    lines = ["# 成功动作质量：达标时是否已经停稳？", "",
        "这里独立核对当前native normal final checkpoint的已有记录。**任务成功真实成立，但判据是累计10帧near，不要求连续停稳；成功立即终止，不能测量成功后的保持、接触力或稳定抓握。**", "",
        "## 任务成功与连续达标分开", "",
        "| 评测 | 成功/全部回合 | 成功中曾连续10帧 | 完成时间均值/中位数(s) | 最长连续near帧数：中位数/范围 |", "| --- | ---: | ---: | ---: | --- |"]
    for name, value in report["conditions"].items():
        s = value["success_summary"]
        m = s["metrics"]
        t, near = m["completion_or_timeout_s"], m["longest_near_run_samples"]
        lines.append(f"| {name} | {value['successes']}/{value['episodes_completed']} | {s['with_consecutive10_near_samples']}/{value['successes']} | {t['mean']:.3f}/{t['median']:.3f} | {near['median']:.1f}/[{near['min']:.0f},{near['max']:.0f}] |")
    lines += ["", "BC仅作背景：其arm filter=.1，native=1；hand均.1。不能把完成速度/运动大小差别单独归因于算法。其他四行是同一训练checkpoint的不同采样/rollout seed，不是四个独立训练seed。所有请求回合均完成，失败仍保留在成功率分母。", "",
        "## 成功终止帧仍有多少运动", "",
        "下表只汇总成功回合。速度是当前policy区间(obs_t→pre-reset next_obs_t)的有限差分平均，不是瞬时PhysX速度；dt约16.667ms。末5个near帧不一定连续。", "",
        "| 评测 | 终止线速度：均值/中位数/p90(m/s) | 终止角速度：均值/中位数/p90(deg/s) | 末5 near线速度均值(m/s) | 末5 near角速度均值(deg/s) |", "| --- | --- | --- | ---: | ---: |"]
    for name, value in report["conditions"].items():
        m = value["success_summary"]["metrics"]
        v, w = m["final_linear_speed_m_s"], m["final_angular_speed_deg_s"]
        lines.append(f"| {name} | {v['mean']:.3f}/{v['median']:.3f}/{v['p90']:.3f} | {w['mean']:.1f}/{w['median']:.1f}/{w['p90']:.1f} | {m['last5_near_linear_speed_mean_m_s']['mean']:.3f} | {np.rad2deg(m['last5_near_angular_speed_mean_rad_s']['mean']):.1f} |")
    lines += ["", "这些数值描述达标时的运动，不是预先定义了合格/不合格速度阈值，也不能外推下一秒会继续稳住或掉落。", "",
        "## 抬升高度、可见抬升时段和近掌部", "",
        "| 评测 | 回合最高高度均值(cm) | 回合最低高度均值(cm) | 终止高度均值(cm) | >10cm最长连续采样跨度中位数(s) | 终止物体–palm距离均值(cm) | 成功中曾低于初始高度(抬升后) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, value in report["conditions"].items():
        s, m = value["success_summary"], value["success_summary"]["metrics"]
        lines.append(f"| {name} | {m['maximum_height_above_initial_m']['mean']*100:.2f} | {m['minimum_height_above_initial_m']['mean']*100:.2f} | {m['final_height_above_initial_m']['mean']*100:.2f} | {m['longest_above10cm_sample_span_s']['median']:.3f} | {m['final_palm_object_distance_m']['mean']*100:.2f} | {s['with_drop_below_initial_after_lift']}/{value['successes']} |")
    rep = report["representative"]
    lines += ["", "高度相对reset初始物体中心，包含起始落到桌面的微小下降。连续k个采样点的首末时间跨度为(k−1)×dt；不能把采样点当作已经验证的连续物理保持。Palm距离不是接触/力闭合检验。", "",
        f"代表图选episode{rep['episode_id']}：完成步数最接近det中位数，再按最长near段接近中位数、最后最低ID，未挑最好回合。图只到成功终止帧，没有后续数据。", "",
        f"![成功动作实际轨迹]({ROOT / 'success_motion_trace.png'})", "",
        "## 核验与边界", "",
        "逐回合确认：step连续；只有末步done；所有内部next_obs[t]与obs[t+1]完全相同；差分仅用同一条transition的前后观测，从不跨reset。使用success_validation中的固定reward尺寸与1.5缩放重建角点，near<=.03m，累计第10帧恰好成功终止；与success标签/每near100分goal bonus一致。记录的物体位置、终点误差、sticky lift与独立几何匹配。", "",
        *[f"- {note}" for note in report["limitations"]], "",
        "JSON保留全部回合的near帧序号、连续段、末5帧速度、终点速度、高度和距离，以及原始文件与actor文件SHA256。四组native记录均指向normal/final/step4883、相同模型digest且评测前后不变；控制均为arm=1、hand=.1。", "",
        f"CPU重跑：`/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python {ROOT / 'audit_success_motion.py'}`。本次无GPU、无环境/核心代码/main report修改。"]
    (ROOT / "success_motion_quality.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
