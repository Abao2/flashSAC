"""Read-only CPU audit of recorded seed1 60M mean-policy vs sampled rollouts."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HELPER = ROOT.parent / "budget50m/native_entry_check/audit_success_motion.py"
spec = importlib.util.spec_from_file_location("motion_helpers", HELPER)
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)


def aggregate(rows):
    if not rows:
        return {"episodes": 0}
    keys = sorted({key for row in rows for key, value in row.items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)
                   and key != "episode_id"})
    return {"episodes": len(rows), "near_count_histogram": {
            str(n): sum(row["near_count"] == n for row in rows) for n in range(11)},
        "ten_consecutive_near": sum(row["longest_near_run_samples"] >= 10 for row in rows),
        "ever_lift_above10cm": sum(row["ever_lift_above10cm"] for row in rows),
        "below_initial_after_lift": sum(row["below_initial_after_lift"] for row in rows),
        "metrics": {key: motion.stats([row[key] for row in rows if row[key] is not None])
                    for key in keys}}


def main():
    criterion_file = ROOT.parent / "budget50m/native_entry_check/success_validation.json"
    criterion = json.loads(criterion_file.read_text())["criterion"]
    offsets = np.array([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]) * np.array(criterion["reward_fixed_size_m"]) * criterion["keypoint_scale"] / 2
    result = {"criterion": criterion, "conditions": {}, "source_sha256": {
        str(HELPER): motion.sha(HELPER), str(criterion_file): motion.sha(criterion_file)}}
    checkpoints, digests, initial_states, configs, traces = [], [], {}, {}, {}
    for sampling in ("deterministic", "stochastic"):
        folder = ROOT / "evaluations/seed1/step58596" / sampling
        metadata = json.loads((folder / "metadata.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        episodes = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
        data = np.load(folder / "transitions.npz", allow_pickle=False)
        assert json.loads(str(data["metadata_json"])) == metadata
        assert metadata["next_obs_is_pre_reset"]
        assert summary["model_unchanged"] and metadata["model_digest_before"] == summary["model_digest_after"]
        assert len(episodes) == summary["episodes_completed"] == summary["episodes_requested"]
        assert set(np.unique(data["episode_id"])) == {row["episode_id"] for row in episodes}
        checkpoint = Path(metadata["arguments"]["checkpoint"])
        checkpoints.append(str(checkpoint))
        digests.append(metadata["model_digest_before"])
        configs[sampling] = metadata["resolved_config"]
        actor = checkpoint / "actor.pt"
        result["source_sha256"][str(actor)] = motion.sha(actor)
        result["source_sha256"].update({str(folder / filename): motion.sha(folder / filename)
            for filename in ("metadata.json", "summary.json", "episodes.jsonl", "transitions.npz")})
        fields = {key: slice(*value) for key, value in metadata["state_field_slices"].items()}
        goal = np.asarray(metadata["fixed_goal_pose_wxyz"], dtype=float)
        dt = metadata["dt_seconds"]
        assert np.isclose(dt, 1/60) and metadata["thresholds"]["success_keypoint_m"] == criterion["threshold_m"]
        obs, nxt = data["obs"], data["next_obs"]
        assert np.isfinite(obs).all() and np.isfinite(nxt).all()
        pos, quat, error = motion.poses(nxt, fields, goal, offsets)
        before_pos, before_quat, _ = motion.poses(obs, fields, goal, offsets)
        assert np.allclose(pos, data["object_pos"], atol=1e-6, rtol=0)
        linear = np.linalg.norm(pos - before_pos, axis=1) / dt
        angular = np.rad2deg(motion.angular_speed(before_quat, quat, dt))
        rotation_error = np.rad2deg(motion.angular_speed(np.broadcast_to(goal[[4, 5, 6, 3]], quat.shape), quat, 1.))
        position_error = np.linalg.norm(pos - goal[:3], axis=1)
        height = pos[:, 2] - metadata["initial_object_z_m"]
        near = error <= criterion["threshold_m"]
        rows, initial, continuity = [], {}, 0.
        for episode in sorted(episodes, key=lambda row: row["episode_id"]):
            ix = np.flatnonzero(data["episode_id"] == episode["episode_id"])
            ix = ix[np.argsort(data["step_in_episode"][ix])]
            assert np.array_equal(data["step_in_episode"][ix], np.arange(episode["length"]))
            done = data["terminated"][ix] | data["truncated"][ix]
            assert done[-1] and not done[:-1].any()
            continuity = max(continuity, float(np.abs(nxt[ix[:-1]] - obs[ix[1:]]).max(initial=0)))
            assert continuity == 0
            hit = near[ix]
            near_ix = np.flatnonzero(hit)
            assert len(near_ix) <= 10
            assert (len(near_ix) == 10) == episode["success"]
            assert (data["success"][ix] == episode["success"]).all()
            if episode["success"]:
                assert near_ix[-1] == len(ix)-1 and episode["terminated"] and not episode["truncated"]
            else:
                assert episode["truncated"] and not episode["terminated"] and len(ix) == 600
            assert abs(episode["reward_components"]["bonus_rew"] - 100*len(near_ix)) < 1e-5
            assert np.isclose(error[ix[-1]], episode["keypoint_error_m"], atol=1e-6, rtol=0)
            assert abs(float(data["rewards"][ix].sum(dtype=np.float64)) - episode["return"]) < 1e-6
            lifted = np.maximum.accumulate(height[ix] > .10)
            assert np.array_equal(lifted, data["task_lift_flag"][ix])
            dropped = bool((lifted & (height[ix] < 0)).any())
            assert dropped == episode["dropped_below_reset_after_task_lift"]
            best = int(np.argmin(error[ix]))
            near_tail = ix[near_ix[-5:]]
            tail = ix[-min(60, len(ix)):]
            spans = motion.runs(hit)
            longest = max((end-start for start, end in spans), default=0)
            row = {"episode_id": episode["episode_id"], "success": episode["success"],
                "terminated": episode["terminated"], "truncated": episode["truncated"],
                "steps": len(ix), "duration_s": len(ix)*dt, "return": episode["return"],
                "near_count": len(near_ix), "near_steps": (near_ix+1).tolist(),
                "near_run_count": len(spans),
                "first_near_s": (near_ix[0]+1)*dt if len(near_ix) else None,
                "last_near_s": (near_ix[-1]+1)*dt if len(near_ix) else None,
                "longest_near_run_samples": longest, "longest_near_sample_span_s": max(0,longest-1)*dt,
                "best_keypoint_error_m": float(error[ix[best]]), "best_keypoint_margin_over_threshold_m": float(error[ix[best]] - criterion["threshold_m"]),
                "best_keypoint_step": best+1, "best_keypoint_time_s": (best+1)*dt,
                "best_frame_position_error_m": float(position_error[ix[best]]), "best_frame_rotation_error_deg": float(rotation_error[ix[best]]),
                "final_keypoint_error_m": float(error[ix[-1]]), "final_position_error_m": float(position_error[ix[-1]]),
                "final_rotation_error_deg": float(rotation_error[ix[-1]]),
                "final_linear_speed_m_s": float(linear[ix[-1]]), "final_angular_speed_deg_s": float(angular[ix[-1]]),
                "last5_near_linear_speed_m_s": float(linear[near_tail].mean()) if len(near_tail) else None,
                "last5_near_angular_speed_deg_s": float(angular[near_tail].mean()) if len(near_tail) else None,
                "last_up_to1s_keypoint_mean_m": float(error[tail].mean()),
                "last_up_to1s_keypoint_min_m": float(error[tail].min()),
                "last_up_to1s_linear_speed_mean_m_s": float(linear[tail].mean()),
                "last_up_to1s_angular_speed_mean_deg_s": float(angular[tail].mean()),
                "maximum_height_above_initial_m": float(height[ix].max()), "final_height_above_initial_m": float(height[ix[-1]]),
                "ever_lift_above10cm": bool(lifted[-1]), "below_initial_after_lift": dropped,
                "final_palm_object_distance_m": float(data["palm_object_distance_m"][ix[-1]]),
                "last_up_to1s_palm_object_distance_mean_m": float(data["palm_object_distance_m"][tail].mean()),
                "goal_bonus": episode["reward_components"]["bonus_rew"],
                "lift_bonus": episode["reward_components"]["lift_bonus_rew"]}
            rows.append(row)
            if episode["episode_id"] < 32:
                initial[episode["episode_id"]] = obs[ix[0]]
                traces[sampling, episode["episode_id"]] = {
                    "time": (np.arange(len(ix))+1)*dt, "error": error[ix], "near_count": np.cumsum(hit)}
        assert sum(row["success"] for row in rows) == round(summary["success"]["mean"]*len(rows))
        initial_states[sampling] = initial
        result["conditions"][sampling] = {"folder": str(folder), "checkpoint": str(checkpoint), "actor_digest": digests[-1],
            "episodes_requested": summary["episodes_requested"], "episodes_completed": len(rows),
            "successes": sum(row["success"] for row in rows), "summary": aggregate(rows),
            "successful": aggregate([r for r in rows if r["success"]]), "failed": aggregate([r for r in rows if not r["success"]]),
            "episodes": rows, "continuity_error": continuity, "no_reset_leakage_verified": True,
            "model_unchanged": True, "dt_seconds": dt}
    assert len(set(checkpoints)) == len(set(digests)) == 1
    assert configs["deterministic"] == configs["stochastic"]
    assert set(initial_states["deterministic"]) == set(initial_states["stochastic"]) == set(range(32))
    result["initial32_obs_max_abs_difference"] = float(max(np.max(np.abs(initial_states["deterministic"][i]-initial_states["stochastic"][i])) for i in range(32)))
    result["initial32_success_counts"] = {mode: sum(row["success"] for row in value["episodes"] if row["episode_id"] < 32)
        for mode, value in result["conditions"].items()}
    paired = [i for i in range(32) if not result["conditions"]["deterministic"]["episodes"][i]["success"]
              and result["conditions"]["stochastic"]["episodes"][i]["success"]]
    example = min(paired)
    result["illustrative_initial_episode"] = {"episode_id": example,
        "selection": "Lowest initial episode ID with deterministic failure and stochastic success; illustrative, not representative frequency.",
        "initial_obs_max_abs_difference": float(np.abs(initial_states["deterministic"][example]-initial_states["stochastic"][example]).max())}
    plot_example(traces, example)
    result["interpretation_limits"] = [
        "Same checkpoint/config/rollout seed and first32 initial states checked; later episodes reset at different times, so not trajectory-by-trajectory paired trials.",
        "Stochastic SAC policy is the optimized behavior; deterministic tanh(mean) is an evaluation choice and has no equal-success guarantee. A stochastic advantage alone proves neither multimodality nor mean-action averaging failure.",
        "Native stochastic noise has its original temporal repetition. This compares that whole sampling rule with deterministic actions, not independent per-step Gaussian noise.",
        "Near criterion is cumulative10, not consecutive10. Success stops recording immediately; finite-difference motion/proximity are not post-goal stability or verified contact.",
        "Final speed is per-transition backward finite difference at 16.667ms, never across reset; quaternion normalization and sign-invariant shortest rotation used.",
        "Success and failure motion cohorts remain separate. Last-up-to1s windows for short successes are shorter than one second.",
        "One seed1-trained policy and one rollout RNG seed on fixed object/start/goal. Sampling result does not prove generalization or a training mechanism.",
        "Near-dependent metrics are null if never near; aggregate n counts observations available, not substituted zeros."]
    result["context"] = {}
    for label, relative in {"seed1_50M_det": "seed1/step48830/deterministic", "seed1_50M_stoch": "seed1/step48830/stochastic",
                            "seed0_60M_det": "seed0/step58596/deterministic", "seed0_60M_stoch": "seed0/step58596/stochastic"}.items():
        path = ROOT / "evaluations" / relative / "summary.json"
        summary = json.loads(path.read_text())
        result["context"][label] = {"successes": round(summary["success"]["mean"]*summary["episodes_completed"]),
            "episodes_completed": summary["episodes_completed"], "path": str(path)}
        result["source_sha256"][str(path)] = motion.sha(path)
    for path, digest in result["source_sha256"].items():
        assert motion.sha(path) == digest, f"Source changed while reading: {path}"
    (ROOT / "seed1_sampling_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    write_markdown(result)
    print(json.dumps({k: {"successes": v["successes"], "episodes": v["episodes_completed"],
        "near_hist_failed": v["failed"]["near_count_histogram"], "failed_best_error": v["failed"]["metrics"]["best_keypoint_error_m"],
        "failed_final_error": v["failed"]["metrics"]["final_keypoint_error_m"]} for k,v in result["conditions"].items()}))


def plot_example(traces, episode_id):
    fig, axes = motion.plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for mode, label in (("deterministic", "Deterministic failure"), ("stochastic", "Stochastic success")):
        trace = traces[mode, episode_id]
        window = trace["time"] <= 1.5
        axes[0].plot(trace["time"][window], trace["error"][window]*100, label=label)
        axes[1].step(trace["time"][window], trace["near_count"][window], where="post", label=label)
    axes[0].axhline(3, color=".4", ls="--", label="3 cm near threshold")
    axes[1].axhline(10, color=".4", ls="--", label="10 cumulative samples")
    axes[0].set_ylabel("Maximum reward-keypoint error (cm)")
    axes[1].set_ylabel("Cumulative near-goal samples")
    for ax in axes:
        ax.set_xlabel("Episode time (s)")
        ax.set_xlim(0, 1.5)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[1].set_ylim(-.2, 11)
    fig.suptitle(f"Seed1 60M · same initial episode {episode_id} · first 1.5 s only", fontsize=12)
    fig.supxlabel("Illustrative lowest-ID divergence; stochastic recording ends at success, deterministic continues to 10 s timeout.", fontsize=8)
    fig.savefig(ROOT / "seed1_sampling_trace.png", dpi=175)
    motion.plt.close(fig)


def write_markdown(result):
    lines = ["# Seed1 60M：确定性与随机采样的差别", "",
        "使用同一冻结checkpoint、同一任务配置的完整已有回合；不是重新训练。独立几何核验成功，未把抬高/靠近palm代理值当成功。", "",
        "| 采样 | 真实成功/请求回合 | 成功中连续10帧near | 失败near次数分布 |", "| --- | ---: | ---: | --- |"]
    for mode, condition in result["conditions"].items():
        histogram = ", ".join(f"{k}帧:{v}回合" for k,v in condition["failed"]["near_count_histogram"].items() if v)
        lines.append(f"| {mode} | {condition['successes']}/{condition['episodes_completed']} | {condition['successful']['ten_consecutive_near']}/{condition['successes']} | {histogram} |")
    lines += ["", "## 失败究竟离目标多远", "",
        "以下只看失败回合；误差为固定reward尺寸的4个对应角点最大距离，阈值3cm，而不是只看物体中心。所有失败均600步时间截断；没有被错误计成成功。", "",
        "| 采样 | 最佳误差 min/median/max(cm) | 最佳时刻 median(s) | 终点误差 min/median/max(cm) | 最后1秒误差均值(cm) |", "| --- | --- | ---: | --- | ---: |"]
    for mode, condition in result["conditions"].items():
        metrics = condition["failed"]["metrics"]
        best, end = metrics["best_keypoint_error_m"], metrics["final_keypoint_error_m"]
        lines.append(f"| {mode} | {best['min']*100:.3f}/{best['median']*100:.3f}/{best['max']*100:.3f} | {metrics['best_keypoint_time_s']['median']:.3f} | {end['min']*100:.3f}/{end['median']*100:.3f}/{end['max']*100:.3f} | {metrics['last_up_to1s_keypoint_mean_m']['mean']*100:.3f} |")
    det = result["conditions"]["deterministic"]["failed"]["metrics"]
    returned_successes = sum(row["success"] and row["near_run_count"] > 1 for row in result["conditions"]["stochastic"]["episodes"])
    lines += ["", "**不是‘还差一点才碰到阈值’：63个确定性失败全部进入了3cm范围，且最佳误差约1.3–1.4cm；问题是没有累计够10帧。** 全部在0.450s首次进入，0.467s达到最佳；最后near帧中位时刻0.567s，最晚0.900s，之后没有再达到阈值，最终10s超时。多数是连续8帧后离开。", "",
        f"这些失败仍获得平均{det['goal_bonus']['mean']:.1f}分goal bonus（每near帧100分）与300分lift bonus，所以高reward不等于任务完成。末1秒平均线速度{det['last_up_to1s_linear_speed_mean_m_s']['mean']:.4f}m/s、角速度{det['last_up_to1s_angular_speed_mean_deg_s']['mean']:.2f}deg/s，停留在偏离目标的位置，而不是仍在高速寻找目标。这个描述只来自运动记录，不证明其内部优化原因。", "",
        f"随机成功88回合中，{returned_successes}个在离开near区后再次进入，才累计到10帧；其余连续10帧完成。首次near中位时刻仍0.450s，成功中位时刻0.700s。随机采样改变后续闭环轨迹，能补足缺的near帧；不能仅凭结果断言具体是哪个关节噪声、均值策略多峰或Q函数原因。", "",
        f"![同一起始回合的前1.5秒]({ROOT / 'seed1_sampling_trace.png'})", "",
        "图按最低初始episode ID选一个确定性失败/随机成功的实例，不是总体成功率或典型性证明。随机曲线在真实成功处结束，不画虚构的后续保持。"]
    lines += ["", "## 成功与失败动作分别统计", "",
        "| 采样/结果 | 回合数 | 持续时间 median(s) | 终点中心误差均值(cm) | 终点旋转误差均值(deg) | 终点FD线速度均值(m/s) | 终点FD角速度均值(deg/s) | 终点高度均值(cm) | 终点物体–palm距离均值(cm) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode, condition in result["conditions"].items():
        for cohort in ("successful", "failed"):
            c = condition[cohort]
            m = c["metrics"]
            lines.append(f"| {mode}/{cohort} | {c['episodes']} | {m['duration_s']['median']:.3f} | {m['final_position_error_m']['mean']*100:.3f} | {m['final_rotation_error_deg']['mean']:.2f} | {m['final_linear_speed_m_s']['mean']:.3f} | {m['final_angular_speed_deg_s']['mean']:.1f} | {m['final_height_above_initial_m']['mean']*100:.2f} | {m['final_palm_object_distance_m']['mean']*100:.2f} |")
    lines += ["", "抬升/近掌部并非力闭合或稳定抓握的证据。JSON包含失败与成功的完整near帧序号、最佳误差/时刻、末秒窗口、连续near、运动和reward分量。", "",
        "## 可得出与不能得出的结论", "",
        "随机策略与tanh(mean)策略在这个checkpoint上的任务表现确实不同。SAC优化的是带采样的策略；将其部署成确定性均值动作不是等价操作，也没有成功率不降低的保证。但本结果本身不证明策略多峰、‘平均了两种动作’，或特定探索机制。", "",
        f"最初32个回合step0观测两组最大绝对差为{result['initial32_obs_max_abs_difference']:.9g}；这批初始回合确定性成功{result['initial32_success_counts']['deterministic']}/32、随机成功{result['initial32_success_counts']['stochastic']}/32。随后reset时刻不同，不把64与128个回合强行配成同一批逐轨迹对照。两组配置完全相同，actor digest相同且评测前后均未改变；源actor/data SHA256已保存。", "",
        *[f"- {note}" for note in result["interpretation_limits"]], "",
        "## 背景，不混用checkpoint", "",
        *[f"- {key}: {value['successes']}/{value['episodes_completed']}（只引用原summary，未对这些回合重做运动核验）。" for key,value in result["context"].items()], "",
        f"CPU复现：`/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python {Path(__file__).resolve()}`。无GPU、无核心/配置/主报告修改。"]
    (ROOT / "seed1_sampling_audit.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
