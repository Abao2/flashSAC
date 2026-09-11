"""CPU-only independent geometric verification of the completed frozen sampling grid."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HELPER = ROOT.parents[1] / "budget50m/native_entry_check/audit_success_motion.py"
spec = importlib.util.spec_from_file_location("motion_helpers", HELPER)
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    status = json.loads((ROOT / "queue_status.json").read_text())
    assert status["status"] == "completed", status["status"]
    sources = {"deterministic": ROOT.parent / "evaluations/seed1/step58596/deterministic",
               "native1_seed0": ROOT.parent / "evaluations/seed1/step58596/stochastic"}
    sources.update({job["id"]: Path(job["expected_completion_file"]).parent for job in manifest["jobs"]})
    criterion_file = ROOT.parents[1] / "budget50m/native_entry_check/success_validation.json"
    criterion = json.loads(criterion_file.read_text())["criterion"]
    offsets = np.array([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]) * np.array(criterion["reward_fixed_size_m"]) * criterion["keypoint_scale"] / 2
    actor = Path(manifest["checkpoint"])/"actor.pt"
    report = {"status": "complete_independently_verified", "new_jobs_completed": len(manifest["jobs"]),
              "reused_baseline_conditions": 2, "criterion": criterion, "conditions": {},
              "source_sha256": {str(path): motion.sha(path) for path in (ROOT/"manifest.json", ROOT/"queue_status.json", HELPER, criterion_file, actor, Path(__file__).resolve())}}
    normalized_configs, models, first_states = [], [], {}
    for name, folder in sources.items():
        meta = json.loads((folder/"metadata.json").read_text())
        summary = json.loads((folder/"summary.json").read_text())
        rows = [json.loads(line) for line in (folder/"episodes.jsonl").read_text().splitlines()]
        data = np.load(folder/"transitions.npz", allow_pickle=False)
        assert summary["status"] == "complete" and all(row["complete"] for row in rows)
        assert len(rows) == summary["episodes_requested"] == summary["episodes_completed"]
        assert set(np.unique(data["episode_id"])) == {row["episode_id"] for row in rows}
        assert meta["next_obs_is_pre_reset"] and json.loads(str(data["metadata_json"])) == meta
        assert summary["model_unchanged"] and meta["model_digest_before"] == summary["model_digest_after"]
        assert meta["arguments"]["checkpoint"] == manifest["checkpoint"]
        models.append(meta["model_digest_before"])
        config = json.loads(json.dumps(meta["resolved_config"]))
        config_hash = digest(config)
        for sub in (config, config["agent"], config["env"]):
            assert sub["seed"] == meta["arguments"]["seed"]
            sub["seed"] = "EVAL_SEED"
        config["save_path"] = "EVAL_SAVE_PATH"
        normalized_configs.append(digest(config))
        fields = {k: slice(*v) for k,v in meta["state_field_slices"].items()}
        goal = np.asarray(meta["fixed_goal_pose_wxyz"], dtype=float)
        dt = meta["dt_seconds"]
        assert np.isclose(dt, 1/60) and meta["thresholds"]["success_keypoint_m"] == criterion["threshold_m"]
        obs, nxt = data["obs"], data["next_obs"]
        assert np.isfinite(obs).all() and np.isfinite(nxt).all()
        pos, quat, error = motion.poses(nxt, fields, goal, offsets)
        before_pos, before_quat, _ = motion.poses(obs, fields, goal, offsets)
        linear = np.linalg.norm(pos-before_pos, axis=1)/dt
        angular = np.rad2deg(motion.angular_speed(before_quat, quat, dt))
        assert np.allclose(pos, data["object_pos"], atol=1e-6, rtol=0)
        near = error <= criterion["threshold_m"]
        records, first_states[name] = [], {}
        for row in sorted(rows, key=lambda r: r["episode_id"]):
            ix = np.flatnonzero(data["episode_id"] == row["episode_id"])
            ix = ix[np.argsort(data["step_in_episode"][ix])]
            assert np.array_equal(data["step_in_episode"][ix], np.arange(row["length"]))
            assert np.array_equal(nxt[ix[:-1]], obs[ix[1:]])
            done = data["terminated"][ix] | data["truncated"][ix]
            assert done[-1] and not done[:-1].any()
            hit = near[ix]
            count = int(hit.sum())
            assert count <= 10 and (count == 10) == row["success"]
            assert (data["success"][ix] == row["success"]).all()
            assert row["terminated"] == bool(data["terminated"][ix[-1]])
            assert row["truncated"] == bool(data["truncated"][ix[-1]])
            if row["success"]:
                assert hit[-1] and row["terminated"] and not row["truncated"]
            else:
                assert row["truncated"] and not row["terminated"] and len(ix) == 600
            assert abs(row["reward_components"]["bonus_rew"] - 100*count) < 1e-5
            assert abs(data["rewards"][ix].sum(dtype=np.float64) - row["return"]) < 1e-6
            assert np.isclose(error[ix[-1]], row["keypoint_error_m"], atol=1e-6, rtol=0)
            lifted = np.maximum.accumulate(pos[ix,2]-meta["initial_object_z_m"] > .1)
            assert np.array_equal(lifted, data["task_lift_flag"][ix])
            assert row["reward_components"]["lift_bonus_rew"] == 300*bool(lifted[-1])
            runs = motion.runs(hit)
            records.append({"episode_id": row["episode_id"], "success": row["success"], "steps": len(ix),
                "duration_s": len(ix)*dt, "near_count": count, "near_steps": (np.flatnonzero(hit)+1).tolist(),
                "longest_consecutive_near_samples": max((end-start for start,end in runs), default=0),
                "near_runs": len(runs), "return": row["return"], "goal_bonus": 100*count,
                "lift_bonus": row["reward_components"]["lift_bonus_rew"], "ever_lift": bool(lifted[-1]),
                "best_keypoint_error_m": float(error[ix].min()), "final_keypoint_error_m": float(error[ix[-1]]),
                "final_linear_speed_m_s": float(linear[ix[-1]]), "final_angular_speed_deg_s": float(angular[ix[-1]]),
                "final_palm_object_distance_m": float(data["palm_object_distance_m"][ix[-1]])})
            if row["episode_id"] < 32:
                first_states[name][row["episode_id"]] = obs[ix[0]]
        successful = [r for r in records if r["success"]]
        failed = [r for r in records if not r["success"]]
        assert len(successful) == round(summary["success"]["mean"]*len(rows))
        assert len(rows) == (64 if name == "deterministic" else 128)
        item = {"folder": str(folder), "sampling": meta["arguments"]["sampling"],
                "noise_multiplier": meta["arguments"]["noise_multiplier"], "noise_repeat": meta["arguments"]["noise_repeat"],
                "eval_seed": meta["arguments"]["seed"], "episodes_completed": len(rows), "episodes_requested": summary["episodes_requested"],
                "successes": len(successful), "initial32_successes": sum(r["success"] for r in records if r["episode_id"]<32),
                "successful_with_consecutive10": sum(r["longest_consecutive_near_samples"]>=10 for r in successful),
                "successful_with_reentry": sum(r["near_runs"]>1 for r in successful), "episodes": records,
                "failed_near_count_histogram": {str(n):sum(r["near_count"]==n for r in failed) for n in range(10)},
                "raw_return_all": motion.stats([r["return"] for r in records]),
                "independent_frame_reward_terminal_check": True, "model_unchanged": True,
                "config_sha256": config_hash, "normalized_config_sha256": normalized_configs[-1],
                "model_digest": models[-1], "dt_seconds": dt}
        for label, cohort in (("successful", successful), ("failed", failed)):
            item[label] = {"episodes": len(cohort), "metrics": {key: motion.stats([r[key] for r in cohort])
                for key in ("duration_s", "near_count", "longest_consecutive_near_samples", "best_keypoint_error_m", "final_keypoint_error_m", "final_linear_speed_m_s", "final_angular_speed_deg_s", "final_palm_object_distance_m")}}
        report["conditions"][name] = item
        report["source_sha256"].update({str(folder/filename): motion.sha(folder/filename) for filename in ("metadata.json", "summary.json", "episodes.jsonl", "transitions.npz")})
    assert len(set(models)) == len(set(normalized_configs)) == 1
    for name, states in first_states.items():
        assert set(states) == set(range(32))
        report["conditions"][name]["initial32_obs_max_abs_difference_from_native_seed0"] = float(max(np.abs(states[i]-first_states["native1_seed0"][i]).max() for i in range(32)))
    report["limits"] = manifest["limits"] + [
        "All full128 vs initial32 results are separate denominators; initial32 is a subset, not another independent trial.",
        "Points summarize one fixed trained policy/task. No statistical significance or optimal multiplier claim.",
        "Motion is per-transition backward finite difference; no episode boundary crossing, no post-success holding frames.",
        "Seeds here are rollout RNG seeds, not new training seeds. Frozen inference sampling does not identify the cause of training exploration gains."]
    for path, expected in report["source_sha256"].items():
        assert motion.sha(path) == expected
    write_report(report)
    plot(report)
    (ROOT/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:{"success":v["successes"],"n":v["episodes_completed"],"first32":v["initial32_successes"],"consecutive10":v["successful_with_consecutive10"]} for k,v in report["conditions"].items()}))


def write_report(report):
    order = ("deterministic", "std025", "std05", "std075", "native1_seed0", "std15", "native_per_step", "native_seed1")
    lines = ["# Seed1 60M：冻结推理采样对累计达标的影响", "",
        "6个新增评测全部完成；另引用并重新逐帧核验原确定性与native1.0基线。没有训练、参数更新、控制器变化；变的是推理采样标准差倍率/时间相关性/rollout seed。", "",
        "| 条件 | 成功/全部 | 相同起始32回合成功 | 成功中连续10帧 | 成功中离开后重入 | 全部回合原始reward均值 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in order:
        c=report["conditions"][name]
        lines.append(f"| {name} | {c['successes']}/{c['episodes_completed']} | {c['initial32_successes']}/32 | {c['successful_with_consecutive10']}/{c['successes']} | {c['successful_with_reentry']}/{c['successes']} | {c['raw_return_all']['mean']:.2f} |")
    lines += ["", "std025/std05/std075/std15是原生时间相关采样，pre-tanh标准差分别乘.25/.5/.75/1.5；native1_seed0是已有1.0基线。native_per_step保持1.0但每步独立抽噪声；native_seed1保持原生1.0，仅更换评测随机种子。确定性是tanh(mean)，不是新训练policy。", "",
        "倍率越大的这些已测点成功率越高，但这只是当前范围、同一训练权重/固定任务的冻结采样结果；不能把1.5称为最优，也不能当成训练时加大探索能提高学习效果的因果证明。独立每步82/128与原生88/128不构成统计显著性结论；换一个rollout seed也88/128不等于轨迹相同或训练复现。", "",
        "## 独立核验", "",
        "所有回合逐帧重建reward所用固定尺寸角点最大距离：≤3cm记一次near，累计第10次恰好成功terminated；所有未成功回合600步truncated。goal bonus严格等于100×near帧数；sticky抬升与300分lift bonus匹配；逐帧reward总和与回合return一致。回合内部next_obs[t]==obs[t+1]完全相同，末帧使用pre-reset final_obs，未跨reset差分。", "",
        "8组是同一个actor checkpoint，评测前后模型digest相同；实际actor与数据文件SHA256保留。完整resolved config哈希记录在JSON；只归一化eval seed（顶层/agent/env）与save_path后8组配置哈希相同，其他配置无漂移。", "",
        f"最初32回合step0观测对native seed0基线的最大绝对差：{max(v['initial32_obs_max_abs_difference_from_native_seed0'] for v in report['conditions'].values()):.9g}。后续reset时刻随轨迹变化；全128不是逐条状态相同的配对轨迹。32回合只是完整预算子集，不能重复计入样本量。", "",
        "## 失败与成功不能只靠reward判断", "",
        "| 条件 | 失败near次数:回合数 | 成功时间中位数(s) | 成功终点FD速度均值(m/s) | 成功终点FD角速度均值(deg/s) |", "| --- | --- | ---: | ---: | ---: |"]
    for name in order:
        c=report["conditions"][name]
        h=", ".join(f"{k}:{v}" for k,v in c["failed_near_count_histogram"].items() if v)
        m=c["successful"]["metrics"]
        lines.append(f"| {name} | {h} | {m['duration_s']['median']:.3f} | {m['final_linear_speed_m_s']['mean']:.3f} | {m['final_angular_speed_deg_s']['mean']:.1f} |")
    lines += ["", "失败仍可以取得7–9次near带来的700–900分奖励；成功也只是累计10帧，并不要求连续停稳。速度是16.667ms区间平均差分。数据成功即终止，不能判断后续保持、接触力或稳定抓握，更不能据此说可以部署真机。", "",
        f"![采样倍率与真实成功率]({ROOT/'success_vs_sampling.png'})", "",
        *[f"- {note}" for note in report["limits"]], "",
        f"CPU复核：`/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python {Path(__file__).resolve()}`。无GPU、无核心/manifest/main report修改。"]
    (ROOT/"analysis.md").write_text("\n".join(lines)+"\n")


def plot(report):
    fig, axes = motion.plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    names=("std025","std05","std075","native1_seed0","std15")
    x=[report["conditions"][n]["noise_multiplier"] for n in names]
    full=[report["conditions"][n]["successes"]/128*100 for n in names]
    initial=[report["conditions"][n]["initial32_successes"]/32*100 for n in names]
    axes[0].scatter(x,full,label="Native temporal noise: all128",s=45)
    axes[0].scatter(x,initial,label="Same initial32 subset",marker="s",facecolors="none",edgecolors="C1",s=50)
    axes[0].scatter([0],[100/64],marker="D",color=".3",s=40,label="Deterministic:1/64")
    axes[0].set_xlabel("Pre-tanh sampling std multiplier (0 = deterministic)")
    axes[0].set_ylabel("Goal success (%)")
    axes[0].legend(fontsize=8,loc="upper left")
    axes[0].set_xticks([0,.25,.5,.75,1,1.5])
    controls=("native1_seed0","native_per_step","native_seed1")
    vals=[report["conditions"][n]["successes"] for n in controls]
    axes[1].bar(range(3),np.array(vals)/128*100,color=["C0","C2","C3"],width=.5)
    axes[1].set_xticks(range(3),["Native\neval seed0","Independent\nper-step","Native\neval seed1"])
    axes[1].set_ylabel("Goal success (%)")
    axes[1].set_title("Multiplier1.0: temporal rule / rollout seed",fontsize=11)
    for i,n in enumerate(vals):
        axes[1].text(i,n/128*100+2,f"{n}/128",ha="center",fontsize=9)
    for ax in axes:
        ax.set_ylim(0,100)
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Seed1 60M · frozen inference only · no new training",fontsize=12)
    fig.supxlabel("Observed points, not an optimum/significance claim; initial32 are a subset. Cumulative10 success does not verify stable holding.",fontsize=8)
    fig.savefig(ROOT/"success_vs_sampling.png",dpi=175)
    motion.plt.close(fig)


if __name__ == "__main__":
    main()
