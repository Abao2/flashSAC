"""Read-only completed protocol/reset-reward evidence; writes isolated md/json only."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
from summarize_str_retention import evaluation
from summarize_str_control_ab import sha256


def cohort(rows):
    assert rows and all(row["complete"] for row in rows)
    flags = ["success", "ever_task_lift", "dropped_below_reset_after_task_lift", "terminated", "truncated"]
    return {"episodes": len(rows), "episode_ids": [row["episode_id"] for row in rows],
        "counts": {key: sum(bool(row[key]) for row in rows) for key in flags},
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_length": float(np.mean([row["length"] for row in rows]))}


def main():
    baseline_path = ROOT / "evaluations/arm1_hand01_seed0/step48830/stochastic/summary.json"
    baseline = evaluation(baseline_path)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "completed_evidence_audit",
        "reused_baseline": baseline, "groups": {},
        "source_sha256": {str(p): sha256(p) for p in [baseline_path, baseline_path.with_name("metadata.json"), baseline_path.with_name("episodes.jsonl")]},
        "limitations": ["All eight jobs used the same final50M hand.1 checkpoint and fixed task; episode counts are not independent training seeds.",
            "Startup bundles randomized initial progress with one initial random action. It is not full from-scratch warmup/RNG-history restoration.",
            "TF32/high-matmul is a bundle; inference compile was still disabled. Startup and TF32 were not crossed together.",
            "First-reset previous-reward intervention affects only initial32 episodes directly; later episodes are downstream outcomes, not independently intervened repeats.",
            "No live training simulator or previous near-success actor snapshot was recovered; a short online window and final frozen checkpoint remain different objects."]}
    digests = set()
    all_task_overrides = []
    for group in ["protocol_check", "reset_reward_probe"]:
        folder = ROOT / group
        manifest = json.loads((folder / "manifest.json").read_text())
        status = json.loads((folder / "queue_status.json").read_text())
        assert status["status"] == "completed"
        report["source_sha256"].update({str(folder / name): sha256(folder / name) for name in ["manifest.json", "queue_status.json"]})
        results = {}
        for job in manifest["jobs"]:
            path = Path(job["expected_completion_file"])
            result = evaluation(path)
            assert result["complete_budget"] and result["model_unchanged"] and result["status"] == "complete"
            metadata = json.loads(path.with_name("metadata.json").read_text())
            rows = [json.loads(line) for line in path.with_name("episodes.jsonl").read_text().splitlines()]
            assert all(row["complete"] for row in rows)
            args = metadata["arguments"]
            assert args["sampling"] == "stochastic" and args["noise_multiplier"] == 1 and args["noise_repeat"] == 0
            assert args["checkpoint"] == manifest["checkpoint"] and metadata["model_digest_before"] == json.loads(path.read_text())["model_digest_after"]
            digests.add(metadata["model_digest_before"])
            all_task_overrides.append(metadata["resolved_config"]["env"]["task_cfg_overrides"])
            result.update({"num_envs": args["num_envs"], "seed": args["seed"], "training_startup": args["training_startup"],
                "training_numerics_requested": args["training_numerics"],
                "actual_numerics": metadata["torch_numerics_after_env_init"], "cohorts": {}})
            assert metadata["torch_numerics_after_env_init"] == metadata["torch_numerics_after_rollout"]
            for label, selected in [("initial_batch", [row for row in rows if row["episode_id"] < args["num_envs"]]),
                                    ("subsequent_episodes", [row for row in rows if row["episode_id"] >= args["num_envs"]])]:
                result["cohorts"][label] = cohort(selected)
            if args["training_startup"]:
                assert all(row["startup_episode"] == (row["episode_id"] < args["num_envs"]) for row in rows)
            inputs = [path, path.with_name("metadata.json"), path.with_name("episodes.jsonl")]
            intervention_path = path.with_name("reset_reward_intervention.json")
            if group == "reset_reward_probe":
                intervention = json.loads(intervention_path.read_text())
                assert intervention["status"] == "initial_reset_verified" and intervention["explicit_reset_calls"] == 1
                assert intervention["initial_cohort_episode_ids"] == list(range(32))
                encoded = intervention["expected_encoded_reward"]
                assert np.allclose(intervention["returned_policy_reward"], encoded, atol=1e-7, rtol=0)
                result["initial_reward_intervention"] = {"raw_previous_reward": intervention["raw_previous_reward"],
                    "expected_encoded_reward": encoded, "actual_policy_reward": intervention["returned_policy_reward"][0],
                    "explicit_reset_calls": 1, "initial_cohort_episode_ids": list(range(32))}
                inputs.append(intervention_path)
            report["source_sha256"].update({str(p): sha256(p) for p in inputs})
            results[job["id"]] = result
        report["groups"][group] = results
    assert len(digests) == 1 and all(task == all_task_overrides[0] for task in all_task_overrides)
    assert [len(report["groups"][group]) for group in report["groups"]] == [6, 2]
    model_path = Path(manifest["checkpoint"]) / "actor.pt"
    report["source_sha256"][str(model_path)] = sha256(model_path)
    report["same_frozen_actor_digest_all_eight_jobs"] = next(iter(digests))
    report["same_task_overrides_all_eight_jobs"] = True
    report["eight_job_completed_episodes"] = sum(result["episodes"] for group in report["groups"].values() for result in group.values())
    zero = report["groups"]["reset_reward_probe"]["initial_reward0_seed0"]
    report["zero_injection_matches_previous_baseline_aggregate"] = zero["counts"] == baseline["counts"] and zero["mean_return"] == baseline["mean_return"]
    (ROOT / "protocol_followup.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_markdown(report)
    print(json.dumps({"completed_jobs": 8, "episodes": report["eight_job_completed_episodes"],
        "same_actor": len(digests) == 1, "zero_matches_baseline": report["zero_injection_matches_previous_baseline_aggregate"]}))


def write_markdown(report):
    lines = ["# 50M train/frozen差异：8项后续对照", "",
        "**8/8任务完成全部预算，6784个记录回合均0 goal。** 同一final50M hand=.1 actor、arm1/hand.1、原生随机采样（noise multiplier1、global zeta repeat）、同一固定eraser/目标；没有新训练。6784是记录总数，不能当成6784个独立训练seed或泛化任务。", "",
        "## 协议矩阵", "",
        "| 已完成任务 | env | 启动处理 | 实际TF32/matmul | seed | 完成/请求 | goal | 曾抬升 | 抬升后低于初始高度 | raw return | 平均步数 |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in report["groups"]["protocol_check"].items():
        c, n, numerics = result["counts"], result["episodes"], result["actual_numerics"]
        link = f"[{name}]({result['path']})"
        lines.append(f"| {link} | {result['num_envs']} | {'初始随机progress+1随机action' if result['training_startup'] else 'clean'} | {numerics['cuda_matmul_allow_tf32']}/{numerics['float32_matmul_precision']} | {result['seed']} | {n}/{result['requested']} | {c['success']}/{n} | {c['ever_task_lift']}/{n} | {c['dropped_below_reset_after_task_lift']}/{n} | {result['mean_return']:.3f} | {result['mean_length']:.3f} |")
    baseline = report["reused_baseline"]
    lines += ["", f"已有clean32/seed0基准复用，没有记成新实验：goal0/{baseline['episodes']}、lift128/128、低于初始高度127/128、rawreturn{baseline['mean_return']:.3f}。", "",
        "启动对照必须分组：随机progress缩短初始回合可用时间，不能只拿总lift下降推断控制变差。", "",
        "| 启动任务 | cohort | 回合数 | goal | 曾抬升 | 后续低于初始高度 | raw return | 步数 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in report["groups"]["protocol_check"].items():
        if not result["training_startup"]:
            continue
        for label, group in result["cohorts"].items():
            c, n = group["counts"], group["episodes"]
            lines.append(f"| {name} | {label} | {n} | {c['success']}/{n} | {c['ever_task_lift']}/{n} | {c['dropped_below_reset_after_task_lift']}/{n} | {group['mean_return']:.3f} | {group['mean_length']:.3f} |")
    lines += ["", "## 只改变首帧上一回合reward", "",
        "两组32env、seed0、native stochastic、TF32=False/highest；只在第一次显式reset前写入raw.reward_buf，随后真实reward计算和auto-reset完全保留。成功值取已记录BC成功终止reward中位数100.14024353，实际actor首帧字段=1.001402378（FP32），0组为0。intervention sidecar验证只有一次explicit reset；summary另外验证各128/128回合完成。", "",
        "| 首帧reward原始值 | cohort | 完成回合 | goal | 曾抬升 | 后续低于初始高度 | raw return | 步数 |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in report["groups"]["reset_reward_probe"].items():
        raw = result["initial_reward_intervention"]["raw_previous_reward"]
        for label, group in result["cohorts"].items():
            c, n = group["counts"], group["episodes"]
            lines.append(f"| {raw:.8f} | {label} | {n} | {c['success']}/{n} | {c['ever_task_lift']}/{n} | {c['dropped_below_reset_after_task_lift']}/{n} | {group['mean_return']:.3f} | {group['mean_length']:.3f} |")
    lines += ["", "initial_batch就是IDs0–31，直接受干预；subsequent_episodes是IDs32–127，是后续轨迹，不能写成另外96个独立首帧干预。两组总计分别0/128。注入0与之前clean基准的完整计数和平均return完全相同，支持该包装在基准条件下未改变结果。成功reward注入改变动作但未产生goal，不能把少量drop/return差别过度解读。", "",
        "## 排除了什么，尚未排除什么", "",
        "- 在这一个checkpoint和这些预算内，单独增至1024env、启动bundle、TF32/high-matmul bundle、另一个rollout seed或初始成功reward，均不足以恢复训练窗口中的高goal率；不是“32env一定看不到成功”或“首帧reward=0是唯一原因”的证据。",
        "- TF32的实际前后状态已记录且一致，但compile仍关闭；没有测试TF32×startup联合，也没有重建50M原训练进程的隐藏仿真/RNG状态。",
        "- 在线窗口仍含策略/BN更新中的多个版本、完成回合时间选择效应；没有逐回合成功终态dump和每个版本actor，因此无法仅靠最后actor.pt验证那30个vector-step的完整事实。",
        "- 抬升/近掌部代理不证明稳定抓握；低于初始高度也是明确几何指标，不是接触传感器确认。", "",
        "## 最有区分度的两个廉价下一步（仅建议，未执行）", "",
        "1. **同state同noise的编译/加载前向对照，先不跑物理。** 同一actor.pt、同一批已记录reset/抬升/近goal观测，分别走原生训练配置的compiled actor和当前eager冻结actor；固定同一cachednoise，比较μ、σ、action、BN buffers。若不一致，先定位保存/加载/编译路径；若一致，降低该路径嫌疑，再决定是否值得短rollout。TF32对照本身不能代替compile检查。",
        "2. **在原生训练再次出现成功窗口时，原进程冻结，再clean reset。** 保存每次done的pre-reset物体/goal/keypoint误差和actor标识；冻结所有优化器及BN更新，先不重建/不额外reset跑两个horizon，再用同一冻结actor执行clean reset跑两个horizon。继续成功但clean后失败指向状态/重置分布；冻结后立刻失效指向移动策略/时间窗口；独立keypoint重建不支持日志则回到成功接口。当前原训练进程已结束，不能假装这个live-state实验已经做过，必要时在下一次受控短续训中捕获。", "",
        "这些是鉴别实验，不是加reward、换网络或继续盲调超参数。", "",
        "## 可追溯性", "",
        "所有8组actor前后digest相同、task overrides相同、episode IDs唯一且达到请求预算。JSON列出每个summary/metadata/episodes.jsonl、两个manifest/status与intervention sidecar的SHA256，以及actor.pt SHA256。`prepared_only`保留历史准备含义，完成事实采用queue_status和实际summary。", "",
        f"重跑CPU汇总：`/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python {ROOT / 'summarize_protocol_followup.py'}`。无GPU、无core/manifest/main report修改。"]
    (ROOT / "protocol_followup.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
