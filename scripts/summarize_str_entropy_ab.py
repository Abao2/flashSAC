"""Partial-safe CPU summary for the alpha-initialization x critic-gate jobs."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from summarize_str_retention import evaluation, snapshot, tb_curves
from summarize_str_control_ab import sha256


def validate_branch(branch, config):
    """Do not silently map a different budget/final model into this comparison."""
    assert config["num_env_steps"] == 2000896 and config["num_train_envs"] == 1024
    assert config["num_interaction_steps"] == config["num_env_steps"] // config["num_train_envs"] == 1954
    final = Path(config["output_root"]) / config["save_path"] / "step1954"
    assert final == Path(branch["final_checkpoint"])
    assert Path(config["output_root"]) == Path(branch["output_root"])
    assert Path(config["agent_load_path"]) == Path(branch["initial_checkpoint"])
    assert config["agent"]["temp_initial_value"] == branch["alpha"]
    assert config["env"]["task_cfg_overrides"]["action"] == {"arm_moving_average": .1, "hand_moving_average": .1}


def checkpoint_label(branch, path):
    path = Path(path)
    if path == Path(branch["final_checkpoint"]):
        return "final"
    if path.parent == Path(branch["output_root"]) / "update_snapshots" and path.name.startswith("update") and path.name[6:].isdigit():
        return path.name
    raise ValueError(f"Checkpoint does not belong to branch {branch['id']}: {path}")


def critic_summary(path):
    if not path.is_file():
        return {"status": "pending", "path": str(path)}
    data = json.loads(path.read_text())
    keys = ["q_recorded", "minq_bottom5_atoms_mass", "minq_top5_atoms_mass", "target_unprojected_mean",
            "target_clipped_mean", "target_mass_below_support", "target_mass_above_support", "next_entropy_bonus"]
    return {"status": "available", "path": str(path), "sha256": sha256(path), "dataset": data["dataset"],
            "actual_alpha": data["alpha"], "reward_divisor": data["reward_denominator"], "limitations": data["limitations"],
            "phases": {phase: {"samples": row["sampled_transitions"], **{key: row["metrics"][key]["mean"] for key in keys},
                               "sampled_next_action_entropy_estimate": row["metrics"]["next_entropy_bonus"]["mean"] / data["alpha"],
                               "q_to_entropy_gradient_norm_ratio": row["actor_gradient"]["q_to_entropy_gradient_norm_ratio"]["mean"]}
                       for phase, row in data["phases"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(2)
    manifest = json.loads((root / "manifest.json").read_text())
    assertions = json.loads((root / "config_assertions.json").read_text())
    status_path = root / "queue_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "not_started", "jobs": {}}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "queue_status": status["status"],
              "manifest_sha256": sha256(root / "manifest.json"), "branches": {},
              "limitations": ["Pending means missing, not zero success, zero updates, or zero entropy.",
                  "Initial alpha is manipulated, but native adaptive temperature remains active. Read actual saved alpha and TB trajectory.",
                  "Critic gate0/2000 changes data collection and actor/temp update count and scheduler position at equal environment budget.",
                  "Network-call snapshot numbers are not actual actor optimizer steps; AMP can skip applied steps.",
                  "Shared BC initialization and fixed single-task seed0 reset; this is a retention diagnostic, not from-scratch training/generalization.",
                  "TB H is a logged actor-batch window. Critic H estimate is sampled next-action -logpi on expert data; they are different distributions/BN conventions.",
                  "Critic target clipping is recomputed on frozen phase-balanced expert samples, not historical replay clipping frequency.",
                  "Prepared-only metadata is historical; queue_status.json supplies live state. This summary never edits queue/config/model files."]}
    for branch in manifest["branches"]:
        name, directory = branch["id"], Path(branch["output_root"])
        config = json.loads(Path(branch["resolved_config_file"]).read_text())
        validate_branch(branch, config)
        initial_hashes = {key: sha256(Path(branch["initial_checkpoint"]) / key)
                          for key in assertions["initial_checkpoints"][name]["component_sha256"]}
        assert initial_hashes == assertions["initial_checkpoints"][name]["component_sha256"]
        temp = torch.load(Path(branch["initial_checkpoint"]) / "temperature.pt", map_location="cpu", weights_only=True)
        initial_alpha = float(next(value for key, value in temp["network_state_dict"].items() if key.endswith("log_temp")).exp())
        assert math.isclose(initial_alpha, branch["alpha"], rel_tol=1e-5)
        audit_file = directory / "warmup_audit.json"
        live = json.loads(audit_file.read_text()) if audit_file.exists() else {"status": "not_started"}
        runs = tb_curves(directory / "runs")
        recorded = {checkpoint_label(branch, item["path"]): item for item in live.get("snapshots", [])}
        recorded["final"] = {"network_audit": live.get("final_network_audit", {}), "attempted_updates": live.get("attempted_updates")}
        checkpoints, evaluations, critics = {}, {}, {}
        for job in manifest["jobs"]:
            if not job["id"].startswith(name + "_") or job["kind"] == "training":
                continue
            argv = job["argv"]
            path = Path(argv[argv.index("--checkpoint") + 1])
            label = checkpoint_label(branch, path)
            if label not in checkpoints:
                checkpoints[label] = snapshot(path, recorded.get(label, {}), config, runs)
            output = Path(job["expected_completion_file"])
            if job["kind"] == "diagnostic":
                sampling = argv[argv.index("--sampling") + 1]
                entry = evaluation(output)
                entry["queue_job_status"] = status.get("jobs", {}).get(job["id"], {}).get("status", "pending")
                evaluations.setdefault(label, {})[sampling] = entry
            else:
                critics[label] = critic_summary(output)
        # Include saved early checkpoints even when their rollout was not requested.
        for label, item in recorded.items():
            if label != "final" and label not in checkpoints:
                checkpoints[label] = snapshot(Path(item["path"]), item, config, runs)
        report["branches"][name] = {"initial_alpha": branch["alpha"], "loaded_initial_alpha": initial_alpha,
            "critic_warmup_calls": branch["critic_warmup_updates"], "training_transitions": config["num_env_steps"],
            "mapping_check_passed": True, "config_sha256": sha256(branch["resolved_config_file"]), "initial_component_sha256": initial_hashes,
            "train_queue_status": status.get("jobs", {}).get(name + "_train", {}).get("status", "pending"),
            "live_audit": live, "checkpoints": checkpoints, "evaluations": evaluations, "critics": critics, "tensorboard_runs": runs}
    (root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_markdown(root, report)
    print(json.dumps({"queue_status": report["queue_status"], "branches_checked": len(report["branches"]),
        "available_eval": sum(e["status"] != "pending" for b in report["branches"].values() for row in b["evaluations"].values() for e in row.values()),
        "available_critics": sum(c["status"] != "pending" for b in report["branches"].values() for c in b["critics"].values())}))


def count_cell(value):
    if not value or value["status"] == "pending":
        return "待完成"
    return f"{value['counts']['success']}/{value['episodes']}; {value['counts']['ever_task_lift']}/{value['episodes']}"


def write_markdown(root, report):
    lines = ["# 初始熵权重 × critic-only gate 对照", "", f"读取：{report['created_utc']}；队列状态：{report['queue_status']}。", "",
             "6个分支：初始alpha=.01/.0001/.000001 × gate0/2000，均从同一成功BC actor初始化。每支2,000,896 transitions、1024env；arm/hand平滑均.1。**温度仍自适应，初始alpha不是固定alpha。**", "",
             "## Snapshot 评测与实际更新", "",
             "det/stoch格依次表示：真实goal成功/完成回合；抬升/完成回合。pending不当作0。", "",
             "| 初始alpha | gate | snapshot | actor实际step | 实际alpha | D | det：goal；lift | stoch：goal；lift |",
             "| ---: | ---: | --- | --- | ---: | ---: | --- | --- |"]
    for branch in report["branches"].values():
        for label, values in branch["evaluations"].items():
            checkpoint = branch["checkpoints"][label]
            if checkpoint["status"] == "saved_complete":
                counts = checkpoint["actual_optimizer_steps"]["actor"] or [0]
                steps = f"{min(counts)}–{max(counts)}"
                alpha, divisor = f"{checkpoint['alpha']:.7g}", f"{checkpoint['reward_normalizer']['reward_divisor']:.3f}"
            else:
                steps = alpha = divisor = "—"
            lines.append(f"| {branch['initial_alpha']:g} | {branch['critic_warmup_calls']} | {label} | {steps} | {alpha} | {divisor} | {count_cell(values.get('deterministic'))} | {count_cell(values.get('stochastic'))} |")
    lines += ["", "## 已完成的critic查询", "",
              "同一官方成功轨迹上的冻结诊断，非训练replay。H_next为该查询的单样本-per-state估计；不和TB的actor-batch H混称。", "",
              "| 初始alpha / gate | snapshot / phase | Q | 下界外target质量 | 上界外target质量 | D | 实际alpha | H_next估计 | Q/熵梯度比 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    found = False
    for branch in report["branches"].values():
        for label, critic in branch["critics"].items():
            if critic["status"] == "pending":
                continue
            found = True
            for phase in ("reset", "far_from_object", "lifted_proxy", "near_goal_center_proxy"):
                row = critic["phases"].get(phase)
                if row:
                    lines.append(f"| {branch['initial_alpha']:g} / {branch['critic_warmup_calls']} | {label}/{phase} | {row['q_recorded']:.4f} | {row['target_mass_below_support']*100:.2f}% | {row['target_mass_above_support']*100:.2f}% | {critic['reward_divisor']:.3f} | {critic['actual_alpha']:.7g} | {row['sampled_next_action_entropy_estimate']:.3f} | {row['q_to_entropy_gradient_norm_ratio']:.5g} |")
    if not found:
        lines += ["", "尚无已完成critic报告；没有填0或生成推断曲线。"]
    lines += ["", "## 当前TB窗口", "", "| 分支 | 指标 | 最新值 / transitions | 最小 | 最大 |", "| --- | --- | --- | ---: | ---: |"]
    for name, branch in report["branches"].items():
        for run in branch["tensorboard_runs"]:
            for tag in ("actor/entropy", "temperature/value", "episode/return", "episode/final/all_goals_hit"):
                curve = run["curves"].get(tag)
                if curve and "error" not in curve:
                    lines.append(f"| {name} | {tag} | {curve['last']['value']:.6g} / {curve['last']['step']:,} | {curve['minimum']['value']:.6g} | {curve['peak']['value']:.6g} |")
    lines += ["", "## 边界", "", *[f"- {value}" for value in report["limitations"]], "",
              "JSON还包括评测raw reward及各分量、实际优化器/AMP计数、checkpoint有限性与hash、live audit核对和全部TB标量。当前不生成空图。"]
    (root / "analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
