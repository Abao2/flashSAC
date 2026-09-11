"""CPU-only, rerunnable retention artifact summary, including incomplete queues.

Reads checkpoint/TB/rollout files without touching training or queue state.
Writes only retention_summary.json and retention_summary.md below --root.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from summarize_str_control_ab import checkpoint_audit, curve_stats, sha256


def evaluation(path):
    if not path.is_file():
        return {"status": "pending", "path": str(path)}
    summary = json.loads(path.read_text())
    rows = [json.loads(line) for line in (path.parent / "episodes.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["complete"]]
    n = len(rows)
    assert n == summary["episodes_completed"] == len({row["episode_id"] for row in rows})
    flags = ["success", "ever_task_lift", "ever_sustained_lift_near_palm_proxy",
             "dropped_below_reset_after_task_lift", "terminated", "truncated"]
    counts = {key: sum(bool(row[key]) for row in rows) for key in flags}
    for key in flags[:4]:
        if n:
            assert abs(summary[key]["mean"] * n - counts[key]) < 1e-5
    components = {key: float(np.mean([row["reward_components"][key] for row in rows]))
                  for key in rows[0]["reward_components"]} if rows else {}
    return {"path": str(path), "sha256": sha256(path), "status": summary["status"],
            "episodes": n, "requested": summary["episodes_requested"],
            "complete_budget": n == summary["episodes_requested"], "counts": counts,
            "mean_return": summary["return"]["mean"], "mean_length": summary["length"]["mean"],
            "reward_components": components, "model_unchanged": summary["model_unchanged"]}


def tb_curves(root):
    # Separate run directories are never merged: a restart must remain visible.
    result = []
    for directory in sorted({file.parent for file in root.rglob("events.out.tfevents*")}):
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        accumulator.Reload()
        curves = {}
        for tag in accumulator.Tags()["scalars"]:
            if tag.startswith(("actor/", "critic/", "temperature/", "warmstart/", "episode/")):
                events = accumulator.Scalars(tag)
                try:
                    curves[tag] = {**curve_stats(events), "samples": [
                        {"step": int(e.step), "value": float(e.value)} for e in events]}
                except ValueError as error:
                    curves[tag] = {"error": str(error)}
        result.append({"directory": str(directory), "curves": curves})
    return result


def bracketing_tb(runs, transitions):
    tags = ["actor/entropy", "temperature/value", "episode/return", "episode/final/all_goals_hit"]
    answer = []
    for run in runs:
        selected = {}
        for tag in tags:
            samples = run["curves"].get(tag, {}).get("samples", [])
            before = [s for s in samples if s["step"] <= transitions]
            after = [s for s in samples if s["step"] >= transitions]
            selected[tag] = {"at_or_before": before[-1] if before else None,
                             "at_or_after": after[0] if after else None}
        answer.append({"directory": run["directory"], "brackets": selected})
    return answer


def snapshot(path, recorded, config, runs):
    if not (path / "agent_state.pt").is_file():
        return {"path": str(path), "status": "not_saved_complete"}
    try:
        audit = checkpoint_audit(path)
        norm = torch.load(path / "reward_normalizer.pt", map_location="cpu", weights_only=True)
        temp = torch.load(path / "temperature.pt", map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        return {"path": str(path), "status": "read_error_or_write_in_progress", "error": str(error)}
    norm = {key: float(norm[key]) for key in ("G_r_max", "G_rms_mean", "G_rms_var", "G_rms_count")}
    norm["reward_divisor"] = max(math.sqrt(norm["G_rms_var"] + 1e-8),
                                 norm["G_r_max"] / config["agent"]["normalized_G_max"])
    steps = {name.removesuffix(".pt"): item["optimizer_applied_steps"]
             for name, item in audit["files"].items() if name in ("actor.pt", "critic.pt", "temperature.pt")}
    checks = {}
    for name, values in steps.items():
        expected = recorded.get("network_audit", {}).get(name)
        if expected:
            checks[name] = ((min(values, default=0), max(values, default=0)) ==
                            (expected["optimizer_step_min"], expected["optimizer_step_max"]))
    assert all(checks.values()), f"Saved optimizer steps disagree with live audit: {path}"
    logs = [v for key, v in temp["network_state_dict"].items() if key.endswith("log_temp")]
    assert len(logs) == 1
    transitions = round(norm["G_rms_count"])
    return {"path": str(path), "status": "saved_complete", "agent_network_calls": audit["files"]["agent_state.pt"]["stored_update_step"],
            "actual_optimizer_steps": steps, "attempted_updates": recorded.get("attempted_updates"),
            "live_audit_matches_optimizer_checkpoint": checks,
            "alpha": float(logs[0].exp()), "reward_normalizer": norm,
            "rms_sample_count_approx_transitions": transitions,
            "tensor_count": audit["tensor_count"], "tensor_elements": audit["tensor_elements"],
            "all_finite": audit["all_finite"], "checkpoint_files": audit["files"],
            "tb_brackets": bracketing_tb(runs, transitions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="warmstart directory containing retention_manifest.json")
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(2)
    manifest = json.loads((root / "retention_manifest.json").read_text())
    status_path = root / "retention_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "not_started", "jobs": {}}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "queue_status": status["status"],
              "manifest_sha256": sha256(root / "retention_manifest.json"), "branches": {}, "baselines": {},
              "limitations": ["Partial read-only snapshot: pending outputs are not failures or zero success.",
                 "Success is task all_goals_hit; lift and sustained-near-palm are separate flags/proxies.",
                 "Fixed single-task reset distribution, not generalization or from-scratch learning.",
                 "TB scalars are logged window means in environment transitions; snapshot IDs are network calls.",
                 "TB brackets are adjacent logged windows, not exact instantaneous checkpoint entropy/alpha.",
                 "Reward normalizer is not logged to TB; exact saved statistics and scaling divisor are read from checkpoint.",
                 "RMS sample count approximates processed transitions; it includes initialization epsilon/count and may differ with other initializers.",
                 "Warmup changes collected data and actor/temp optimizer schedule positions; not an isolated offline-Q test."]}
    for label, path in {"selected_seed0_native": root / "frozen_bc_std005_eigen/summary.json",
                        "selected_seed1_native": root / "selected_baselines/seed1_native/summary.json",
                        "selected_seed0_independent": root / "selected_baselines/seed0_independent/summary.json"}.items():
        report["baselines"][label] = evaluation(path)
    for branch in ("immediate", "warmup2000"):
        directory = root / "retention" / branch
        config = json.loads((root / "retention_configs" / f"{branch}.json").read_text())
        audit_path = directory / "warmup_audit.json"
        live = json.loads(audit_path.read_text()) if audit_path.exists() else {"status": "not_started"}
        runs = tb_curves(directory / "runs")
        snapshots = {path.name: snapshot(path, record, config, runs)
                     for record in live.get("snapshots", []) for path in [Path(record["path"]) ]}
        final_record = {"network_audit": live.get("final_network_audit", {}),
                        "attempted_updates": live.get("attempted_updates")}
        final_path = directory / "models/final/step9766"
        snapshots["final"] = snapshot(final_path, final_record, config, runs)
        evaluations = {}
        for job in manifest["jobs"]:
            if job["id"].startswith(branch + "_") and job["kind"] == "diagnostic":
                item = evaluation(Path(job["expected_completion_file"]))
                item["queue_job_status"] = status.get("jobs", {}).get(job["id"], {}).get("status", "pending")
                evaluations[job["id"]] = item
        report["branches"][branch] = {"live_audit": live, "snapshots": snapshots, "evaluations": evaluations,
                                      "tensorboard_runs": runs, "resolved_config_sha256": sha256(root / "retention_configs" / f"{branch}.json")}
    (root / "retention_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_markdown(root, report)
    print(json.dumps({"queue_status": report["queue_status"], "branches": {
        key: {"audit_status": value["live_audit"]["status"], "saved_snapshots": sum(s["status"] == "saved_complete" for s in value["snapshots"].values()),
              "evaluations_available": sum(e["status"] != "pending" for e in value["evaluations"].values())} for key, value in report["branches"].items()}}))


def write_markdown(root, report):
    lines = ["# BC → 原生 SAC：技能保留中途汇总", "", f"读取时间：{report['created_utc']}；队列状态：{report['queue_status']}。", "",
             "只读 CPU 汇总。缺文件表示未完成，不代表成功率为0。成功为真实 task all_goals_hit；抬升和持握代理分开。", "",
             "## Rollout", "", "| 条件 / snapshot | 完成/请求 | 目标成功 | 抬升 | 持握代理 | 平均reward | 平均步数 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    evaluations = {**report["baselines"], **{key: value for branch in report["branches"].values() for key, value in branch["evaluations"].items()}}
    for label, value in evaluations.items():
        if value["status"] == "pending":
            lines.append(f"| {label} | 待完成 | — | — | — | — | — |")
        else:
            n, c = value["episodes"], value["counts"]
            lines.append(f"| {label} | {n}/{value['requested']} | {c['success']}/{n} | {c['ever_task_lift']}/{n} | {c['ever_sustained_lift_near_palm_proxy']}/{n} | {value['mean_return']:.2f} | {value['mean_length']:.2f} |")
    lines += ["", "## Checkpoint 更新次数 / reward normalization", "",
              "每格实际 optimizer step 为 min–max，不用网络文件中恒为0的包装计数代替。", "",
              "| 条件 / snapshot | network calls | actor / critic / temp 实际step | alpha | reward除数 | RMS样本数 | 全tensor有限 |", "| --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for branch, value in report["branches"].items():
        for label, item in value["snapshots"].items():
            if item["status"] != "saved_complete":
                continue
            counts = [item["actual_optimizer_steps"][name] for name in ("actor", "critic", "temperature")]
            steps = " / ".join(f"{min(c, default=0)}–{max(c, default=0)}" for c in counts)
            lines.append(f"| {branch}/{label} | {item['agent_network_calls']} | {steps} | {item['alpha']:.6f} | {item['reward_normalizer']['reward_divisor']:.3f} | {item['rms_sample_count_approx_transitions']} | {item['all_finite']} |")
    lines += ["", "## 已写入TB的窗口", "", "| 条件 | 指标 | 首值 | 最新值 / transitions | 最小值 | 最大值 |", "| --- | --- | ---: | --- | ---: | ---: |"]
    for branch, value in report["branches"].items():
        for run in value["tensorboard_runs"]:
            for tag in ("actor/entropy", "temperature/value", "episode/return", "episode/final/all_goals_hit"):
                curve = run["curves"].get(tag)
                if curve and "error" not in curve:
                    lines.append(f"| {branch} | {tag} | {curve['first']['value']:.5g} | {curve['last']['value']:.5g} / {curve['last']['step']} | {curve['minimum']['value']:.5g} | {curve['peak']['value']:.5g} |")
    lines += ["", "## 解释限制", "", *[f"- {note}" for note in report["limitations"]], "",
              "完整JSON含每个snapshot实际优化器计数与live audit核对、checkpoint SHA256/有限性、reward各分量、TB原始标量序列及snapshot两侧窗口。"]
    (root / "retention_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
