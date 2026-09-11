"""Read-only CPU summary/figure for partial or complete matched 50M STR jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

from summarize_str_retention import evaluation, snapshot, tb_curves
from summarize_str_control_ab import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(2)
    manifest = json.loads((root / "manifest.json").read_text())
    status_path = root / "queue_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "not_started", "jobs": {}}
    assertions = json.loads((root / "config_assertions.json").read_text())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "queue_status": status["status"],
              "manifest_sha256": sha256(root / "manifest.json"), "conditions": {},
              "metadata_note": "prepared_only/launched fields in original preparation artifacts are historical; live status comes from queue_status.json, which is not modified.",
              "limitations": ["Missing checkpoints/evaluations mean pending, never zero success.",
                  "Eval success is true all_goals_hit; task lift and sustained-near-palm proxy remain separate.",
                  "Online TensorBoard values are logged window means, not independent full-budget success rates.",
                  "TB x-axis is processed transitions; checkpoint step names are vector interactions; optimizer counters are read independently.",
                  "Only seed0 and fixed single-task initial settings; no generalization, convergence, or paper-reproduction claim.",
                  "Each condition trains continuously to50M before its stored10/20/30/40/50M checkpoints are evaluated. Early missing evals do not indicate failure.",
                  "Connecting eval points are guides between measured checkpoints; online train returns and evaluation returns use different protocols/state occupancy."]}
    for condition in ("arm1_hand1", "arm1_hand01"):
        cfg_path = root / f"configs/{condition}_seed0.json"
        cfg = json.loads(cfg_path.read_text())
        assert sha256(cfg_path) == assertions["conditions"][condition]["resolved_sha256"]
        runs = tb_curves(root / "runs/budget50m" / condition)
        milestones = {}
        for step in assertions["checkpoint_steps"]:
            record = snapshot(root / f"models/{condition}/seed0/step{step}", {}, cfg, runs)
            record["vector_interactions"] = step
            record["transitions"] = step * cfg["num_train_envs"]
            record["evaluations"] = {}
            for sampling in ("deterministic", "stochastic"):
                job_id = f"{condition}_seed0_step{step}_{sampling}"
                job = next(job for job in manifest["jobs"] if job["id"] == job_id)
                argv = job["argv"]
                overrides = dict(argv[i + 1].split("=", 1) for i, value in enumerate(argv[:-1]) if value == "--override")
                for part in ("arm", "hand"):
                    assert float(overrides[f"env.task_cfg_overrides.action.{part}_moving_average"]) == cfg["env"]["task_cfg_overrides"]["action"][f"{part}_moving_average"]
                value = evaluation(Path(job["expected_completion_file"]))
                value["queue_job_status"] = status.get("jobs", {}).get(job_id, {}).get("status", "pending")
                record["evaluations"][sampling] = value
            milestones[str(step)] = record
        report["conditions"][condition] = {"control": cfg["env"]["task_cfg_overrides"]["action"],
            "config_sha256": sha256(cfg_path), "train_queue_status": status.get("jobs", {}).get(f"{condition}_seed0_train50m", {}).get("status", "pending"),
            "tensorboard_runs": runs, "milestones": milestones}
    source_capture = json.loads((root / "source_hashes.json").read_text())
    changes = [{"path": row["path"], "captured_sha256": row["sha256"], "current_sha256": sha256(row["path"])}
               for row in source_capture["files"] if sha256(row["path"]) != row["sha256"]]
    report["source_integrity"] = {"captured": len(source_capture["files"]), "matching": len(source_capture["files"]) - len(changes),
                                  "changes": changes, "note": "Original source_hashes.json is preserved; intentional diagnostic harness extensions are not training-core equivalence claims."}
    saved = [m for c in report["conditions"].values() for m in c["milestones"].values() if m["status"] == "saved_complete"]
    report["checkpoint_audit"] = {"completed_milestones": len(saved), "all_finite": all(m["all_finite"] for m in saved) if saved else None,
                                  "tensor_count": sum(m["tensor_count"] for m in saved), "tensor_elements": sum(m["tensor_elements"] for m in saved)}
    (root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_markdown(root, report)
    plot(root, report)
    print(json.dumps({"queue_status": report["queue_status"], "saved_checkpoints": len(saved), "available_evals": sum(e["status"] != "pending" for c in report["conditions"].values() for m in c["milestones"].values() for e in m["evaluations"].values())}))


def write_markdown(root, report):
    lines = ["# 50M 连续训练预算对照", "", f"读取时间：{report['created_utc']}。实时队列状态：{report['queue_status']}。", "",
             "arm filter均为1.0，对照hand=1.0与0.1；seed0，固定STR单物体/初态/goal。1024env，从零连续训练50,001,920 transitions；不加载10M replay再分段续跑。前10M学习率配方保持原值，之后沿用该原日程end学习率。", "",
             "每个条件完成连续50M后，才逐个评测保存的10/20/30/40/50M模型。因此目前缺评测不表示训练没成功；不要把准备阶段metadata的prepared_only当作当前运行状态。", "",
             "![Training and checkpoint evaluation curves](" + str(root / "curve.png") + ")", "",
             "## Checkpoint 与独立评测", "",
             "| hand filter | transitions | sampling | 完成/请求 | 真实goal成功 | 抬升 | 持握代理 | 平均raw reward |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for condition in report["conditions"].values():
        hand = condition["control"]["hand_moving_average"]
        for milestone in condition["milestones"].values():
            for mode, e in milestone["evaluations"].items():
                if e["status"] == "pending":
                    lines.append(f"| {hand} | {milestone['transitions']:,} | {mode} | 待完成 | — | — | — | — |")
                else:
                    n, c = e["episodes"], e["counts"]
                    lines.append(f"| {hand} | {milestone['transitions']:,} | {mode} | {n}/{e['requested']} | {c['success']}/{n} | {c['ever_task_lift']}/{n} | {c['ever_sustained_lift_near_palm_proxy']}/{n} | {e['mean_return']:.2f} |")
    lines += ["", "## 已保存模型的实际更新次数", "", "| hand filter | transitions | network calls | actor / critic / temp optimizer steps | alpha | reward除数 | 有限性 |", "| --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for condition in report["conditions"].values():
        for m in condition["milestones"].values():
            if m["status"] != "saved_complete":
                continue
            counts = [m["actual_optimizer_steps"][key] or [0] for key in ("actor", "critic", "temperature")]
            steps = " / ".join(f"{min(c)}–{max(c)}" for c in counts)
            lines.append(f"| {condition['control']['hand_moving_average']} | {m['transitions']:,} | {m['agent_network_calls']} | {steps} | {m['alpha']:.6f} | {m['reward_normalizer']['reward_divisor']:.3f} | {m['all_finite']} |")
    lines += ["", "## 最新在线TB窗口", "", "| hand filter | 指标 | 最新值 / transitions | 历史最小 | 历史最大 |", "| --- | --- | --- | ---: | ---: |"]
    for condition in report["conditions"].values():
        for run in condition["tensorboard_runs"]:
            for tag in ("episode/return", "episode/final/all_goals_hit", "episode/cumulative/lift_bonus_rew", "actor/entropy", "temperature/value"):
                c = run["curves"].get(tag)
                if c and "error" not in c:
                    lines.append(f"| {condition['control']['hand_moving_average']} | {tag} | {c['last']['value']:.6g} / {c['last']['step']:,} | {c['minimum']['value']:.6g} | {c['peak']['value']:.6g} |")
    integrity = report["source_integrity"]
    lines += ["", f"源码捕获复核：{integrity['matching']}/{integrity['captured']}相同；差异路径和前后hash保留在summary.json，不覆盖旧捕获。", "",
              "## 限制", "", *[f"- {note}" for note in report["limitations"]], "",
              "JSON另存全部TB标量、checkpoint两侧真实日志窗口、完整checkpoint SHA256/optimizer计数/有限性，以及评测reward各分量。"]
    (root / "analysis.md").write_text("\n".join(lines) + "\n")


def plot(root, report):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7))
    colors = {"arm1_hand1": "#c85a17", "arm1_hand01": "#17668c"}
    active, plotted_eval = [], False
    for name, condition in report["conditions"].items():
        present = False
        for run in condition["tensorboard_runs"]:
            samples = run["curves"].get("episode/return", {}).get("samples", [])
            if samples:
                axes[1].plot([s["step"] / 1e6 for s in samples], [s["value"] for s in samples], color=colors[name], alpha=.7, linewidth=1)
                present = True
        for mode, marker in (("deterministic", "o"), ("stochastic", "s")):
            rows = [(m["transitions"] / 1e6, m["evaluations"][mode]) for m in condition["milestones"].values() if m["evaluations"][mode]["status"] != "pending"]
            if rows:
                plotted_eval = present = True
                x = [row[0] for row in rows]
                axes[0].plot(x, [100 * e["counts"]["success"] / e["episodes"] for _, e in rows], marker=marker, linestyle="--", color=colors[name], markersize=6)
                axes[1].plot(x, [e["mean_return"] for _, e in rows], marker=marker, linestyle="--", color=colors[name], markersize=6)
        if present:
            active.append(Line2D([], [], color=colors[name], label=f"hand filter {condition['control']['hand_moving_average']}"))
    if not plotted_eval:
        axes[0].text(.5, .5, "No completed rollout evaluations yet\nMissing results are not zero success", transform=axes[0].transAxes, ha="center", va="center")
    axes[0].set(ylabel="Task goal success (%)", ylim=(-5, 105), title="Independent checkpoint evaluation")
    axes[1].set(ylabel="Raw task return per episode", title="Online windows and checkpoint evaluation")
    for ax in axes:
        ax.set(xlabel="Training transitions (million)", xlim=(0, 52), xticks=[0, 10, 20, 30, 40, 50])
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    handles = active + [Line2D([], [], color=".35", linewidth=1, label="Online TB window"),
                       Line2D([], [], color=".35", marker="o", linestyle="--", label="Det. eval"),
                       Line2D([], [], color=".35", marker="s", linestyle="--", label="Stoch. eval")]
    fig.suptitle(f"Continuous 50M STR budget diagnostic — {report['queue_status']}", fontsize=14)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .055), ncol=len(handles), frameon=False)
    fig.text(.5, .018, "Seed 0; fixed task; arm filter 1.0. Dashed connections are guides. Only available logs/results are plotted.", ha="center", fontsize=8.7)
    fig.tight_layout(rect=(0, .14, 1, .92))
    fig.savefig(root / "curve.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
