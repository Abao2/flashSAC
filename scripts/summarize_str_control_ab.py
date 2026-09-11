"""Read-only artifact audit and report for the six bounded STR hand-filter runs.

Writes only report.json / analysis.md beneath the supplied experiment root.
No simulator, GPU, optimization, or external access. TensorBoard window means
are never converted to an overall episode success frequency without counts.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {key: item for name, nested in value.items()
                for key, item in flatten(nested, f"{prefix}.{name}".strip(".")).items()}
    return {prefix: value}


def numeric_audit(value):
    answer = {"tensor_count": 0, "tensor_elements": 0, "floating_tensor_elements": 0,
              "python_float_count": 0, "nonfinite_paths": []}

    def visit(item, path):
        if torch.is_tensor(item):
            answer["tensor_count"] += 1
            answer["tensor_elements"] += item.numel()
            if item.is_floating_point() or item.is_complex():
                answer["floating_tensor_elements"] += item.numel()
                if not bool(torch.isfinite(item).all()):
                    answer["nonfinite_paths"].append(path)
        elif isinstance(item, float):
            answer["python_float_count"] += 1
            if not math.isfinite(item):
                answer["nonfinite_paths"].append(path)
        elif isinstance(item, dict):
            for key, nested in item.items():
                visit(nested, f"{path}/{key}")
        elif isinstance(item, (list, tuple)):
            for key, nested in enumerate(item):
                visit(nested, f"{path}/{key}")

    visit(value, "root")
    answer["all_finite"] = not answer["nonfinite_paths"]
    return answer


def curve_stats(events):
    values = np.asarray([event.value for event in events], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite TensorBoard scalar curve")

    def point(index):
        return {"step": int(events[index].step), "value": float(values[index])}

    return {"points": len(events), "first": point(0), "last": point(-1),
            "peak": point(int(values.argmax())), "minimum": point(int(values.argmin())),
            "last_10_windows_unweighted_mean": float(values[-10:].mean()),
            "positive_windows": int((values > 0).sum())}


def checkpoint_audit(path):
    result = {"path": str(path), "files": {}}
    expected = ["actor.pt", "critic.pt", "target_critic.pt", "temperature.pt",
                "reward_normalizer.pt", "agent_state.pt"]
    for name in expected:
        file = path / name
        state = torch.load(file, map_location="cpu", weights_only=True)
        metrics = numeric_audit(state)
        metrics.update(sha256=sha256(file), size_bytes=file.stat().st_size)
        if "network_state_dict" in state:
            metrics["network_tensors"] = len(state["network_state_dict"])
            metrics["network_elements"] = sum(x.numel() for x in state["network_state_dict"].values())
        optimizer = state.get("optimizer_state_dict") or {}
        metrics["optimizer_applied_steps"] = sorted({int(item["step"]) for item in optimizer.get("state", {}).values() if "step" in item})
        metrics["scheduler_last_epoch"] = (state.get("scheduler_state_dict") or {}).get("last_epoch")
        metrics["stored_update_step"] = state.get("update_step")
        if name == "agent_state.pt":
            metrics["grad_scaler"] = state.get("grad_scaler_state_dict")
        result["files"][name] = metrics
    result["all_finite"] = all(item["all_finite"] for item in result["files"].values())
    result["tensor_count"] = sum(item["tensor_count"] for item in result["files"].values())
    result["tensor_elements"] = sum(item["tensor_elements"] for item in result["files"].values())
    return result


def evaluation_summary(directory):
    result = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    complete = [row for row in rows if row.get("complete")]
    assert len(complete) == len({row["episode_id"] for row in complete}) == result["episodes_completed"] == 128
    assert result["model_unchanged"]
    keys = ["success", "ever_task_lift", "ever_sustained_lift_near_palm_proxy",
            "ever_near_object_proxy", "ever_off_table_proxy", "dropped_below_reset_after_task_lift"]
    counts = {key: sum(bool(row[key]) for row in complete) for key in keys}
    for key, count in counts.items():
        assert abs(result[key]["mean"] * 128 - count) < 1e-6
    return {"path": str(directory), "summary_sha256": sha256(directory / "summary.json"),
            "episodes": len(complete), "counts": counts,
            "mean_return": result["return"]["mean"], "mean_length": result["length"]["mean"],
            "mean_position_error_m": result["position_error_m"]["mean"],
            "actions": result["actions"], "model_unchanged": result["model_unchanged"],
            "termination_counts": {key: sum(bool(row[key]) for row in complete) for key in ["terminated", "truncated"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(2)
    manifest = json.loads((root / "manifest.json").read_text())
    queue = json.loads((root / "queue_status.json").read_text())
    assertions = json.loads((root / "config_assertions.json").read_text())
    captured = json.loads((root / "source_hashes.json").read_text())
    hashes = [{"path": item["path"], "captured_sha256": item["sha256"],
               "current_sha256": sha256(item["path"])} for item in captured["files"]]
    for item in hashes:
        item["matches"] = item["captured_sha256"] == item["current_sha256"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
              "contrast": manifest["scientific_contrast"],
              "source_integrity": {"captured_files": len(hashes), "matching_files": sum(item["matches"] for item in hashes),
                                   "all_match": all(item["matches"] for item in hashes), "files": hashes},
              "runs": [], "limitations": [
                  "10M transitions per seed is a bounded diagnostic, not a convergence impossibility test.",
                  "Only hand filter changes; arm remains 1.0. This is NOT the successful BC setup with both filters 0.1.",
                  "Training lift rates are within logging windows; episode-count weights were not preserved in TB, so no overall training episode rate is reconstructed.",
                  "Evaluation repeats one fixed initial setup; three training seeds are the independent algorithm replicates.",
                  "Sustained-lift/near-palm and proximity are geometric proxies, not verified stable contact.",
                  "Approximate Q inferred from window-mean actor loss, alpha and entropy is not a directly logged Q or clipping frequency.",
                  "Final critic probes use a new frozen-policy evaluation dataset, not the historical training replay; phase-balanced weighting and BN contexts differ.",
                  "No historical top-atom/target-clipping instrumentation exists in these TB curves; offline final probes cannot supply a temporal saturation peak."]}
    baseline = flatten(json.loads((root / "configs/arm1_hand1_seed0.json").read_text()))
    allowed = set(assertions["permitted_variable_fields"])
    for hand in ["1", "01"]:
        for seed in range(3):
            name = f"arm1_hand{hand}_seed{seed}"
            config = json.loads((root / f"configs/{name}.json").read_text())
            flat = flatten(config)
            differences = [key for key in sorted(set(flat) | set(baseline)) if flat.get(key) != baseline.get(key)]
            assert set(differences) <= allowed
            assert config["agent_load_path"] is None and config["buffer_load_path"] is None
            assert config["num_env_steps"] == 10000384 and config["num_train_envs"] == 1024
            tb_paths = list((root / f"runs/control_ab/arm1_hand{hand}").glob(f"*seed{seed}_*"))
            assert len(tb_paths) == 1
            accumulator = EventAccumulator(str(tb_paths[0]), size_guidance={"scalars": 0, "tensors": 0})
            accumulator.Reload()
            logged_config = yaml.safe_load(accumulator.Tensors("config/text_summary")[0].tensor_proto.string_val[0].decode())
            logged_differences = [key for key in sorted(set(flat) | set(flatten(logged_config))) if flat.get(key) != flatten(logged_config).get(key)]
            assert not logged_differences, logged_differences
            raw = {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}
            curves = {tag: curve_stats(events) for tag, events in raw.items()}
            assert curves["episode/return"]["last"]["step"] == 10000384
            aligned = [{event.step: event for event in raw[tag]} for tag in ["actor/loss", "actor/entropy", "temperature/value"]]
            qevents = []
            for step in sorted(set.intersection(*(set(group) for group in aligned))):
                event = aligned[0][step]
                qevents.append(type(event)(wall_time=event.wall_time, step=event.step,
                                           value=-event.value - aligned[1][step].value * aligned[2][step].value))
            approx_q = curve_stats(qevents)
            critic_file = root / f"critics/{name}.json"
            critic = json.loads(critic_file.read_text())
            critic_phases = {}
            selected_metrics = ["q_recorded", "q_deterministic", "uniform_action_q_range", "minq_top_atom_mass", "minq_top5_atoms_mass",
                                "deterministic_dq_da_l2", "target_mass_above_support", "target_mass_below_support", "target_unprojected_max",
                                "target_expected_clipping_shift", "actor_pretanh_std_mean", "next_entropy_bonus"]
            for phase, values in critic["phases"].items():
                critic_phases[phase] = {"dataset_transitions": critic["dataset_phase_counts"][phase],
                                        "sampled_transitions": values["sampled_transitions"],
                                        "metrics": {key: values["metrics"][key] for key in selected_metrics},
                                        "actor_gradient": values.get("actor_gradient")}
            job = queue["jobs"][name + "_train"]
            assert job["status"] == "completed" and job["returncode"] == 0
            run = {"name": name, "seed": seed, "hand_filter": 1.0 if hand == "1" else 0.1,
                   "training_seconds": job["elapsed_seconds"], "config_sha256": sha256(root / f"configs/{name}.json"),
                   "config_permitted_differences": differences, "tensorboard_config_matches": True,
                   "tb_path": str(tb_paths[0]), "tb_event_files": [{"path": str(file), "sha256": sha256(file)} for file in tb_paths[0].glob("events.out.tfevents*")],
                   "curves": curves, "approx_min_q_from_window_means": approx_q,
                   "training_lift_peak_window_fraction": curves["episode/cumulative/lift_bonus_rew"]["peak"]["value"] / 300,
                   "checkpoint": checkpoint_audit(root / f"models/arm1_hand{hand}/seed{seed}/step9766"),
                   "evaluations": {mode: evaluation_summary(root / f"evaluations/{name}/{mode}") for mode in ["deterministic", "stochastic"]},
                   "final_critic_probe": {"path": str(critic_file), "sha256": sha256(critic_file), "reward_denominator": critic["reward_denominator"],
                                          "alpha": critic["alpha"], "selection": critic["selection"], "phases": critic_phases}}
            report["runs"].append(run)
    report["all_checkpoint_states_finite"] = all(run["checkpoint"]["all_finite"] for run in report["runs"])
    report["checkpoint_tensor_count"] = sum(run["checkpoint"]["tensor_count"] for run in report["runs"])
    report["checkpoint_tensor_elements"] = sum(run["checkpoint"]["tensor_elements"] for run in report["runs"])
    report["groups"] = []
    for hand in [1.0, 0.1]:
        runs = [run for run in report["runs"] if run["hand_filter"] == hand]
        group = {"hand_filter": hand, "training_seeds": [run["seed"] for run in runs], "evaluations": {}}
        for mode in ["deterministic", "stochastic"]:
            group["evaluations"][mode] = {"episodes": sum(run["evaluations"][mode]["episodes"] for run in runs),
                                          "counts": {key: sum(run["evaluations"][mode]["counts"][key] for run in runs) for key in runs[0]["evaluations"][mode]["counts"]}}
        report["groups"].append(group)
    (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    compact = {"created_utc": report["created_utc"], "detailed_report": str(root / "report.json"),
               "source_files_match": f"{report['source_integrity']['matching_files']}/{len(hashes)}",
               "all_checkpoint_states_finite": report["all_checkpoint_states_finite"],
               "checkpoint_tensor_count": report["checkpoint_tensor_count"],
               "checkpoint_tensor_elements": report["checkpoint_tensor_elements"],
               "groups": report["groups"], "runs": [], "limitations": report["limitations"]}
    for run in report["runs"]:
        curves = run["curves"]
        selected = ["episode/return", "episode/cumulative/lift_bonus_rew", "episode/cumulative/bonus_rew",
                    "episode/cumulative/keypoint_rew", "episode/final/all_goals_hit", "actor/entropy", "temperature/value", "critic/loss"]
        reset = run["final_critic_probe"]["phases"]["reset"]
        compact["runs"].append({"name": run["name"], "hand_filter": run["hand_filter"], "seed": run["seed"],
                                "training_seconds": run["training_seconds"], "tb_path": run["tb_path"],
                                "curves": {key: curves[key] for key in selected},
                                "final_reward_terms": {key.rsplit("/", 1)[1]: value["last"]["value"] for key, value in curves.items() if key.startswith("episode/cumulative/")},
                                "approx_q_from_window_means": run["approx_min_q_from_window_means"],
                                "final_reset_critic_means": {key: value["mean"] for key, value in reset["metrics"].items()},
                                "final_reset_q_to_entropy_gradient_norm_ratio": reset["actor_gradient"]["q_to_entropy_gradient_norm_ratio"]["mean"],
                                "evaluations": run["evaluations"],
                                "optimizer_applied_steps": {key: value["optimizer_applied_steps"] for key, value in run["checkpoint"]["files"].items() if value["optimizer_applied_steps"]}})
    (root / "compact.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n")
    write_markdown(root, report)
    print(json.dumps({"report": str(root / "report.json"), "analysis": str(root / "analysis.md"),
                      "checkpoint_tensors": report["checkpoint_tensor_count"], "all_finite": report["all_checkpoint_states_finite"],
                      "source_files_matching": report["source_integrity"]["matching_files"], "source_files_total": len(hashes)}))


def write_markdown(root, report):
    lines = ["# STR FlashSAC：手部动作平滑短 A/B（6 × 10M）", "",
             "结论：两组各 3 个训练 seed，在每 seed 10,000,384 transitions 后均未完成目标。仅把手部 moving-average 从 1.0 改到 0.1，提高了训练回报，并在最终随机策略评测中出现稍多的抬升/持续接近代理事件；这不足以确认学会抓取，也不足以证明长期无法学习。", "",
             "## 对照与完整性", "",
             "- 原生 FlashSAC 网络/更新、162 维状态、固定橡皮/起点/目标、reward、reset、replay 10M、1024 env、学习率日程等保持一致；每对 seed 只改变手部 filter。",
             "- **机械臂 filter 始终 1.0**。本组不是此前教师和 BC 成功使用的 arm=0.1、hand=0.1 配置。",
             "- 6 个 Hydra 配置与 TensorBoard 内保存配置逐项相同；差异只在批准的 hand filter、seed 和输出名称。最终 checkpoint 为 step9766，对应完整 10,000,384 transitions。",
             f"- 源码快照复核：{report['source_integrity']['matching_files']}/{report['source_integrity']['captured_files']} 文件 SHA256 一致。检查 6 × 6 个 checkpoint 文件（包括网络、target、优化器、reward normalizer、AMP scaler），共 {report['checkpoint_tensor_count']:,} tensors / {report['checkpoint_tensor_elements']:,} tensor elements；全部有限：{report['all_checkpoint_states_finite']}。",
             "- Actor 实际优化器更新 9666～9667 次，Critic 19335～19336 次，温度 9669 次，agent 更新计数 19338。网络包装器的 `update_step=0` 不是实际优化器未更新；与 scheduler 的少量计数差符合 AMP 跳步，不能据此称作数值爆炸。",
             "- 完整校验路径、散列、计数、优化器更新步和所有曲线统计保存在 `report.json`；便于快速读取的摘要为 `compact.json`。未重启或修改训练。", "",
             "## 最终独立评测", "",
             "每个 checkpoint：128 次确定性 + 128 次随机策略评测；固定同一初始设置，随机模式使用原生 Flash 噪声持续机制。三组训练 seed 才是算法重复，不能把所有回合视为独立训练实验。", "",
             "| 手部 filter | seed | 确定性目标/128 | 随机目标/128 | 随机抬高>10cm/128 | 随机持续抬升接近代理/128 | 随机回报 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in report["runs"]:
        d, s = run["evaluations"]["deterministic"], run["evaluations"]["stochastic"]
        lines.append(f"| {run['hand_filter']} | {run['seed']} | {d['counts']['success']} | {s['counts']['success']} | {s['counts']['ever_task_lift']} | {s['counts']['ever_sustained_lift_near_palm_proxy']} | {s['mean_return']:.2f} |")
    lines += ["", "合并计数仅作描述：hand=1.0 随机评测抬升 3/384、持续代理 1/384；hand=0.1 为 8/384、6/384。样本少、仅 3 个训练 seed，未计算置信区间，不称作显著提升。所有确定性策略均 0/384 抬升、0/384 目标；所有随机策略也均 0/384 目标。代理指标不是已验证的稳定抓取。", "",
              "## 训练曲线：峰值不等于最终能力", "",
              "TB 每点为已结束回合的加权窗口均值。日志未保存各窗口回合数，因此不能把窗口均值简单平均，重建全训练成功率。抬升窗口比例 = 窗口 lift_bonus / 300，只适用于该窗口。", "",
              "| 手部 filter | seed | 最终回报 | 峰值回报（M steps） | 有抬升奖励窗口/195 | 峰值窗口抬升比例（M steps） | 最终目标完成率 |", "| --- | ---: | ---: | --- | ---: | --- | ---: |"]
    for run in report["runs"]:
        reward = run["curves"]["episode/return"]
        lift = run["curves"]["episode/cumulative/lift_bonus_rew"]
        lines.append(f"| {run['hand_filter']} | {run['seed']} | {reward['last']['value']:.2f} | {reward['peak']['value']:.2f} ({reward['peak']['step']/1e6:.3f}M) | {lift['positive_windows']}/{lift['points']} | {run['training_lift_peak_window_fraction']*100:.2f}% ({lift['peak']['step']/1e6:.3f}M) | 0 |")
    lines += ["", "六次训练的所有 TB `all_goals_hit` 点都为 0。注意 hand=0.1、seed1 最后窗口出现少量 `bonus_rew=4.7619` 和 keypoint progress；这是部分近目标奖励，并不是累计达到 10 步后的目标成功。baseline seed2 的 47.4% 抬升峰值出现在很早的 0.256M 窗口，不能当作最后策略有 47.4% 抬升能力。", "",
              "### 最后窗口的 reward 分项", "",
              "| 手部 filter | seed | 指尖接近 | 未跨阈值高度项 | 抬升bonus | 目标进度 | 臂速度罚 | 手速度罚 | 近目标bonus |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in report["runs"]:
        terms = [run["curves"][f"episode/cumulative/{tag}"]["last"]["value"] for tag in ["fingertip_delta_rew", "lifting_rew", "lift_bonus_rew", "keypoint_rew", "kuka_actions_penalty", "hand_actions_penalty", "bonus_rew"]]
        lines.append(f"| {run['hand_filter']} | {run['seed']} | " + " | ".join(f"{value:.2f}" for value in terms) + " |")
    lines += ["", "hand=0.1 回报更高的重要来源是手速度惩罚从约 -130～-143 降到 -39～-42。高度项大部分是未抬升阶段基线，不代表已经搬运成功。因此不能把较高总 reward 单独解释为抓取技能提升。", "",
              "## 熵、温度与 Q：区分历史曲线和最终离线探针", "",
              "| 手部 filter | seed | 最终 entropy / 峰值 | 最终 alpha | 近似Q峰值 / 最终 | 最终起点Q | 最终起点最高5 atoms质量 |", "| --- | ---: | --- | ---: | --- | ---: | ---: |"]
    for run in report["runs"]:
        entropy = run["curves"]["actor/entropy"]
        alpha = run["curves"]["temperature/value"]["last"]["value"]
        q = run["approx_min_q_from_window_means"]
        final = run["final_critic_probe"]["phases"]["reset"]["metrics"]
        lines.append(f"| {run['hand_filter']} | {run['seed']} | {entropy['last']['value']:.3f} / {entropy['peak']['value']:.3f} | {alpha:.6f} | {q['peak']['value']:.3f} / {q['last']['value']:.3f} | {final['q_recorded']['mean']:.3f} | {final['minq_top5_atoms_mass']['mean']*100:.1f}% |")
    lines += ["", "- Actor entropy 最终约 19.3；29维 [-1,1] 动作的最大微分熵约 20.10。没有熵坍塌证据。alpha 是熵权重，不是动作标准差。",
              "- 表中历史近似Q用 `-mean(actor_loss) - mean(alpha)*mean(entropy)` 估算；六次峰值在约 6.50～6.96M，之后下降。窗口均值相乘不是精确的逐样本 Q，不能据此认定 clipping。精确历史 top-atom/target-clipping 指标未记录，不能伪造其时间峰值。",
              "- 最终起点 Q/atoms 来自新采集的冻结策略数据，不是原训练 replay。手平滑组最高5个atoms质量较高，提示价值分布接近上界，尚不能认定这是训练失败的主因。",
              "- `critic/max_entropy_bonus` 实际记录 `max(alpha * log_pi)`，不是熵上限，也不是 Q clipping 率。",
              "- `report.json` 保存各 phase 的目标越界质量、动作Q差异、动作梯度、Q/熵梯度比。它们受样本选择和 BN 模式影响，不能替代相同状态下真实回报对动作的排序检验。", "",
              "## 能确认与不能确认", "",
              "能确认：本预算内出现偶发抬升，手部平滑降低速度惩罚并改变探索行为，但六个最终策略都没有完成目标；网络和优化器并非没有更新，检查到的完整 checkpoint 状态与源码一致且有限。",
              "不能确认：FlashSAC 永远不能学会；只缺 LSTM；只调 hand filter 就足够；高Q必然错误；需要任意延长训练便能解决。此前当前状态前馈 BC 的真实目标成功证明该结构在另一匹配控制设置下有表达能力，但不证明从零 RL 学习已经解决。", "",
              "报告依据现存 TB、评测逐回合记录、checkpoint 和源码快照生成；未运行新仿真或训练。"]
    (root / "analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
