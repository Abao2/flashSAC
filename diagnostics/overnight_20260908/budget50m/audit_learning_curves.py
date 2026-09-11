"""CPU evidence audit of completed 50M TB windows, evaluation and LR counters."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
from summarize_str_control_ab import sha256


def contiguous_positive_runs(samples):
    runs, start, last, count = [], None, None, 0
    for item in samples:
        if item["value"] > 0:
            if start is None:
                start = item["step"]
            last, count = item["step"], count + 1
        elif start is not None:
            runs.append({"first_logged_transition": start, "last_logged_transition": last, "positive_windows": count})
            start, count = None, 0
    if start is not None:
        runs.append({"first_logged_transition": start, "last_logged_transition": last, "positive_windows": count})
    return runs


def main():
    torch.set_num_threads(2)
    source = json.loads((ROOT / "summary.json").read_text())
    assert source["queue_status"] == "completed"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "summary_sha256": sha256(ROOT / "summary.json"),
              "conditions": {}, "notes": ["Online success is a logged episode-final task flag, not independent checkpoint success.",
                  "A positive-window count or unweighted mean is not an overall training episode success rate.",
                  "Reward shaping accumulation and true goal bonus are separate; undiscounted TB return is not the SAC objective.",
                  "Both50M conditions completed; comparisons remain one seed and one fixed task.",
                  "Training vs clean-reset evaluation goal mismatch remains unresolved. No simulator/GPU or model update was run for this audit."]}
    components = ["fingertip_delta_rew", "lifting_rew", "lift_bonus_rew", "keypoint_rew", "kuka_actions_penalty", "hand_actions_penalty", "bonus_rew"]
    for name, condition in source["conditions"].items():
        assert len(condition["tensorboard_runs"]) == 1
        curves = condition["tensorboard_runs"][0]["curves"]
        indexed = {tag: {sample["step"]: sample["value"] for sample in value["samples"]}
                   for tag, value in curves.items() if "samples" in value}
        goals = curves["episode/final/all_goals_hit"]["samples"]
        assert indexed["episode/final/all_goals_hit"] == indexed["episode/final/done_max_successes"]
        reward_errors = [abs(sum(indexed["episode/cumulative/" + key][step] for key in components) - total)
                         for step, total in indexed["episode/return"].items()]
        relative_reward_errors = [error / max(1., abs(total)) for error, total in zip(reward_errors, indexed["episode/return"].values())]
        assert max(reward_errors) < .01 and max(relative_reward_errors) < 2e-5
        event_steps = {"final": goals[-1]["step"],
                       "peak_return": curves["episode/return"]["peak"]["step"],
                       "peak_goal": curves["episode/final/all_goals_hit"]["peak"]["step"],
                       "peak_lift_bonus": curves["episode/cumulative/lift_bonus_rew"]["peak"]["step"]}
        for label, threshold in (("first_positive_goal", 1e-12), ("first_50pct_goal_window", .5), ("first_80pct_goal_window", .8)):
            found = [sample["step"] for sample in goals if sample["value"] >= threshold]
            event_steps[label] = min(found) if found else None
        lifted = [step for step, value in indexed["episode/cumulative/lift_bonus_rew"].items() if value >= 299.999]
        event_steps["first_all_lift_window"] = min(lifted) if lifted else None
        records = {label: {"transitions": step, "values": {tag: values.get(step) for tag, values in indexed.items()}}
                   for label, step in event_steps.items() if step is not None}
        checkpoints = {}
        for step, milestone in condition["milestones"].items():
            item = {"transitions": milestone["transitions"], "network_calls": milestone["agent_network_calls"],
                    "optimizer_applied_steps": milestone["actual_optimizer_steps"], "schedulers": {}, "evaluations": {}}
            for network in ("actor", "critic", "temperature"):
                model = torch.load(Path(milestone["path"]) / (network + ".pt"), map_location="cpu", weights_only=True)
                item["schedulers"][network] = {"last_epoch": model["scheduler_state_dict"]["last_epoch"],
                    "actual_learning_rates": [group["lr"] for group in model["optimizer_state_dict"]["param_groups"]]}
            for mode, evaluation in milestone["evaluations"].items():
                assert evaluation["complete_budget"] and evaluation["model_unchanged"]
                raw = json.loads(Path(evaluation["path"]).read_text())
                item["evaluations"][mode] = {"completed": evaluation["episodes"], "counts": evaluation["counts"],
                    "mean_return": evaluation["mean_return"], "mean_length": evaluation["mean_length"], "reward_components": evaluation["reward_components"],
                    **{key: raw[key]["mean"] for key in ("discounted_raw_return", "position_error_m", "keypoint_error_m", "max_object_center_height_above_reset_m")}}
            checkpoints[step] = item
        report["conditions"][name] = {"hand_filter": condition["control"]["hand_moving_average"],
            "logged_windows": len(goals), "goal_positive_windows": sum(s["value"] > 0 for s in goals),
            "positive_goal_runs": contiguous_positive_runs(goals), "event_steps": event_steps, "event_records": records,
            "reward_components_reconstruct_return_max_abs_error": max(reward_errors),
            "reward_components_reconstruct_return_max_relative_error": max(relative_reward_errors),
            "goal_matches_done_max_successes_each_window": True, "final_checkpoint": checkpoints["48830"], "checkpoints": checkpoints,
            "alpha_curve": {key: curves["temperature/value"][key] for key in ("first", "last", "minimum", "peak")},
            "entropy_curve": {key: curves["actor/entropy"][key] for key in ("first", "last", "minimum", "peak")}}
    (ROOT / "learning_curve_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_markdown(report)
    print(json.dumps({name: {"positive_goal_windows": value["goal_positive_windows"], "first_goal": value["event_steps"]["first_positive_goal"],
                     "final_eval_goals": {mode: result["counts"]["success"] for mode, result in value["final_checkpoint"]["evaluations"].items()}}
                     for name, value in report["conditions"].items()}))


def write_markdown(report):
    lines = ["# 50M学习曲线独立核对：reward不等于goal成功", "",
             "**hand=.1的50M独立评测中全部回合曾跨过10cm抬升阈值，但多数随后掉落；20项clean-reset checkpoint评测仍全部0 goal。其训练日志晚期出现80%左右goal窗口，与独立评测显著不一致，不能据TB宣称已获得可靠目标策略。** hand=1.0则主要提高了阈值以下的高度shaping。", "",
             "## 1. 同样约1200的训练reward，含义截然不同", "",
             "| 最终训练窗口分量 | hand1.0 | hand0.1 |", "| --- | ---: | ---: |"]
    a, b = [report["conditions"][name] for name in ("arm1_hand1", "arm1_hand01")]
    tags = ["episode/return", "episode/cumulative/fingertip_delta_rew", "episode/cumulative/lifting_rew", "episode/cumulative/lift_bonus_rew",
            "episode/cumulative/keypoint_rew", "episode/cumulative/kuka_actions_penalty", "episode/cumulative/hand_actions_penalty",
            "episode/cumulative/bonus_rew", "episode/final/all_goals_hit", "episode/length", "episode/final/done_timeout"]
    for tag in tags:
        lines.append(f"| {tag} | {a['event_records']['final']['values'][tag]:.6f} | {b['event_records']['final']['values'][tag]:.6f} |")
    lines += ["", "逐窗口验证：7个reward分量之和与return最大差约.0047，最大相对差约1.2e−5，符合不同求和顺序/FP32累加误差的量级；all_goals_hit与done_max_successes每个窗口完全一致。", "",
              "官方当前reward代码在抬升前每步给 `20*clamp(.05+Δz,0,.5)`，跨过`.05+Δz>.15`即真实升高10cm后才给一次300，并关闭这项dense shaping。因此lifting_rew大不代表已完成任务抬升。", "",
              "hand1.0最终确定性评测平均最高升高约5.12cm，仍低于10cm阈值；其rawreturn1169.67中1161.89来自这一dense项，lift/goal bonus均0。说它完全没移动物体也不准确：它学到的是阈值以下的抬高/维持，而不是已完成抬升搬运。", "",
              "这些rawreturn是未折扣和，episode长度也不同。不能据长时间hover的rawreturn更高，直接宣布它是SAC最优策略；SAC用γ=.99、reward normalization及熵项。", "",
              "## 2. 独立评测显示的实际阶段", "",
              "| hand | 预算M | det goal；lift | stoch goal；lift | det rawreturn | stoch rawreturn |", "| --- | ---: | --- | --- | ---: | ---: |"]
    for condition in (a, b):
        for checkpoint in condition["checkpoints"].values():
            det, stoch = [checkpoint["evaluations"][mode] for mode in ("deterministic", "stochastic")]
            lines.append(f"| {condition['hand_filter']} | {checkpoint['transitions']/1e6:.3f} | {det['counts']['success']}/{det['completed']}; {det['counts']['ever_task_lift']}/{det['completed']} | {stoch['counts']['success']}/{stoch['completed']}; {stoch['counts']['ever_task_lift']}/{stoch['completed']} | {det['mean_return']:.2f} | {stoch['mean_return']:.2f} |")
    lines += ["", "hand=.1在20M/30M/50M的det与stoch评测中抬升均100%；持握近掌部代理几乎/全部100%，但全部未达goal并主要等到600步。40M曾丢失抬升，之后恢复，不能把过程称为单调稳定收敛。", "",
              "50M hand=.1最高升高约13.1cm，最终位置误差约17.2cm；det rawreturn333.13中有300抬升奖、0goal奖，stoch相同结论。随后物体低于初始高度的回合为det55/64、stoch127/128；reset_when_dropped=false允许它们继续到timeout。曾满足持握代理不是持续抓稳、接触确认或真机部署证明。", "",
              "折扣rawreturn还揭示另一点：50M det hand1≈206.10，hand.1≈292.36；这与未折扣rawreturn1169.67对333.13的排序相反。比较奖励时必须注明折扣/长度，不能只盯一条TB return。", "",
              "## 3. 训练中确有goal标志，但暂不等于checkpoint可复现", "",
              f"hand=.1在{b['event_steps']['first_all_lift_window']/1e6:.4f}M第一次出现窗口平均lift bonus300；在{b['event_steps']['first_positive_goal']/1e6:.4f}M第一次出现正goal窗口。976个日志窗口中只有{b['goal_positive_windows']}个goal>0；这是窗口数，不是训练episode总成功率。", "",
              f"首次≥50%的goal窗口到{b['event_steps']['first_50pct_goal_window']/1e6:.4f}M才出现；最高84.94%在49.8688M，最后窗口80.21%。同时goal bonus与done_max_successes支持“训练日志确实标了成功”，并非只有抬升奖。", "",
              "但clean-reset评测所有保存点goal均0，包括50M。这是当前真正待解释的train/eval差异。可能涉及同步/异步重置、初始随机progress/首个随机action、1024训练env对32评测env、训练中的持续策略/BN更新与冻结checkpoint、仿真数值或数据分布；这里只列排查候选，未确认其中任何一个为原因。", "",
              "特别是原生随机progress只在初次reset注入，不能直接用它解释50M末尾的所有差异；需要匹配协议实测。未做这项验证前，不应该宣布目标任务已可靠训成，也不应该忽略已验证的抬升技能。", "",
              "## 4. 熵、alpha和学习率预算没有混算", "",
              "| hand | 预算M | network calls | actor/critic/temp applied steps | actor LR | critic LR | actor scheduler step |", "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for condition in (a, b):
        for checkpoint in condition["checkpoints"].values():
            counts = "/".join(str(checkpoint["optimizer_applied_steps"][name][0]) for name in ("actor", "critic", "temperature"))
            sch = checkpoint["schedulers"]
            lines.append(f"| {condition['hand_filter']} | {checkpoint['transitions']/1e6:.3f} | {checkpoint['network_calls']} | {counts} | {sch['actor']['actual_learning_rates'][0]:.9f} | {sch['critic']['actual_learning_rates'][0]:.9f} | {sch['actor']['last_epoch']} |")
    lines += ["", "LR decay_steps固定19532，但每个optimizer用自身scheduler：10M actor仍约2.2617e−4，critic已近1.5e−4；20M actor也接近end，30M起两者均1.5e−4。不是50M重启新日程，更不是10M后停止更新。AMP跳步导致实际optimizer计数略小于scheduler/attempted数，均从模型直接读取。", "",
              f"hand1最终H={a['entropy_curve']['last']['value']:.4f}，alpha={a['alpha_curve']['last']['value']:.8g}；hand.1最终H={b['entropy_curve']['last']['value']:.4f}，alpha={b['alpha_curve']['last']['value']:.8g}。hand.1首次成功窗口H≈−22.41、alpha≈2.14e−5；后期alpha在48.8448M达到最低约1.82e−5后略回升，与H落到native目标约−13.87以下的温度调节方向一致。", "",
              "这给“高熵早期难形成技能、降低熵后出现动作结构”的假设提供时间相关证据，但不是因果隔离。当前另排的alpha-init配对实验才能进一步检验；本次不能把先后发生说成唯一原因。", "",
              "## 结论边界", "", *[f"- {note}" for note in report["notes"]], "",
              "完整JSON保留所有事件窗口、连续正goal区间、每个checkpoint actual optimizer/scheduler/LR、每次评测reward分量及误差。没有修改reward、训练或评测配置。"]
    (ROOT / "learning_curve_audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
