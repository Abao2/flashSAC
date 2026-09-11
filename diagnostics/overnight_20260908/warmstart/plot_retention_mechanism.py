"""Static retention figure plus bounded CPU, recorded-trajectory entropy audit."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ALPHAS = [0, .000001, .0001, .001, .01]
sys.path.insert(0, str(ROOT.parents[2]))
from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.agents.utils.distribution import safe_tanh_log_det_jacobian
from audit_policy_distribution import digest, stats


def retention_figure(report):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    points = []
    colors = {"immediate": "#c85a17", "warmup2000": "#17668c"}
    for ax, sampling in zip(axes, ("deterministic", "stochastic")):
        for name, branch in report["branches"].items():
            rows = []
            if sampling == "stochastic":
                baseline = report["baselines"]["selected_seed0_native"]
                rows.append({"actor_steps": 0, "rate": baseline["counts"]["success"] / baseline["episodes"],
                             "source": "shared_selected_checkpoint_baseline", "episodes": baseline["episodes"]})
            for label, evaluation in branch["evaluations"].items():
                if not label.endswith("_" + sampling) or evaluation["status"] == "pending":
                    continue
                checkpoint = label.removeprefix(name + "_").removesuffix("_" + sampling)
                item = branch["snapshots"][checkpoint]
                counts = item["actual_optimizer_steps"]["actor"] or [0]
                assert len(counts) == 1
                rows.append({"actor_steps": counts[0], "rate": evaluation["counts"]["success"] / evaluation["episodes"],
                             "source": checkpoint, "episodes": evaluation["episodes"], "network_calls": item["agent_network_calls"]})
            points.extend({"branch": name, "sampling": sampling, **row} for row in rows)
            unique = {}
            for row in rows:
                previous = unique.get(row["actor_steps"])
                assert previous is None or previous["rate"] == row["rate"], "Cannot silently combine unequal zero-update results"
                unique[row["actor_steps"]] = row
            rows = sorted(unique.values(), key=lambda row: row["actor_steps"])
            ax.plot([r["actor_steps"] for r in rows], [100 * r["rate"] for r in rows],
                    color=colors[name], marker="o" if name == "immediate" else "s", markersize=6,
                    label="Immediate SAC" if name == "immediate" else "Critic-only first 2,000 calls", linewidth=1.7, linestyle="--")
        ax.set_xscale("symlog", linthresh=5)
        ax.set_xticks([0, 5, 50, 1000, 10000], labels=["0", "5", "50", "1,000", "10,000"])
        ax.set_xlim(-.3, 12500)
        ax.set_ylim(-5, 106)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(sampling.capitalize() + (" (64 episodes)" if sampling == "deterministic" else " (128 episodes)"))
        ax.set_xlabel("Actual actor optimizer steps (symlog)")
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Task goal success (%)")
    axes[0].annotate("0/64 by actor step 50", (50, 0), xytext=(12, 25), fontsize=9,
                     arrowprops={"arrowstyle": "->", "color": colors["immediate"]})
    axes[1].annotate("95/128 after 5 steps", (5, 95 / 128 * 100), xytext=(13, 90), fontsize=9,
                     arrowprops={"arrowstyle": "->", "color": colors["immediate"]})
    fig.suptitle("A working BC policy loses its skill after native SAC updates", fontsize=14, y=.98)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(.5, .105), frameon=False)
    fig.text(.5, .055, "Fixed STR task; arm/hand filter = 0.1; seed 0. Dashed lines are guides, not measured intermediate success.", ha="center", fontsize=8.7)
    fig.text(.5, .018, "Warmup calls 10 / 100 / 1,000 / 2,000 overlap at ZERO actor updates. Its loss onset between 0 and 999 updates is unresolved.", ha="center", fontsize=8.4)
    fig.tight_layout(rect=(0, .16, 1, .93))
    fig.savefig(ROOT / "retention_actor_steps.png", dpi=170)
    fig.savefig(ROOT / "retention_actor_steps.svg")
    plt.close(fig)
    return points


def finite_trajectory_scale(retention):
    """Evaluate conditional entropy along recorded successful BC state sequences.

    This is a frozen-data surrogate, NOT fresh on-policy Monte Carlo Q calibration.
    The native Bellman convention places entropy at successor, not initial state.
    """
    path = ROOT / "frozen_bc_std005_eigen/transitions.npz"
    checkpoint = ROOT / "bc_std005_eigen/actor.pt"
    baseline = json.loads((path.parent / "summary.json").read_text())
    assert baseline["episodes_completed"] == 128 and baseline["success"]["mean"] == 1.0
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in ("obs", "rewards", "terminated", "truncated", "episode_id", "step_in_episode")}
    actor = FlashSACActor(num_blocks=2, input_dim=162, hidden_dim=128, action_dim=29)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)["network_state_dict"]
    actor.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in saved.items()})
    actor.eval()
    original = {key: value.clone() for key, value in actor.state_dict().items()}
    obs = torch.from_numpy(data["obs"].astype(np.float32))
    with torch.no_grad():
        mu, sigma = actor.get_mean_and_std(obs, training=False)
        generator = torch.Generator().manual_seed(7109)
        noise = torch.randn((8, len(obs), 29), generator=generator)
        u = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
        gaussian_h = (sigma.double().log() + .5 * math.log(2 * math.pi * math.e)).sum(-1)
        entropy = gaussian_h.unsqueeze(0) + safe_tanh_log_det_jacobian(u.double()).sum(-1)
    assert all(torch.equal(value, original[key]) for key, value in actor.state_dict().items())
    entropy = entropy.numpy()
    denominators = {"immediate_update0": retention["branches"]["immediate"]["snapshots"]["update0"]["reward_normalizer"]["reward_divisor"],
                    "warmup_update2000": retention["branches"]["warmup2000"]["snapshots"]["update2000"]["reward_normalizer"]["reward_divisor"]}
    rows = []
    for episode in np.unique(data["episode_id"]):
        indices = np.flatnonzero(data["episode_id"] == episode)
        indices = indices[np.argsort(data["step_in_episode"][indices])]
        assert np.array_equal(data["step_in_episode"][indices], np.arange(len(indices)))
        assert data["terminated"][indices[-1]] and not data["terminated"][indices[:-1]].any() and not data["truncated"][indices].any()
        discount = .99 ** np.arange(len(indices))
        raw = float(np.sum(discount * data["rewards"][indices]))
        # Q(s0,a0) excludes entropy at s0 and has no successor beyond terminal.
        entropy_sum = (entropy[:, indices[1:]] * discount[None, 1:]).sum(-1)
        rows.append({"episode": int(episode), "length": len(indices), "discounted_raw_reward": raw,
                     "discounted_future_entropy_sum": float(entropy_sum.mean()),
                     "entropy_sum_mc_se_fixed_trajectory": float(entropy_sum.std(ddof=1) / math.sqrt(8)),
                     "normalized_reward": {key: raw / value for key, value in denominators.items()},
                     "soft_sum": {key: {str(alpha): raw / value + alpha * float(entropy_sum.mean()) for alpha in ALPHAS}
                                  for key, value in denominators.items()}})
    return {"dataset": str(path), "dataset_sha256": digest(path), "actor_sha256": digest(checkpoint),
            "episodes": len(rows), "mc_draws": 8, "gamma": .99, "reward_denominators": denominators,
            "definition": "sum_t gamma^t r_t / D + alpha sum_(k=1..T-1) gamma^k H_pi(a|s_k), no entropy at s0, all terminal goal episodes; frozen D and alpha",
            "length": stats([row["length"] for row in rows]),
            "discounted_raw_reward": stats([row["discounted_raw_reward"] for row in rows]),
            "future_entropy_sum": stats([row["discounted_future_entropy_sum"] for row in rows]),
            "soft_sum": {key: {str(alpha): stats([row["soft_sum"][key][str(alpha)] for row in rows]) for alpha in ALPHAS}
                         for key in denominators}, "episode_values": rows,
            "limitations": ["Recorded successful state occupancy, not an alternative-policy comparison or claimed optimum.",
                "Behavior used cached/repeated noise; conditional entropy is the native actor marginal, not exact conditional density given the cached-noise history.",
                "Inference BN and frozen reward divisor/alpha; actual training uses changing statistics and batch composition.",
                "MC uncertainty only integrates action noise at fixed recorded states; it is not a confidence interval over tasks/seeds.",
                "Negative soft sums outside support describe this frozen diagnostic surrogate; they are not calibrated historical Q values."]}


def main():
    torch.set_num_threads(2)
    retention = json.loads((ROOT / "retention_summary.json").read_text())
    distribution = json.loads((ROOT / "policy_distribution_audit.json").read_text())
    points = retention_figure(retention)
    scale = finite_trajectory_scale(retention)
    critics = {}
    wanted = ["q_recorded", "minq_bottom5_atoms_mass", "target_unprojected_mean", "target_clipped_mean",
              "target_mass_below_support", "target_mass_above_support", "next_entropy_bonus", "uniform_action_q_range"]
    for path in sorted((ROOT / "retention").glob("*/critics/*.json")):
        data = json.loads(path.read_text())
        critics[str(path.relative_to(ROOT))] = {"sha256": digest(path), "alpha": data["alpha"], "reward_divisor": data["reward_denominator"],
            "phases": {key: {"samples": value["sampled_transitions"], "raw_reward_mean": value["raw_reward"]["mean"],
                             **{metric: value["metrics"][metric]["mean"] for metric in wanted},
                             "q_to_entropy_gradient_norm_ratio": value["actor_gradient"]["q_to_entropy_gradient_norm_ratio"]["mean"]}
                       for key, value in data["phases"].items()}}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "retention_source_sha256": digest(ROOT / "retention_summary.json"),
              "figure_points": points, "recorded_trajectory_scale": scale, "critic_reports": critics,
              "selected_bc_matched_expert_entropy": distribution["policies"]["bc_std005_eigen"]["groups"]["all_selected"]["conditional_entropy_nats"]["mean"]}
    (ROOT / "retention_mechanism_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_markdown(result)
    print(json.dumps({"figure_points": len(points), "episodes": scale["episodes"], "mean_length": scale["length"]["mean"],
                      "mean_discounted_raw_reward": scale["discounted_raw_reward"]["mean"], "mean_future_entropy_sum": scale["future_entropy_sum"]["mean"],
                      "mean_soft_sums": {key: {alpha: value["mean"] for alpha, value in item.items()} for key, item in scale["soft_sum"].items()}}))


def write_markdown(result):
    scale = result["recorded_trajectory_scale"]
    lines = ["# 技能保留失败：熵项、reward尺度与categorical critic的证据", "",
             "本报告没有新训练。它区分已测现象、固定数据上的诊断与下一步假设。**下面的轨迹soft sum不是已校准SAC Q，也不是策略优劣/最优性的证明。**", "",
             "## 1. 更新后发生了什么", "",
             "立即更新分支：5次actor实际优化后，det目标成功64/64，stochastic95/128；50次actor优化后，两者均0成功。先只训critic分支：network calls10/100/1000/2000期间actor实际更新均为0，det64/64、stochastic128/128一直保留；恢复更新后，到999次实际actor优化的首个后续采样点，成功率已为0。", "",
             "warmup分支没有在恢复后的5或50次actor更新处评测，因此**不能据图声称warmup延迟了遗忘到999次**：其丢失时刻只定位在0～999次之间。零actor更新的四个点重叠，不应画成四个递增actor步数。", "",
             "![Actual actor optimizer steps versus task success](" + str(ROOT / "retention_actor_steps.png") + ")", "",
             "## 2. 成功轨迹的reward和熵项：逐步折扣，不是63×平均熵", "",
             f"另取selected BC自己的128条成功rollout，平均{scale['length']['mean']:.3f}步。冻结actor、normalizer与alpha，逐个状态重新计算条件微分熵：解析Gaussian熵 + 每状态8次MC的tanh log-Jacobian。", "",
             "原生一步target为 `r/D + γ(1−terminated)[Z_next − α logπ(a_next|s_next)]`。因此我们定义固定已记录轨迹的诊断量：", "",
             "`S0 = Σ(t=0..T−1) γ^t r_t/D + α Σ(k=1..T−1) γ^k Hπ(a|s_k)`，γ=.99。", "",
             "不加起点s0的熵，不在真实goal terminal后补尾项；所有被选回合均以真实终止结束，没有把timeout强行截成terminal。熵随每个状态变化，且每个回合长度分别计算。", "",
             f"平均折扣raw task reward为{scale['discounted_raw_reward']['mean']:.6f}；平均折扣future-entropy sum为{scale['future_entropy_sum']['mean']:.6f} nats。下面每一列使用一个固定checkpoint的reward除数，不混用训练中的动态除数。", "",
             "| 固定alpha | S0：D=immediate update0 | S0：D=warmup update2000 |", "| ---: | ---: | ---: |"]
    for alpha in ALPHAS:
        lines.append(f"| {alpha:g} | {scale['soft_sum']['immediate_update0'][str(alpha)]['mean']:.6f} | {scale['soft_sum']['warmup_update2000'][str(alpha)]['mean']:.6f} |")
    lines += ["", f"两个D分别为{scale['reward_denominators']['immediate_update0']:.6f}、{scale['reward_denominators']['warmup_update2000']:.6f}。alpha=0行只是相同轨迹的normalized task reward，不是另一条已训练策略。", "",
              "重要边界：行为rollout使用缓存/重复噪声，而这里Hπ是网络给定状态的单步边际条件熵，不是给定完整噪声历史的真实行为条件密度。固定记录的state occupancy也不等于重新采样整条独立动作策略。这是**贴近原生actor边际定义的冻结轨迹surrogate**，不把它称为精确SAC Q/on-policy MC校准。", "",
              "这个尺度检查提示：在该冻结高密度BC策略下，alpha=.01的surrogate远低于critic支持下限−5；alpha减小可以把它移回当前支持范围。但这不证明存在相应的最终SAC最优策略，也不保证低alpha学习成功。其他策略可以改变均值/方差、成功率、时长和状态分布。", "",
              "## 3. 实际critic报告：并非只靠上面的surrogate猜测", "",
              "warmup update2000时actor还没改，仍能成功，alpha=.01。下面是CPU从该checkpoint重新计算的诊断；不是历史训练replay的clipping频率。", "",
              "| 查询数据/phase | normalized即时reward均值 | Q(recorded action) | 最低5atoms质量 | 未投影target均值 | clamp后target均值 | 下界外概率质量 | αH_next估计 | Q/熵梯度范数比 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind in ("expert", "native"):
        report = result["critic_reports"][f"retention/warmup2000/critics/update2000_{kind}.json"]
        for phase in ("reset", "far_from_object"):
            row = report["phases"][phase]
            lines.append(f"| {kind}/{phase} | {row['raw_reward_mean']/report['reward_divisor']:.6f} | {row['q_recorded']:.6f} | {row['minq_bottom5_atoms_mass']*100:.3f}% | {row['target_unprojected_mean']:.6f} | {row['target_clipped_mean']:.6f} | {row['target_mass_below_support']*100:.3f}% | {row['next_entropy_bonus']:.6f} | {row['q_to_entropy_gradient_norm_ratio']:.7f} |")
    lines += ["", "例如native/reset：即时reward/D约.0033，而下一状态熵项约−1.3955；γ再乘下一状态的Q与熵项。当前Q已靠近−5，target均值约−6.327，被截回约−4.999。这个报告直接测到概率质量挤在下界，不能用“成功reward很高”推断soft Q也应高。", "",
              "梯度是在phase-balanced公共current/next batch上、训练actor BN+推理critic BN、FP32无AMP/优化器/投影下测的。范数比极小显示该诊断batch的actor梯度主要来自熵项；不代表每个真实训练batch都同比例，也不等于Adam后的参数位移比例。", "",
              "专家数据可能有状态分布差异；native数据来自仍成功的冻结BC，与warmup rollout更接近。两者reset/far方向一致，增加了这个机制的可信度，但都不是训练replay抽样。", "",
              "## 4. 目前最有根据的假设与如何反证", "",
              "- 已确认：前馈actor能完成这个固定任务；仅冻结actor/temperature时技能保留；原生更新后技能丢失。不能再把这例失败简单归成网络无法表达或必须LSTM。",
              "- 有证据的机制假设：成功但高密度/低微分熵的BC初始化，与当前熵权重、normalized reward尺度和[-5,5] categorical支持范围组合后，使soft target大量落在下界以下；Q对动作的区分梯度很小，actor更新受熵梯度强烈影响。",
              "- 下一项配对诊断应从同一BC初始化开始，仅改初始alpha，保留native自适应温度、任务和网络；记录实际alpha轨迹、早期actor更新次数、det/stoch成功、Q边界质量及梯度比。降低初始alpha不是“永远固定低熵权重”，因为温度仍会自适应。",
              "- 如果低alpha保住技能并减少下界饱和，支持该组合机制，但仍需区分actor直接熵梯度与critic soft target的作用；若低alpha仍同样遗忘，就需继续查Q学习、BN模式、optimizer/projection及采样分布，而不能追加未经验证的故事。",
              "- 不据此宣称SAC不能解STR；不据此把entropy/reward尺度选择说成唯一原因；不据此要求修改环境reward。", "",
              "JSON保留完整source SHA256、逐回合长度与折扣项、MC标准误、critic来源和图中每一个点的真实optimizer计数。"]
    (ROOT / "retention_mechanism_audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
