"""CPU-only matched-state tanh-Gaussian conditional-distribution audit."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.agents.utils.distribution import safe_tanh_log_det_jacobian


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def stats(value):
    value = np.asarray(value, dtype=float).reshape(-1)
    assert np.isfinite(value).all() and len(value)
    return {"mean": float(value.mean()), "std": float(value.std()), "min": float(value.min()),
            "p10": float(np.quantile(value, .1)), "p50": float(np.median(value)),
            "p90": float(np.quantile(value, .9)), "p99": float(np.quantile(value, .99)), "max": float(value.max())}


def stratified_rows(phase, maximum=2048, seed=0):
    names, counts = np.unique(phase, return_counts=True)
    total = min(maximum, len(phase))
    ideal = counts * total / len(phase)
    quotas = np.floor(ideal).astype(int)
    for index in np.argsort(-(ideal - quotas))[:total - int(quotas.sum())]:
        quotas[index] += 1
    rng = np.random.default_rng(seed)
    rows = np.concatenate([rng.choice(np.flatnonzero(phase == name), size=quota, replace=False)
                           for name, quota in zip(names, quotas) if quota])
    return np.sort(rows)


def main():
    torch.set_num_threads(2)
    dataset_path = ROOT.parent / "official_filter01_labeled128/transitions.npz"
    with np.load(dataset_path, allow_pickle=False) as data:
        phase = data["phase"]
        rows = stratified_rows(phase)
        selected_phase = phase[rows]
        observations = torch.from_numpy(data["obs"][rows].astype(np.float32))
        episode_ids = data["episode_id"][rows]
    paths = {"bc_native_original_std": ROOT / "bc_initial/actor.pt",
             "bc_std005_eigen": ROOT / "bc_std005_eigen/actor.pt",
             "bc_std015_eigen": ROOT / "bc_std015_eigen/actor.pt",
             "old_flash_step9760": REPO / "models/simtoolreal_diagnostics/state162_eraser_onegoal_seed0/Isaacsimenvs-SimToolReal-Direct-v0/seed0-0908-153538/step9760/actor.pt"}
    noise = torch.randn((8, len(rows), 29), generator=torch.Generator().manual_seed(7108))
    target_h = .5 * 29 * math.log(2 * math.pi * math.e * .15 ** 2)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu", "dataset": str(dataset_path),
              "dataset_sha256": digest(dataset_path), "dataset_rows": len(phase), "selected_rows": len(rows),
              "selection": "proportional phase-stratified without replacement, seed0, same rows for all policies",
              "selected_row_indices": rows.tolist(), "selected_episodes": len(np.unique(episode_ids)),
              "selected_phase_counts": {str(name): int((selected_phase == name).sum()) for name in np.unique(selected_phase)},
              "mc_draws_per_state": 8, "noise_seed": 7108,
              "noise_sharing": "8 independent Gaussian vectors per state; common random numbers across policies only",
              "target_entropy_from_native_sigma015_formula": target_h,
              "bounded_action_max_differential_entropy": 29 * math.log(2),
              "entropy_method": "H(a|s)=sum(log(sigma)+0.5*log(2*pi*e)) + MC E sum(native safe_tanh_log_det_jacobian(u)); u=mu+sigma*epsilon",
              "action_std_method": "Within-state standard deviation across 8 tanh samples (sample variance correction=1); mean std is an MC estimate, not across-state variation",
              "numerics": "Network and tanh actions use FP32; native safe log-Jacobian evaluated in FP64 on sampled FP32 pre-tanh values. Extremely saturated FP32 actions may round to exactly +/-1; entropy refers to underlying continuous tanh model, not quantized symbol entropy.",
              "limitations": ["Frozen matched expert states, not a rollout or an optimizer update.",
                              "Expert-state distribution is off-policy/OOD for the failed old Flash checkpoint; this is not its on-policy exploration success or replay entropy.",
                              "Inference BatchNorm is used and never updated; SAC training can use different BN modes and state batches.",
                              "Eight MC draws per state estimate the Jacobian term and action moments; the reported MC SE excludes state-sampling and training uncertainty.",
                              "Differential entropy can be negative because probability density, unlike probability mass, can exceed one.",
                              "alpha*H values are per-step conditional entropy terms only, not total returns or an optimum calculation.",
                              "The target_sigma=.15 reference is converted to a Gaussian entropy target, not a demand that every pre-tanh or tanh action std equals .15.",
                              "Conditional action distribution ignores temporal noise repetition and subsequent controller filtering; it is not a full sequence-entropy or effective-joint-target distribution audit."],
              "policies": {}}
    baseline_mean = None
    for name, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = {key.removeprefix("_orig_mod."): value for key, value in checkpoint["network_state_dict"].items()}
        model = FlashSACActor(num_blocks=2, input_dim=162, hidden_dim=128, action_dim=29)
        model.load_state_dict(state, strict=True)
        model.eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            mu, sigma = model.get_mean_and_std(observations, training=False)
            u = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
            action = u.tanh()
            log_jac = safe_tanh_log_det_jacobian(u.double())
            gaussian_h_dim = sigma.double().log() + .5 * math.log(2 * math.pi * math.e)
            entropy_dim_samples = gaussian_h_dim.unsqueeze(0) + log_jac
            entropy_dim = entropy_dim_samples.mean(0)
            entropy_state = entropy_dim.sum(-1)
            raw_gaussian_logp = -.5 * (noise.double().square() + 2 * sigma.double().log().unsqueeze(0) + math.log(2 * math.pi))
            native_mc_neglogp = -(raw_gaussian_logp - log_jac).sum(-1).mean(0)
            action_std = action.std(dim=0, correction=1)
            action_edge = (action.abs() > .99).double().mean(0)
            threshold = math.atanh(.99)
            edge_exact = .5 * torch.erfc((threshold - mu.double()) / (sigma.double() * math.sqrt(2)))
            edge_exact += .5 * torch.erfc((threshold + mu.double()) / (sigma.double() * math.sqrt(2)))
            deterministic = mu.tanh()
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
        assert all(torch.isfinite(tensor).all() for tensor in [mu, sigma, action, log_jac, entropy_state])
        if baseline_mean is None:
            baseline_mean = mu.clone()
        mean_difference = float((mu - baseline_mean).abs().max())
        if name.startswith("bc_"):
            assert mean_difference == 0, "Std calibration changed deterministic policy mean"
        policy = {"checkpoint": str(path), "checkpoint_sha256": digest(path), "network_parameters_and_buffers_unchanged": True,
                  "mean_head_max_difference_from_bc_native_on_matched_states": mean_difference, "groups": {}}
        masks = {"all_selected": np.ones(len(rows), bool), **{str(label): selected_phase == label for label in np.unique(selected_phase)}}
        for label, mask_np in masks.items():
            mask = torch.from_numpy(mask_np)
            entropies = entropy_state[mask].numpy()
            group = {"states": int(mask.sum()), "conditional_entropy_nats": stats(entropies),
                     "gaussian_entropy_nats": stats(gaussian_h_dim[mask].sum(-1).numpy()),
                     "tanh_log_jacobian_term_nats": stats(log_jac[:, mask].sum(-1).mean(0).numpy()),
                     "native_neglogpi_mc_nats": stats(native_mc_neglogp[mask].numpy()),
                     "entropy_below_target_state_fraction": float((entropies < target_h).mean()),
                     "per_step_alpha_H": {str(alpha): float(alpha * entropies.mean()) for alpha in [.01, .001]},
                     "entropy_mc_standard_error_fixed_states": float(entropy_dim_samples[:, mask].sum(-1).mean(-1).std(correction=1) / math.sqrt(8)),
                     "parts": {}}
            for part, indices in [("all_actions", slice(None)), ("arm7", slice(0, 7)), ("hand22", slice(7, 29))]:
                group["parts"][part] = {
                    "pretanh_abs_mean": stats(mu[mask, indices].abs().numpy()),
                    "pretanh_sigma": stats(sigma[mask, indices].numpy()),
                    "conditional_tanh_action_std": stats(action_std[mask, indices].numpy()),
                    "conditional_entropy_sum_mean_nats": float(entropy_dim[mask, indices].sum(-1).mean()),
                    "sampled_action_abs_mean": float(action[:, mask, indices].abs().mean()),
                    "sampled_abs_action_gt099_fraction_mc": float(action_edge[mask, indices].mean()),
                    "abs_action_gt099_probability_analytic": float(edge_exact[mask, indices].mean()),
                    "deterministic_action_abs_mean": float(deterministic[mask, indices].abs().mean()),
                    "deterministic_abs_action_gt099_fraction": float((deterministic[mask, indices].abs() > .99).double().mean())}
            policy["groups"][label] = group
        report["policies"][name] = policy
    (ROOT / "policy_distribution_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_markdown(report)
    print(json.dumps({"report": str(ROOT / "policy_distribution_audit.json"), "states": len(rows),
                      "policies": {name: {"entropy": value["groups"]["all_selected"]["conditional_entropy_nats"]["mean"],
                                           "sigma": value["groups"]["all_selected"]["parts"]["all_actions"]["pretanh_sigma"]["mean"],
                                           "action_std": value["groups"]["all_selected"]["parts"]["all_actions"]["conditional_tanh_action_std"]["mean"]}
                                   for name, value in report["policies"].items()}}))


def write_markdown(report):
    lines = ["# 相同专家状态下的策略分布审计（CPU）", "",
             "比较的是 **给定同一个状态 s 时的动作分布**，不是把不同状态的动作混在一起算标准差。没有运行新仿真、训练或 Q 网络。", "",
             f"从官方成功轨迹8139个transition按phase占比抽取{report['selected_rows']}个相同状态，覆盖{report['selected_episodes']}个回合；每状态8个独立Gaussian噪声样本，各策略共用这些噪声以便配对比较。", "",
             "## 总体结果", "",
             "| 策略 | 平均 |μ|（tanh前） | 平均σ（tanh前） | 给定状态的tanh动作std | P(|action|>.99)，解析 | 条件熵 H，nats | α=.01时每步αH | α=.001时每步αH |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    # Escape literal bars in the first table's header for Markdown rendering.
    lines[-2] = "| 策略 | tanh前均值绝对值 | tanh前σ | 给定状态的tanh动作std | 动作绝对值>.99的概率 | 条件熵H(nats) | 每步.01H | 每步.001H |"
    for name, policy in report["policies"].items():
        group = policy["groups"]["all_selected"]
        part = group["parts"]["all_actions"]
        h = group["conditional_entropy_nats"]["mean"]
        lines.append(f"| {name} | {part['pretanh_abs_mean']['mean']:.4f} | {part['pretanh_sigma']['mean']:.4f} | {part['conditional_tanh_action_std']['mean']:.4f} | {part['abs_action_gt099_probability_analytic']*100:.2f}% | {h:.3f} | {.01*h:.4f} | {.001*h:.4f} |")
    lines += ["", "三份BC策略的确定性均值在这批状态上完全相同；校准只改变std头。原生BC的std头未经过BC优化，不能将其随机采样表现等同于其确定性控制能力。", "",
              "## 为什么σ不等于实际动作随机性", "",
              "`u = μ(s) + σ(s) ε`，`a = tanh(u)`。在均值接近0时，tanh近似线性；在均值绝对值很大时，动作挤在±1附近，相同σ造成的动作变化小得多。因此应同时看pre-tanh均值、σ、条件动作std和边界概率。", "",
              "连续动作的条件微分熵满足 `H(a|s) = H(u|s) + E[log(1 - tanh(u)^2)]`。后项非正，靠近边界时可能非常负。即使Gaussian层有一定σ，tanh后的分布仍可能高度集中。密度可以大于1，所以微分熵为负不是非法概率或数值错误。", "",
              "本审计用解析Gaussian熵加上8次MC的Jacobian项，并复用仓库的 `safe_tanh_log_det_jacobian`。另存原生 `-logπ` MC结果交叉核对。动作std只在同一状态的8个样本间计算，不把跨状态策略变化算作探索。", "",
              "| 策略 | tanh前Gaussian熵 | tanh变换Jacobian项 | 相加后动作熵 |", "| --- | ---: | ---: | ---: |"]
    for name, policy in report["policies"].items():
        group = policy["groups"]["all_selected"]
        lines.append(f"| {name} | {group['gaussian_entropy_nats']['mean']:.3f} | {group['tanh_log_jacobian_term_nats']['mean']:.3f} | {group['conditional_entropy_nats']['mean']:.3f} |")
    lines += ["", "例如σ≈.15校准版：Gaussian熵为−16.72，数值接近原生目标−13.87，但tanh变换再减去94.24，实际动作熵变成−110.96。不能用pre-tanh σ或其均值直接判断是否达到温度目标。", "",
              "原生BC的平均σ≈.995也很容易误导：所有状态×动作维度中，σ中位数只有.0348，10%分位为.000105，90%分位却达到4.009。Gaussian熵依赖log(σ)，不是σ的算术平均。", "",
              "**低微分熵不等于动作安静或控制安全。** 原生BC的动作std≈.181，明显大于.05校准版的.0117，但它的熵反而更低。一些维度极窄，另一些维度很宽或挤在边界，可以同时发生；因此不能按总熵排序控制平稳程度。", "",
              "## 原生温度目标", "",
              f"代码由 `temp_target_sigma=.15` 计算 `0.5 * 29 * log(2πe*.15²) = {report['target_entropy_from_native_sigma015_formula']:.6f}` nats。29维[-1,1]动作的最大条件微分熵为 `{report['bounded_action_max_differential_entropy']:.6f}` nats。",
              "这里的.15是构造Gaussian熵参考的参数，不等于执行tanh后的每维std必须为.15。SAC温度更新比较实际条件熵与该目标；对于高度饱和的均值，即使pre-tanh σ=.15，实际熵也可能远低于目标。", "",
              "若实际更新批次的熵低于目标，现有温度损失对提高alpha有压力；但本表使用冻结专家状态和inference BN，不能替代真实训练批次的梯度测试。策略最终如何变化还取决于Q、状态分布及优化器，不能只凭本表预测会必然破坏技能。", "",
              "表中αH仅是指定α下的 **每步条件熵项**，没有乘600步、没有当作discounted return，也未与未归一化STR reward直接比较。它不是最优策略收益推导，也不说明SAC无法完成任务。", "",
              "## Phase分层", "",
              "| phase | 状态数 | BC原σ：H | BCσ≈.05：H | BCσ≈.15：H | 旧Flash：H |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for phase, count in report["selected_phase_counts"].items():
        hs = [value["groups"][phase]["conditional_entropy_nats"]["mean"] for value in report["policies"].values()]
        lines.append(f"| {phase} | {count} | " + " | ".join(f"{h:.3f}" for h in hs) + " |")
    lines += ["", "## 限制", "",
              "- 旧Flash在这些专家状态上属于off-policy/OOD查询；不能把这里的熵当作它原训练replay或实际采样分布的统计。",
              "- 使用inference BN且状态不更新；不等于SAC actor训练的BN环境。各模型和BN buffers经核对未修改。",
              "- 原始噪声持续与控制器平滑会影响时间连续性和关节目标；本次只审计单状态动作边际，不审计整个序列的熵率或实际控制目标std。",
              "- tanh动作按FP32计算，极端饱和可能舍入为±1；熵公式描述底层连续tanh模型，而非量化后符号的离散熵。",
              "- MC只有每状态8次；JSON中的MC标准误仅针对固定状态的噪声积分，不代表跨状态或训练seed的不确定性。",
              "- 所有模型、数据SHA256、同一抽样索引、各phase及arm/hand分量统计在 `policy_distribution_audit.json`。"]
    (ROOT / "policy_distribution_audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
