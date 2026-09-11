#!/usr/bin/env python3
"""CPU-only, paired descriptive analysis of check_str_q_action_ranking summaries."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ("initial_q", "discounted_raw_return", "discounted_normalized_return",
           "finite_soft_return", "finite_nstep_soft_return", "goals_gained",
           "max_height_delta", "above10cm_longest_seconds")
ASSET_FAMILIES = frozenset(("brush", "eraser", "hammer", "screwdriver", "spatula", "marker"))
INFORMATIVE_RETURNS = frozenset(("discounted_normalized_return", "finite_soft_return", "finite_nstep_soft_return"))
WARNINGS = [
    "Primary: compare matching (prefix, noise seed, env), require every configured noise seed, then average delta within env. No termination-based primary filtering.",
    "Intervals are descriptive percentile bootstrap intervals over env clusters, not independent noise-seed episodes or confirmatory significance tests. Asset families are mixed and not stratified.",
    "Observed mean-repeat maximum is only an empirical repeat reference, NOT a noise confidence bound; exceeding it is NOT statistical significance.",
    "Direction uses actual initial-Q delta, never the branch name. exact_sign includes roundoff. Informative counts require abs(delta Q)>1e-6 AND abs(delta normalized/dense/n-step return)>1e-4; these are magnitude screens, NOT confidence bounds or Q accuracy. No informative raw-return statistic is provided. Unknown timeout/horizon tails are omitted.",
    "Dense entropy and n-step-grid entropy returns are distinct. Neither is exact historical training Q: Gaussian continuation differs from temporal-Zeta replay behavior, and off-policy/projection effects remain.",
    "Both-truly-terminated subset is outcome-selected and potentially biased; its available noise seeds can differ by env. It is NOT the primary estimate.",
    "Positive goals/height proxies are not full-task or stable-grasp success. Zero eligible pairs produce null metrics, not zero success.",
]


def asset_family(asset):
    matches = ASSET_FAMILIES.intersection(Path(asset).stem.lower().split("_"))
    return next(iter(matches)) if len(matches) == 1 else "ambiguous" if matches else "unknown"


def stats(values, samples, seed):
    """One value per env; all noise seeds already averaged inside that value."""
    if not values:
        return None
    data = np.asarray(list(values.values()), dtype=float)
    assert np.isfinite(data).all()
    ci = None
    if len(data) >= 2 and samples:
        rng = np.random.default_rng(seed)
        means = data[rng.integers(0, len(data), size=(samples, len(data)))].mean(axis=1)
        ci = np.quantile(means, [.025, .975]).tolist()
    return {"env_count": len(data), "mean": float(data.mean()), "median": float(np.median(data)),
            "min": float(data.min()), "max": float(data.max()),
            "max_abs": float(np.abs(data).max()), "descriptive_ci95": ci,
            "positive_count": int((data > 0).sum()), "negative_count": int((data < 0).sum()),
            "zero_count": int((data == 0).sum()), "per_env": values}


def outcomes(rows):
    if not rows:
        return None
    tails = [float(r["unknown_tail_discount"]) for r in rows
             if not r["terminated"] and r.get("unknown_tail_discount") is not None]
    return {"paired_rollout_count": len(rows),
            **{key + "_count": sum(bool(r.get(key, False)) for r in rows)
               for key in ("terminated", "failure_terminated", "truncated", "horizon_censored")},
            "unknown_tail_count": sum(not r["terminated"] for r in rows),
            "unknown_tail_discount": {"count": len(tails), "min": min(tails),
                                      "max": max(tails), "mean": float(np.mean(tails))} if tails else None}


def paired_report(pairs, samples, seed):
    """pairs: env -> [(baseline row, branch row), ...], primary or labeled subset."""
    result = {"env_count": len(pairs), "pair_count": sum(map(len, pairs.values())),
              "seed_pair_count_by_env": {str(e): len(p) for e, p in pairs.items()},
              "baseline_outcomes": outcomes([a for p in pairs.values() for a, _ in p]),
              "branch_outcomes": outcomes([b for p in pairs.values() for _, b in p]),
              "metrics": {}}
    for metric in METRICS:
        deltas, row_deltas, per_env_row_max = {}, [], {}
        for env, env_pairs in pairs.items():
            values = [(a.get(metric), b.get(metric)) for a, b in env_pairs]
            if any(a is None or b is None or not np.isfinite(a) or not np.isfinite(b) for a, b in values):
                continue
            difference = [float(b) - float(a) for a, b in values]
            deltas[str(env)] = float(np.mean(difference))
            per_env_row_max[str(env)] = max(map(abs, difference))
            row_deltas.extend(difference)
        result["metrics"][metric] = stats(deltas, samples, seed)
        if result["metrics"][metric] is not None:
            result["metrics"][metric]["individual_pair_max_abs"] = max(map(abs, row_deltas))
            result["metrics"][metric]["individual_pair_max_abs_by_env"] = per_env_row_max
    q = result["metrics"]["initial_q"]
    for metric, report in result["metrics"].items():
        if report is None or metric == "initial_q" or q is None:
            continue
        common = sorted(set(report["per_env"]) & set(q["per_env"]))
        nonzero = [e for e in common if q["per_env"][e] != 0 and report["per_env"][e] != 0]
        same = sum(np.sign(q["per_env"][e]) == np.sign(report["per_env"][e]) for e in nonzero)
        report["actual_q_direction_descriptive_not_accuracy"] = {
            "sign_rule": "exact_sign", "common_env_count": len(common), "nonzero_both_count": len(nonzero),
            "same_sign_count": int(same), "same_sign_fraction": same / len(nonzero) if nonzero else None}
        if metric in INFORMATIVE_RETURNS:
            informative = [e for e in common if abs(q["per_env"][e]) > 1e-6 and abs(report["per_env"][e]) > 1e-4]
            matched = sum(np.sign(q["per_env"][e]) == np.sign(report["per_env"][e]) for e in informative)
            report["actual_q_informative_direction_descriptive_not_accuracy"] = {
                "sign_rule": "magnitude_screen_not_confidence_bound", "q_abs_threshold_exclusive": 1e-6,
                "return_abs_threshold_exclusive": 1e-4, "common_env_count": len(common),
                "q_above_threshold_count": sum(abs(q["per_env"][e]) > 1e-6 for e in common),
                "return_above_threshold_count": sum(abs(report["per_env"][e]) > 1e-4 for e in common),
                "informative_both_count": len(informative), "same_sign_count": int(matched),
                "same_sign_fraction": int(matched) / len(informative) if informative else None}
    return result


def analyze(summary, samples=5000, seed=0):
    if summary.get("status") != "complete":
        raise ValueError("Only complete summary.json is accepted, not a partial progress file")
    rows = summary["rows"]
    index = {}
    for row in rows:
        key = (row["prefix_steps"], row.get("continuation_seed"), row["branch"], row["env_id"])
        if key in index:
            raise ValueError(f"Duplicate paired key: {key}")
        index[key] = row
    prefixes = sorted(set(summary.get("arguments", {}).get("prefix_steps", [])) |
                      {r["prefix_steps"] for r in rows})
    branches = summary.get("branches") or sorted({r["branch"] for r in rows})
    if "mean" not in branches or "mean_repeat" not in branches:
        raise ValueError("Both mean and mean_repeat are required")
    n_steps = {r["n_step"] for r in rows if r.get("n_step") is not None}
    if len(n_steps) > 1:
        raise ValueError("Cannot mix different n_step definitions")
    result = {"schema_version": 1, "bootstrap_samples": samples, "bootstrap_seed": seed,
              "warnings": WARNINGS, "n_step": next(iter(n_steps), summary.get("n_step")),
              "gamma": summary.get("gamma"), "gamma_to_horizon": summary.get("gamma_to_horizon"),
              "hard_reset": summary.get("hard_reset"), "prefixes": {}}
    for prefix in prefixes:
        seeds = summary.get("continuation_seeds_by_prefix", {}).get(str(prefix))
        if seeds is None:
            seeds = sorted({r.get("continuation_seed") for r in rows if r["prefix_steps"] == prefix}, key=str)
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError(f"Empty or duplicated seed list for prefix {prefix}")
        count = summary.get("arguments", {}).get("num_envs")
        envs = list(range(count)) if count is not None else sorted({r["env_id"] for r in rows if r["prefix_steps"] == prefix})
        reports = {}
        for branch in branches:
            if branch == "mean":
                continue
            complete, any_pairs, reasons = {}, {}, Counter()
            for env in envs:
                matched = []
                for noise_seed in seeds:
                    a = index.get((prefix, noise_seed, "mean", env))
                    b = index.get((prefix, noise_seed, branch, env))
                    reason = ("missing_baseline_row" if a is None else "missing_branch_row" if b is None else
                              "baseline_gate_failed" if not a["paired"] else "branch_gate_failed" if not b["paired"] else
                              "zero_length" if a["length"] <= 0 or b["length"] <= 0 else None)
                    if reason:
                        reasons[reason] += 1
                    else:
                        matched.append((a, b))
                if matched:
                    any_pairs[env] = matched
                if len(matched) == len(seeds):
                    complete[env] = matched
            report = paired_report(complete, samples, seed)
            report["coverage"] = {"expected_env_count": len(envs), "expected_seeds": seeds,
                                  "expected_pair_count": len(envs) * len(seeds),
                                  "valid_pair_count_before_complete_seed_filter": sum(map(len, any_pairs.values())),
                                  "any_pair_env_count": len(any_pairs), "complete_seed_env_count": len(complete),
                                  "excluded_incomplete_seed_env_count": len(envs) - len(complete),
                                  "unusable_pair_reasons": dict(reasons)}
            report["asset_family_env_counts"] = dict(Counter(asset_family(p[0][0].get("asset", "unknown"))
                                                               for p in complete.values()))
            conditional = {env: [(a, b) for a, b in p if a["terminated"] and b["terminated"]]
                           for env, p in complete.items()}
            conditional = {env: p for env, p in conditional.items() if p}
            report["both_truly_terminated_outcome_selected"] = paired_report(conditional, samples, seed)
            reports[branch] = report
        repeat = reports["mean_repeat"]["metrics"]
        for branch, report in reports.items():
            for metric, values in report["metrics"].items():
                noise = repeat[metric]
                if values is None:
                    continue
                common = sorted(set(values["per_env"]) & set(noise["per_env"])) if noise else []
                repeat_max = max((abs(noise["per_env"][e]) for e in common), default=None) if noise else None
                repeat_row_max = max((noise["individual_pair_max_abs_by_env"][e] for e in common), default=None) if noise else None
                exceed = [e for e in common if abs(values["per_env"][e]) > repeat_max] if repeat_max is not None else []
                q_values = report["metrics"]["initial_q"]
                directional = [e for e in exceed if q_values and q_values["per_env"].get(e, 0) != 0]
                values["observed_repeat_reference_not_confidence_bound"] = {
                    "repeat_env_count": noise["env_count"] if noise else 0,
                    "repeat_env_mean_max_abs": repeat_max,
                    "repeat_individual_pair_max_abs": repeat_row_max,
                    "common_repeat_env_count": len(common),
                    "exceed_env_mean_max_count": len(exceed) if repeat_max is not None else None,
                    "exceed_and_nonzero_q_count": len(directional) if repeat_max is not None else None,
                    "exceed_and_same_q_sign_count": int(sum(np.sign(values["per_env"][e]) == np.sign(q_values["per_env"][e])
                                                            for e in directional)) if repeat_max is not None else None}
        result["prefixes"][str(prefix)] = {"continuation_seeds": seeds, "branches": reports}
    return result


def markdown(result):
    lines = ["# Q/action probe: paired descriptive analysis", "", *[f"- {w}" for w in result["warnings"]], ""]
    def value(item):
        if item is None:
            return "null"
        ci = item["descriptive_ci95"]
        return f"{item['mean']:.6g}" + (f" [{ci[0]:.6g}, {ci[1]:.6g}]" if ci else " [CI null]")
    for prefix, group in result["prefixes"].items():
        lines += [f"## Prefix {prefix}; noise seeds {group['continuation_seeds']}", "",
                  "| Branch | Complete envs | Actual ΔQ | Δn-step soft return [descriptive CI] | Δgoals | Δmax height (m) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for branch, report in group["branches"].items():
            m = report["metrics"]
            lines.append(f"| {branch} | {report['env_count']} | {value(m['initial_q'])} | {value(m['finite_nstep_soft_return'])} | {value(m['goals_gained'])} | {value(m['max_height_delta'])} |")
        repeat = group["branches"]["mean_repeat"]
        lines += ["", f"Repeat outcomes (paired rollout counts, not independent envs): {json.dumps(repeat['branch_outcomes'])}", "",
                  "All raw/dense/n-step/height metrics, missing-pair reasons, observed-repeat references, tail weights, and outcome-selected subsets are in analysis.json.", ""]
    return "\n".join(lines)


def self_test():
    assert asset_family("041_brush_handle_mesh.urdf") == "brush"
    assert asset_family("/some/path/026_screwdriver_handle_mesh.urdf") == "screwdriver"
    assert all(asset_family(f"099_{family}_mesh.urdf") == family for family in ASSET_FAMILIES)
    assert asset_family("041_toothbrush_mesh.urdf") == "unknown"
    assert asset_family("041_handle_mesh.urdf") == "unknown"
    assert asset_family("041_brush_hammer_mesh.urdf") == "ambiguous"
    rows = []
    for prefix in (0, 60):
        for noise_seed in (7, 19):
            for env in range(4):
                for branch in ("mean", "mean_repeat", "arm_q_plus"):
                    base = 1000 * noise_seed + env
                    delta = (2 if noise_seed == 7 else 4) if branch == "arm_q_plus" else (.25 if noise_seed == 7 else -.25) if branch == "mean_repeat" else 0
                    rows.append(dict(prefix_steps=prefix, continuation_seed=noise_seed, env_id=env, branch=branch,
                        paired=prefix == 0 and not (env == 3 and noise_seed == 19 and branch == "arm_q_plus"),
                        length=30, n_step=3, initial_q=10 - (branch == "arm_q_plus"),
                        **{m: base + delta for m in METRICS if m != "initial_q"},
                        terminated=env == 0, truncated=env == 1, horizon_censored=env >= 2,
                        unknown_tail_discount=0 if env == 0 else .74, asset="brush_test.urdf"))
    source = dict(status="complete", rows=rows, branches=["mean", "mean_repeat", "arm_q_plus"],
                  arguments=dict(num_envs=4, prefix_steps=[0, 60]), continuation_seeds_by_prefix={"0": [7, 19], "60": [7, 19]})
    original = copy.deepcopy(source)
    result = analyze(source, 500, 2)
    branch = result["prefixes"]["0"]["branches"]["arm_q_plus"]
    assert source == original and branch["env_count"] == 3 and branch["pair_count"] == 6
    metric = branch["metrics"]["finite_nstep_soft_return"]
    assert metric["mean"] == 3 and metric["descriptive_ci95"] == [3, 3]
    assert metric["actual_q_direction_descriptive_not_accuracy"]["same_sign_fraction"] == 0
    assert metric["actual_q_direction_descriptive_not_accuracy"]["sign_rule"] == "exact_sign"
    assert metric["actual_q_informative_direction_descriptive_not_accuracy"]["informative_both_count"] == 3
    assert branch["asset_family_env_counts"] == {"brush": 3}
    roundoff = copy.deepcopy(source)
    bases = {(r["prefix_steps"], r["continuation_seed"], r["env_id"]): r for r in roundoff["rows"] if r["branch"] == "mean"}
    for r in roundoff["rows"]:
        if r["branch"] == "arm_q_plus":
            a = bases[r["prefix_steps"], r["continuation_seed"], r["env_id"]]
            r["initial_q"] = a["initial_q"] + 5e-7
            for m in INFORMATIVE_RETURNS:
                r[m] = a[m] + 5e-5
    small = analyze(roundoff, 0)["prefixes"]["0"]["branches"]["arm_q_plus"]["metrics"]
    for m in INFORMATIVE_RETURNS:
        assert small[m]["actual_q_direction_descriptive_not_accuracy"]["same_sign_fraction"] == 1
        assert small[m]["actual_q_informative_direction_descriptive_not_accuracy"]["informative_both_count"] == 0
        assert small[m]["actual_q_informative_direction_descriptive_not_accuracy"]["same_sign_fraction"] is None
    assert "actual_q_informative_direction_descriptive_not_accuracy" not in small["discounted_raw_return"]
    for q_delta, return_delta, expected in ((5e-7, .1, 0), (.1, 5e-5, 0), (.1, .1, 3)):
        for r in roundoff["rows"]:
            if r["branch"] == "arm_q_plus":
                a = bases[r["prefix_steps"], r["continuation_seed"], r["env_id"]]
                r["initial_q"] = a["initial_q"] + q_delta
                for m in INFORMATIVE_RETURNS:
                    r[m] = a[m] + return_delta
        screened = analyze(roundoff, 0)["prefixes"]["0"]["branches"]["arm_q_plus"]["metrics"]
        assert all(screened[m]["actual_q_informative_direction_descriptive_not_accuracy"]["informative_both_count"] == expected
                   for m in INFORMATIVE_RETURNS)
    noise = metric["observed_repeat_reference_not_confidence_bound"]
    assert noise["repeat_env_mean_max_abs"] == 0 and noise["repeat_individual_pair_max_abs"] == .25
    assert noise["exceed_env_mean_max_count"] == 3
    conditional = branch["both_truly_terminated_outcome_selected"]
    assert conditional["env_count"] == 1 and conditional["metrics"]["initial_q"]["descriptive_ci95"] is None
    empty = result["prefixes"]["60"]["branches"]["arm_q_plus"]
    assert empty["env_count"] == 0 and empty["metrics"]["initial_q"] is None and empty["branch_outcomes"] is None
    baseline_only = copy.deepcopy(source)
    baseline_only["branches"] = ["mean", "mean_repeat"]
    baseline_only["rows"] = [r for r in rows if r["branch"] != "arm_q_plus"]
    assert set(analyze(baseline_only)["prefixes"]["0"]["branches"]) == {"mean_repeat"}
    source["rows"].append(source["rows"][0])
    try:
        analyze(source)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Duplicate key accepted")
    json.dumps(result, allow_nan=False)
    print("PASS: exact asset tokens and negative cases; roundoff screened separately; exact multi-seed pairing; env clustering; complete-seed gate; actual-Q signs; repeat references; outcome subset; null empty pairs; baseline-only; duplicates; immutable input")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.summary is None or args.output_dir is None or args.bootstrap < 0 or args.bootstrap_seed < 0:
        parser.error("Require --summary, --output-dir and nonnegative bootstrap settings")
    source = args.summary.read_bytes()
    result = analyze(json.loads(source), args.bootstrap, args.bootstrap_seed)
    result["provenance"] = {"source": str(args.summary.resolve()), "source_sha256": hashlib.sha256(source).hexdigest(),
                            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if source != args.summary.read_bytes():
        raise RuntimeError("Source changed during analysis; refusing to save")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"analysis.json": json.dumps(result, indent=2, allow_nan=False) + "\n", "analysis.md": markdown(result)}
    if any((args.output_dir / name).exists() for name in outputs):
        raise FileExistsError("Refusing to overwrite an existing analysis.json or analysis.md")
    for name, contents in outputs.items():
        with (args.output_dir / name).open("x") as handle:
            handle.write(contents)
    print(f"Saved {args.output_dir.resolve()} (source unchanged; CPU only)")


if __name__ == "__main__":
    main()
