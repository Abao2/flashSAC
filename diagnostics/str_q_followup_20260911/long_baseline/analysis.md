# Q/action probe: paired descriptive analysis

- Primary: compare matching (prefix, noise seed, env), require every configured noise seed, then average delta within env. No termination-based primary filtering.
- Intervals are descriptive percentile bootstrap intervals over env clusters, not independent noise-seed episodes or confirmatory significance tests. Asset families are mixed and not stratified.
- Observed mean-repeat maximum is only an empirical repeat reference, NOT a noise confidence bound; exceeding it is NOT statistical significance.
- Direction uses actual initial-Q delta, never the branch name. Direction agreement is NOT Q accuracy. Unknown timeout/horizon tails are omitted.
- Dense entropy and n-step-grid entropy returns are distinct. Neither is exact historical training Q: Gaussian continuation differs from temporal-Zeta replay behavior, and off-policy/projection effects remain.
- Both-truly-terminated subset is outcome-selected and potentially biased; its available noise seeds can differ by env. It is NOT the primary estimate.
- Positive goals/height proxies are not full-task or stable-grasp success. Zero eligible pairs produce null metrics, not zero success.

## Prefix 0; noise seeds [30000]

| Branch | Complete envs | Actual ΔQ | Δn-step soft return [descriptive CI] | Δgoals | Δmax height (m) |
|---|---:|---:|---:|---:|---:|
| mean_repeat | 32 | 0 [0, 0] | -0.00117592 [-0.0035355, 7.75093e-06] | -0.0625 [-0.1875, 0] | 3.58243e-05 [-0.000166319, 0.000273792] |

Repeat outcomes (paired rollout counts, not independent envs): {"paired_rollout_count": 32, "terminated_count": 23, "failure_terminated_count": 23, "truncated_count": 4, "horizon_censored_count": 5, "unknown_tail_count": 9, "unknown_tail_discount": {"count": 9, "min": 0.0024050092913110673, "max": 0.0024050092913110673, "mean": 0.0024050092913110673}}

All raw/dense/n-step/height metrics, missing-pair reasons, observed-repeat references, tail weights, and outcome-selected subsets are in analysis.json.

## Prefix 60; noise seeds [30060]

| Branch | Complete envs | Actual ΔQ | Δn-step soft return [descriptive CI] | Δgoals | Δmax height (m) |
|---|---:|---:|---:|---:|---:|
| mean_repeat | 28 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |

Repeat outcomes (paired rollout counts, not independent envs): {"paired_rollout_count": 28, "terminated_count": 12, "failure_terminated_count": 12, "truncated_count": 5, "horizon_censored_count": 11, "unknown_tail_count": 16, "unknown_tail_discount": {"count": 16, "min": 0.0024050092913110673, "max": 0.004395467595536361, "mean": 0.0030270275113814717}}

All raw/dense/n-step/height metrics, missing-pair reasons, observed-repeat references, tail weights, and outcome-selected subsets are in analysis.json.

## Prefix 120; noise seeds [30120]

| Branch | Complete envs | Actual ΔQ | Δn-step soft return [descriptive CI] | Δgoals | Δmax height (m) |
|---|---:|---:|---:|---:|---:|
| mean_repeat | 21 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |

Repeat outcomes (paired rollout counts, not independent envs): {"paired_rollout_count": 21, "terminated_count": 8, "failure_terminated_count": 8, "truncated_count": 6, "horizon_censored_count": 7, "unknown_tail_count": 13, "unknown_tail_discount": {"count": 13, "min": 0.0024050092913110673, "max": 0.008033289290486696, "mean": 0.0041891971233668605}}

All raw/dense/n-step/height metrics, missing-pair reasons, observed-repeat references, tail weights, and outcome-selected subsets are in analysis.json.
