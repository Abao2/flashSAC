# Q/action probe: paired descriptive analysis

- Primary: compare matching (prefix, noise seed, env), require every configured noise seed, then average delta within env. No termination-based primary filtering.
- Intervals are descriptive percentile bootstrap intervals over env clusters, not independent noise-seed episodes or confirmatory significance tests. Asset families are mixed and not stratified.
- Observed mean-repeat maximum is only an empirical repeat reference, NOT a noise confidence bound; exceeding it is NOT statistical significance.
- Direction uses actual initial-Q delta, never the branch name. exact_sign includes roundoff. Informative counts require abs(delta Q)>1e-6 AND abs(delta normalized/dense/n-step return)>1e-4; these are magnitude screens, NOT confidence bounds or Q accuracy. No informative raw-return statistic is provided. Unknown timeout/horizon tails are omitted.
- Dense entropy and n-step-grid entropy returns are distinct. Neither is exact historical training Q: Gaussian continuation differs from temporal-Zeta replay behavior, and off-policy/projection effects remain.
- Both-truly-terminated subset is outcome-selected and potentially biased; its available noise seeds can differ by env. It is NOT the primary estimate.
- Positive goals/height proxies are not full-task or stable-grasp success. Zero eligible pairs produce null metrics, not zero success.

## Prefix 0; noise seeds [30000, 30060, 30120]

| Branch | Complete envs | Actual ΔQ | Δn-step soft return [descriptive CI] | Δgoals | Δmax height (m) |
|---|---:|---:|---:|---:|---:|
| mean_repeat | 32 | 0 [0, 0] | -0.000391972 [-0.0011785, 2.58364e-06] | -0.0208333 [-0.0625, 0] | 1.19414e-05 [-5.54398e-05, 9.1264e-05] |
| zero | 32 | -0.0136606 [-0.0179029, -0.0099545] | -0.0125475 [-0.0499209, 0.0231246] | 0.0625 [-0.260417, 0.375] | 0.00826806 [-0.0293602, 0.0416313] |
| arm_q_plus | 32 | 0.00032423 [0.000245857, 0.00040878] | 0.00395917 [-0.0563255, 0.065752] | 0.125 [-0.385417, 0.71875] | 0.00141771 [-0.034709, 0.0342024] |
| arm_q_minus | 32 | -0.000610725 [-0.000755795, -0.000477262] | -0.0028611 [-0.0584706, 0.0501971] | 0.21875 [-0.229427, 0.677083] | 0.00334675 [-0.02935, 0.0306025] |
| hand_q_plus | 32 | 0.000439296 [0.00034312, 0.000551012] | -0.0299335 [-0.106839, 0.0414448] | -0.166667 [-0.635417, 0.291667] | 0.00115173 [-0.0227079, 0.0240626] |
| hand_q_minus | 32 | -0.0009061 [-0.00116634, -0.000683342] | -0.019791 [-0.0815345, 0.0360633] | 0.104167 [-0.479167, 0.666667] | 0.00764082 [-0.0181209, 0.0329623] |

Repeat outcomes (paired rollout counts, not independent envs): {"paired_rollout_count": 96, "terminated_count": 68, "failure_terminated_count": 68, "truncated_count": 13, "horizon_censored_count": 15, "unknown_tail_count": 28, "unknown_tail_discount": {"count": 28, "min": 0.0024050092913110673, "max": 0.0024050092913110673, "mean": 0.0024050092913110678}}

All raw/dense/n-step/height metrics, missing-pair reasons, observed-repeat references, tail weights, and outcome-selected subsets are in analysis.json.

## Prefix 60; noise seeds [30000, 30060, 30120]

| Branch | Complete envs | Actual ΔQ | Δn-step soft return [descriptive CI] | Δgoals | Δmax height (m) |
|---|---:|---:|---:|---:|---:|
| mean_repeat | 28 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| zero | 28 | -0.0868795 [-0.127007, -0.0514358] | -0.00504953 [-0.0717068, 0.0563888] | -0.107143 [-0.642857, 0.357143] | -0.0230664 [-0.0484371, -0.00038385] |
| arm_q_plus | 28 | 0.00233088 [0.00149382, 0.00322409] | 0.0241521 [-0.0745602, 0.128362] | 0.0714286 [-0.369048, 0.47619] | -0.000337113 [-0.0258965, 0.0266896] |
| arm_q_minus | 28 | -0.00378793 [-0.00537174, -0.00235742] | 0.0451632 [-0.000549706, 0.0966373] | 0.535714 [0.14256, 0.952381] | -0.0183728 [-0.0377451, -0.000929886] |
| hand_q_plus | 28 | 0.00137826 [0.000750755, 0.00213726] | 0.0121171 [-0.0322775, 0.0636032] | 0.238095 [-0.321429, 0.857143] | -0.011304 [-0.0341142, 0.00977817] |
| hand_q_minus | 28 | -0.00648824 [-0.00915015, -0.00401544] | 0.000370809 [-0.0641371, 0.0586804] | 0.357143 [-0.0595238, 0.833333] | -0.0142126 [-0.034215, 0.00557753] |

Repeat outcomes (paired rollout counts, not independent envs): {"paired_rollout_count": 84, "terminated_count": 43, "failure_terminated_count": 43, "truncated_count": 16, "horizon_censored_count": 25, "unknown_tail_count": 41, "unknown_tail_discount": {"count": 41, "min": 0.0024050092913110673, "max": 0.004395467595536361, "mean": 0.0031817735075941084}}

All raw/dense/n-step/height metrics, missing-pair reasons, observed-repeat references, tail weights, and outcome-selected subsets are in analysis.json.

## Prefix 120; noise seeds [30000, 30060, 30120]

| Branch | Complete envs | Actual ΔQ | Δn-step soft return [descriptive CI] | Δgoals | Δmax height (m) |
|---|---:|---:|---:|---:|---:|
| mean_repeat | 21 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| zero | 21 | -0.0887497 [-0.129755, -0.049948] | 0.0381109 [-0.10005, 0.201617] | 0.174603 [-0.333333, 0.730159] | -0.0216015 [-0.0705564, 0.0214386] |
| arm_q_plus | 21 | 0.00323096 [0.00170894, 0.00496378] | -0.0207834 [-0.110649, 0.0826895] | 0.190476 [-0.31746, 0.793651] | -0.0134974 [-0.0507927, 0.0249072] |
| arm_q_minus | 21 | -0.00369428 [-0.00552943, -0.00201267] | 0.0426145 [-0.0487734, 0.154749] | 0.301587 [-0.142857, 0.84127] | -0.0319454 [-0.0605553, -0.00736895] |
| hand_q_plus | 21 | 0.00165224 [0.000644201, 0.00291565] | -0.0220435 [-0.111887, 0.0777726] | 0.111111 [-0.238095, 0.555556] | -0.0164866 [-0.0324121, -0.00185002] |
| hand_q_minus | 21 | -0.00632059 [-0.00946041, -0.00339594] | 0.0189791 [-0.071102, 0.109071] | 0.47619 [0.0793651, 0.888889] | -0.0177367 [-0.0391727, 0.000413801] |

Repeat outcomes (paired rollout counts, not independent envs): {"paired_rollout_count": 63, "terminated_count": 24, "failure_terminated_count": 24, "truncated_count": 21, "horizon_censored_count": 18, "unknown_tail_count": 39, "unknown_tail_discount": {"count": 39, "min": 0.0024050092913110673, "max": 0.008033289290486696, "mean": 0.004761961947357103}}

All raw/dense/n-step/height metrics, missing-pair reasons, observed-repeat references, tail weights, and outcome-selected subsets are in analysis.json.
