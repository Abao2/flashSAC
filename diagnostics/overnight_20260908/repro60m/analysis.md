# Fresh continuous60M — current CPU snapshot

Updated: 2026-09-08T20:33:06.105318+00:00; queue: completed.

Fresh random native feed-forward SAC, continuous60M; training seeds0/1/2, common rollout seed0; no BC or reload.

|Training seed|Checkpoint|Training job|Online goal before/after checkpoint|Frozen det goals|Frozen native-stochastic goals|
|---|---|---|---|---|---|
|0|50,001,920|completed|0.00% @49,971,200 / 0.00% @50,022,400|0/64|0/128|
|0|60,002,304|completed|82.31% @60,002,304 / 82.31% @60,002,304|62/64|127/128|
|1|50,001,920|completed|25.00% @49,971,200 / 17.57% @50,022,400|1/64|40/128|
|1|60,002,304|completed|44.88% @60,002,304 / 44.88% @60,002,304|1/64|88/128|
|2|50,001,920|completed|96.95% @49,971,200 / 95.57% @50,022,400|64/64|125/128|
|2|60,002,304|completed|99.23% @60,002,304 / 99.23% @60,002,304|64/64|128/128|

Checkpoint details (exact attempted/applied counters, alpha, finite checks, normalizer sample counts and hashes) are in `summary.json`. Online windows and frozen evaluations are different metrics.

## Limits

- Missing data is PENDING, not zero. Partial evaluations are labeled with completed counts.
- TB values are episode logging-window means, not an episode-weighted overall success rate; multiple run directories are not merged.
- Three independent training seeds, each evaluated with the same rollout seed0; correlated vector episodes do not justify an independent-IID confidence interval.
- Fixed eraser/start/goal, DR off, privileged simulator-state162; not paper24-task replication or real-robot validation.
- Extra continuous budget tests whether reload is unnecessary, but does not isolate the previous cold-replay/full-reload cause.
- Checkpoint counters count attempted native calls; AMP can skip applied optimizer steps. Full file audits and normalizer budget checks are separate.
