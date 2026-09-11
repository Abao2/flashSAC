"""Extract selected Wuji SAPG scalars from the copied event without importing training code."""

import json
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader


ROOT = Path(__file__).resolve().parent
EVENT = next(ROOT.glob("events.out.tfevents*"))
TAGS = {
    "return": "rewards/step",
    "fingertip": "episode_cumulative/fingertip_delta_rew",
    "lifting": "episode_cumulative/lifting_rew",
    "lift_bonus": "episode_cumulative/lift_bonus_rew",
    "keypoint": "episode_cumulative/keypoint_rew",
    "goals": "episode_final/successes",
    "successes_tracker": "successes",
}
REVERSE = {tag: key for key, tag in TAGS.items()}
bins = {key: {} for key in TAGS}
counts = {key: 0 for key in TAGS}
first = {}
last = {}

for event in LegacyEventFileLoader(str(EVENT)).Load():
    for value in event.summary.value:
        key = REVERSE.get(value.tag)
        if key is None:
            continue
        point = [int(event.step), float(value.simple_value)]
        first.setdefault(key, point)
        last[key] = point
        counts[key] += 1
        bucket = bins[key].setdefault(event.step // 100_000_000, [0, 0.0, 0])
        bucket[0] += event.step
        bucket[1] += value.simple_value
        bucket[2] += 1

curves = {
    key: [[row[0] / row[2], row[1] / row[2]] for _, row in sorted(values.items())]
    for key, values in bins.items()
}
payload = {
    "kind": "history_100M_binned",
    "metadata": {
        "host": "xug52",
        "source_root": "/home/lixiyuan/wuji-s2r/runs/wuji/nodr_nonoise_expl002_resume_seed0_20260905/0_wuji_m6_left_sapg_24576_seed0_resume_nodr_nonoise_expl002_20260905/summaries",
        "local_event": str(EVENT),
        "source_tags": TAGS,
        "embodiment": "M6 + Wuji left",
        "domain_randomization": "DR/noise/delays disabled; action EMA arm=0.1, hand=0.07",
        "statistics": "mixed_expl block5 completed-episode window; not all 24576 envs",
        "bin_definition": "floor(step/100000000), arithmetic mean of logged x and y",
    },
    "curves": curves,
    "raw_count_by_key": counts,
    "first_raw": first,
    "last_raw": last,
}
print(json.dumps(payload, separators=(",", ":")))
