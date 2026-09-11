"""Read-only xug6 extraction; run via ssh python stdin, never writes remotely."""
import hashlib
import json
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

root = Path('/home/lixiyuan/simtoolreal/tb/s2r_seed0_full/merged')
event = next(root.glob('events.out.tfevents*'))
tags = {'return': 'rewards/step', 'fingertip': 'episode_cumulative/fingertip_delta_rew',
        'lifting': 'episode_cumulative/lifting_rew', 'lift_bonus': 'episode_cumulative/lift_bonus_rew',
        'keypoint': 'episode_cumulative/keypoint_rew', 'goals': 'episode_final/successes',
        'successes_tracker': 'successes'}
reverse = {v: k for k, v in tags.items()}
verify = json.loads((root / 'verify_report.json').read_text())
meta = {'host': 'xug6', 'source_root': str(root), 'resolved_root': str(root.resolve()),
        'event_path': str(event.resolve()), 'event_size_bytes': event.stat().st_size,
        'event_mtime_ns': event.stat().st_mtime_ns, 'source_tags': tags,
        'manifest_sha256': hashlib.sha256((root / 'manifest.snapshot.json').read_bytes()).hexdigest(),
        'verify_sha256': hashlib.sha256((root / 'verify_report.json').read_bytes()).hexdigest(),
        'note': 'Episode logging means; rewards/step denotes x-axis, not per-timestep reward. No new rollout.',
        'bin_definition': 'floor(step/100000000); arithmetic mean of logged y, arithmetic mean of actual logged steps; not episode-weighted.'}
early = {k: [] for k in tags}
bins = {k: {} for k in tags}
counts = {k: 0 for k in tags}
first, last = {}, {}
early_sent = False
for e in LegacyEventFileLoader(str(event)).Load():
    for v in e.summary.value:
        if v.tag not in reverse:
            continue
        key = reverse[v.tag]
        assert v.HasField('simple_value'), v.tag
        point = [e.step, v.simple_value]
        if key in last:
            assert e.step > last[key][0], (key, e.step, last[key])
        first.setdefault(key, point)
        last[key] = point
        counts[key] += 1
        if e.step <= 100_000_000:
            early[key].append(point)
        elif not early_sent and all(early[k] for k in tags):
            print(json.dumps({'kind': 'early_raw', 'metadata': meta, 'curves': early,
                              'points_kind': 'raw_logged_points_step_le100M'}, separators=(',', ':')), flush=True)
            early_sent = True
            if '--early-only' in sys.argv:
                raise SystemExit(0)
        b = bins[key].setdefault(e.step // 100_000_000, [0, 0., 0])
        b[0] += e.step
        b[1] += v.simple_value
        b[2] += 1
for key, tag in tags.items():
    assert counts[key] == verify['count_by_tag'][tag]
    assert first[key][0] == verify['first_step_by_tag'][tag]
    assert last[key][0] == verify['last_step_by_tag'][tag]
curves = {k: [[b[0] / b[2], b[1] / b[2]] for _, b in sorted(values.items())] for k, values in bins.items()}
sizes = {k: [b[2] for _, b in sorted(values.items())] for k, values in bins.items()}
print(json.dumps({'kind': 'history_100M_binned', 'metadata': meta, 'curves': curves,
                  'bin_counts': sizes, 'raw_count_by_key': counts, 'first_raw': first,
                  'last_raw': last, 'points_kind': '100M_bin_arithmetic_means'}, separators=(',', ':')), flush=True)
