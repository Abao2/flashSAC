"""Run the authorized, bounded STR information/memory training comparison."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

from run_str_diagnostic_queue import run_queue


def check_sources(manifest):
    for filename, expected in manifest['source_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen comparison source changed: {filename}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--launch-job')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    check_sources(manifest)
    if args.launch_job:
        job = next(job for job in manifest['jobs'] if job['id'] == args.launch_job)
        os.chdir(job['cwd'])
        os.execvpe(job['argv'][0], job['argv'], {**os.environ, **job.get('env', {})})
    proof = json.loads(Path(manifest['gpu_preflight']).read_text())
    assert proof['status'] == 'PASS' and proof['losses_finite'] and proof['lstm_parameters_changed']
    assert proof['history_length'] == 32 and proof['critic_network_observation_dim'] == 4674
    cpu = json.loads(Path(manifest['cpu_validation']).read_text())
    assert cpu['status'] == 'PASS' and cpu['tests_passed'] >= 59
    for marker in manifest['baseline_completion_files']:
        if not Path(marker).is_file():
            raise RuntimeError(f'Existing FF140 baseline checkpoint missing: {marker}')
    # The shared queue owns its children. Re-check frozen source before each
    # training job, without editing the original manifest or its old baseline.
    launch = dict(manifest)
    launch['jobs'] = [dict(job, argv=[sys.executable, str(Path(__file__).resolve()),
                                    str(args.manifest.resolve()), '--launch-job', job['id']])
                      if job['kind'] == 'training' else job for job in manifest['jobs']]
    deadline = datetime.fromisoformat(manifest['deadline_utc']).timestamp()
    result = run_queue(launch, args.manifest.parent / 'queue_status.json', deadline)
    print(json.dumps({'status': result['status']}), flush=True)
    raise SystemExit(0 if result['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
