"""Run the existing serial diagnostic queue only after verified 500M completion."""
from datetime import datetime
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[2] / 'scripts'))
from run_str_diagnostic_queue import run_queue

parent_status = json.loads((root.parent / 'queue_status.json').read_text())
completion = json.loads((root.parent / 'candidate/training_complete.json').read_text())
checkpoint = root.parent / 'candidate/models/seed0/step195320'
if (parent_status['status'] != 'completed' or completion['status'] != 'complete'
        or completion.get('missing') or not completion.get('replay_saved')
        or Path(completion['checkpoint']).resolve() != checkpoint.resolve()):
    raise RuntimeError('500M training has not completed successfully')
for name in ('actor.pt', 'critic.pt', 'target_critic.pt', 'temperature.pt',
             'agent_state.pt', 'reward_normalizer.pt', 'replay_buffer.pt'):
    if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
        raise RuntimeError(f'Missing complete 500M state: {name}')
manifest = json.loads((root / 'manifest.json').read_text())
result = run_queue(manifest, root / 'queue_status.json',
                   datetime.fromisoformat(manifest['deadline_utc']).timestamp())
print(json.dumps({'status': result['status']}), flush=True)
raise SystemExit(0 if result['status'] == 'completed' else 1)
