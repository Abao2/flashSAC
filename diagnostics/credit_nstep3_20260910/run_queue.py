"""Reuse the existing serial runner; no task or algorithm implementation fork."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[1] / 'scripts'))
from run_str_diagnostic_queue import run_queue

manifest = json.loads((root / 'manifest.json').read_text())
if datetime.now(timezone.utc) >= datetime.fromisoformat(manifest['no_new_training_after_utc']):
    raise SystemExit("No new training after the agreed 04:00 local cutoff")
result = run_queue(manifest, root / 'queue_status.json',
                   datetime.fromisoformat(manifest['deadline_utc']).timestamp())
print(json.dumps({'status': result['status']}), flush=True)
raise SystemExit(0 if result['status'] == 'completed' else 1)
