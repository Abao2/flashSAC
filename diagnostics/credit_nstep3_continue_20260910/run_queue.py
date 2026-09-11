"""Resume queue using the already tested native serial runner."""
from datetime import datetime
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[1] / 'scripts'))
from run_str_diagnostic_queue import run_queue

manifest = json.loads((root / 'manifest.json').read_text())
result = run_queue(manifest, root / 'queue_status.json',
                   datetime.fromisoformat(manifest['deadline_utc']).timestamp())
print(json.dumps({'status': result['status']}), flush=True)
raise SystemExit(0 if result['status'] == 'completed' else 1)
