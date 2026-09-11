"""Clone a fresh native checkpoint on CPU; only its loaded log-temperature changes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch

FILES = ("actor.pt", "critic.pt", "target_critic.pt", "temperature.pt", "reward_normalizer.pt", "agent_state.pt")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_payload(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_payload(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_payload(x, y) for x, y in zip(a, b))
    return a == b


def clone_checkpoint(source, output, alpha):
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Initial alpha must be positive and finite")
    if output == source or source in output.parents:
        raise ValueError("Export must not modify or be nested inside source checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {output}")
    before = {name: sha256(source / name) for name in FILES}
    original = torch.load(source / "temperature.pt", map_location="cpu", weights_only=True)
    state = original["network_state_dict"]
    if list(state) != ["_orig_mod.log_temp"] or state["_orig_mod.log_temp"].shape != (1,):
        raise ValueError("Expected native compiled scalar log-temperature checkpoint")
    if original["update_step"] != 0 or original["optimizer_state_dict"]["state"]:
        raise ValueError("Only fresh initial temperature checkpoints are supported")
    agent_state = torch.load(source / "agent_state.pt", map_location="cpu", weights_only=True)
    if agent_state["update_step"] != 0:
        raise ValueError("Expected fresh agent update counter 0")
    desired = torch.full_like(state["_orig_mod.log_temp"], math.log(alpha))
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(source / name, output / name)
    if not torch.equal(desired, state["_orig_mod.log_temp"]):
        changed = torch.load(source / "temperature.pt", map_location="cpu", weights_only=True)
        changed["network_state_dict"]["_orig_mod.log_temp"].copy_(desired)
        torch.save(changed, output / "temperature.pt")
    after = {name: sha256(output / name) for name in FILES}
    assert all(before[name] == after[name] for name in FILES if name != "temperature.pt")
    restored = torch.load(output / "temperature.pt", map_location="cpu", weights_only=True)
    actual = float(restored["network_state_dict"]["_orig_mod.log_temp"].exp().item())
    assert math.isclose(actual, alpha, rel_tol=1e-6)
    restored["network_state_dict"]["_orig_mod.log_temp"] = state["_orig_mod.log_temp"]
    assert same_payload(original, restored), "A non-log-temperature value changed"
    assert before == {name: sha256(source / name) for name in FILES}, "Source changed"
    assert not torch.cuda.is_initialized(), "CPU clone unexpectedly initialized CUDA"
    report = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "output": str(output), "requested_alpha": alpha, "actual_alpha": actual,
        "source_alpha": float(state["_orig_mod.log_temp"].exp().item()),
        "changed_tensor": "temperature.pt:network_state_dict._orig_mod.log_temp",
        "non_temperature_files_bitwise_identical": True, "temperature_other_fields_exact": True,
        "source_preserved": True, "source_sha256": before, "output_sha256": after,
        "interpretation": "Adaptive initial-alpha intervention. Native temperature updates remain enabled after any critic-only gate; alpha affects both actor objective and critic soft targets, not only actor loss."}
    (output / "alpha_initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = clone_checkpoint(args.source, args.output, args.alpha)
    print(json.dumps({k: result[k] for k in ("output", "requested_alpha", "actual_alpha", "source_preserved")}, indent=2))
