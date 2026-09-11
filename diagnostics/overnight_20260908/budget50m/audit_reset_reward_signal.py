"""Matched real reset observations: reward-channel-only CPU counterfactual, no rollout."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from flash_rl.agents.flashSAC.network import FlashSACActor
from summarize_str_control_ab import sha256


def moments(mu, sigma, noise):
    samples = (mu[None] + sigma[None] * noise).tanh()
    return {"pretanh_mean_abs_mean": float(mu.abs().mean()), "pretanh_sigma_mean": float(sigma.mean()),
            "pretanh_sigma_median": float(sigma.median()), "deterministic_action_abs_mean": float(mu.tanh().abs().mean()),
            "conditional_tanh_action_std_mean_8mc": float(samples.std(dim=0, correction=1).mean())}


def main():
    torch.set_num_threads(2)
    data_path = ROOT / "evaluations/arm1_hand01_seed0/step48830/deterministic/transitions.npz"
    meta = json.loads(data_path.with_name("metadata.json").read_text())
    data = np.load(data_path, allow_pickle=False)
    rows = np.flatnonzero((data["episode_id"] < 32) & (data["step_in_episode"] == 0))
    obs = data["obs"][rows].copy()
    channel = slice(*meta["state_field_slices"]["reward"])
    assert obs.shape == (32, 162) and (obs[:, channel] == 0).all()
    success_path = ROOT.parent / "warmstart/frozen_bc_std005_eigen/transitions.npz"
    successes = np.load(success_path, allow_pickle=False)
    terminal = (successes["terminated"] | successes["truncated"]) & successes["success"]
    terminal_raw = successes["rewards"][terminal]
    assert len(terminal_raw) == 128 and successes["terminated"][terminal].all()
    # Use the median of actually recorded successful terminal rewards, not a guessed bonus.
    raw_value = float(np.median(terminal_raw))
    encoded_value = raw_value * .01
    noise = torch.randn((8, 32, 29), generator=torch.Generator().manual_seed(9909))
    paths = {"native_hand01_50M": ROOT / "models/arm1_hand01/seed0/step48830/actor.pt",
             "selected_bc_control": ROOT.parent / "warmstart/bc_std005_eigen/actor.pt"}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu",
        "observation_source": str(data_path), "observation_source_sha256": sha256(data_path), "row_indices": rows.tolist(),
        "selection": "First 32 clean-reset initial states; same real observations and common MC Gaussian noise for both models.",
        "changed_field": "reward", "slice": [channel.start, channel.stop], "encoding": "obs_reward = raw.reward_buf * 0.01",
        "successful_reward_source": str(success_path), "successful_reward_source_sha256": sha256(success_path),
        "successful_terminal_count": len(terminal_raw), "selected_raw_reward_median": raw_value,
        "selected_encoded_reward": encoded_value,
        "observed_encoded_success_reward_range": [float(terminal_raw.min() * .01), float(terminal_raw.max() * .01)],
        "models": {}, "limitations": ["Single-action CPU counterfactual, no environment transition or success result.",
            "Selected BC normally uses arm .1/hand .1; this matched-state actor-only control has no controller application.",
            "Conditional tanh action std is an 8-sample-per-state estimate, not behavior trajectory variance or filtered motor noise.",
            "A successful reward carried into reset can change actions without explaining the actual train/eval mismatch."]}
    for name, path in paths.items():
        saved = torch.load(path, map_location="cpu", weights_only=True)["network_state_dict"]
        state = {key.removeprefix("_orig_mod."): value for key, value in saved.items()}
        actor = FlashSACActor(num_blocks=2, input_dim=162, hidden_dim=128, action_dim=29)
        actor.load_state_dict(state, strict=True)
        actor.eval()
        with torch.no_grad():
            mu, sigma = actor.get_mean_and_std(torch.from_numpy(obs), training=False)
            changed = obs.copy()
            changed[:, channel] = encoded_value
            assert np.array_equal(changed[:, :channel.start], obs[:, :channel.start])
            new_mu, new_sigma = actor.get_mean_and_std(torch.from_numpy(changed), training=False)
        assert all(torch.equal(value, actor.state_dict()[key]) for key, value in state.items())
        delta = (new_mu.tanh() - mu.tanh()).abs()
        item = {"actor_path": str(path), "actor_sha256": sha256(path), "network_unchanged": True,
            "fresh_reward_zero": moments(mu, sigma, noise), "reward_replaced_with_success_median": moments(new_mu, new_sigma, noise),
            "mean_absolute_pretanh_mu_delta": float((new_mu - mu).abs().mean()),
            "mean_absolute_pretanh_sigma_delta": float((new_sigma - sigma).abs().mean()),
            "max_absolute_pretanh_sigma_delta": float((new_sigma - sigma).abs().max()),
            "mean_absolute_deterministic_action_delta": float(delta.mean()), "max_absolute_deterministic_action_delta": float(delta.max()),
            "arm_mean_absolute_deterministic_delta": float(delta[:, :7].mean()), "hand_mean_absolute_deterministic_delta": float(delta[:, 7:].mean())}
        if name == "native_hand01_50M":
            item["cpu_reference_vs_recorded_eval_action_max_error"] = float(np.max(np.abs(mu.tanh().numpy() - data["actions"][rows])))
        report["models"][name] = item
    (ROOT / "reset_reward_signal_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["models"]))


if __name__ == "__main__":
    main()
