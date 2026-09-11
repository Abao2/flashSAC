"""CPU checks for episode safety, matched controls and loadable native-actor BC."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location("str_memory_probe", Path(__file__).resolve().parents[1] / "scripts/diagnose_str_memory.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def dataset():
    rng = np.random.default_rng(12)
    # Time-major env-interleaved ordering: not contiguous per episode in the file.
    obs = rng.normal(size=(24, 162)).astype(np.float32)
    return {"obs": obs, "next_obs": obs + 0.01, "actions": np.tanh(obs[:, :29]),
            "rewards": obs[:, 0], "episode_id": np.tile(np.arange(4), 6),
            "step_in_episode": np.repeat(np.arange(6), 4),
            "terminated": np.zeros(24, bool), "truncated": np.arange(24) >= 20}


def test_history_split_and_no_future_leakage():
    data = dataset()
    history = probe.history_indices(data, 3)
    assert history[0].tolist() == [0, 0, 0]
    assert history[5].tolist() == [1, 1, 5]
    assert history[13].tolist() == [5, 9, 13]
    assert np.all(data["episode_id"][history] == data["episode_id"][:, None])
    assert np.all(data["step_in_episode"][history] <= data["step_in_episode"][:, None])
    train, val = probe.episode_split(data["episode_id"], 4, 0.25)
    assert set(data["episode_id"][train]).isdisjoint(data["episode_id"][val])
    repeated = probe.selected_inputs(data, history, np.array([13]), "repeat")
    actual = probe.selected_inputs(data, history, np.array([13]), "history")
    assert repeated.shape == actual.shape == (1, 486)
    assert np.allclose(repeated.reshape(3, 162), data["obs"][13])
    assert np.allclose(actual.reshape(3, 162)[-1], data["obs"][13])
    data["terminated"][5] = True
    with pytest.raises(ValueError, match="reused"):
        probe.history_indices(data, 3)


def test_missing_steps_restart_padding():
    data = dataset()
    keep = np.arange(24) != 5
    data = {name: value[keep] for name, value in data.items()}
    history = probe.history_indices(data, 3)
    row = np.flatnonzero((data["episode_id"] == 1) & (data["step_in_episode"] == 2))[0]
    assert (history[row] == row).all()


def test_cpu_training_roundtrip_and_terminal_exclusion(tmp_path):
    torch.set_num_threads(1)
    data = dataset()
    args = SimpleNamespace(seed=0, validation_fraction=0.25, bc_only_success=False, bc_action_key="actions",
                           device="cpu", window=3, width=16, num_blocks=1,
                           learning_rate=3e-4, updates=2, batch_size=8,
                           max_eval_rows=100, log_every=1, output=tmp_path,
                           dataset=tmp_path / "data.npz")
    np.savez(args.dataset, **data, metadata_json=json.dumps({"obs_fields": probe.OBS_FIELDS}))
    loaded, metadata = probe.load_dataset(args.dataset)
    history = probe.history_indices(loaded, args.window)
    result = probe.run_probe(args, loaded, metadata, history, "bc", "history")
    assert result["status"] == "complete"
    model, payload = probe.load_bc_checkpoint(result["checkpoint"])
    x = torch.as_tensor(loaded["obs"][history[:3]])
    actual = probe.predict_bc(model, payload, x)
    expected = model.get_mean_and_std(x.flatten(1), False)[0].tanh()
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all() and actual.abs().max() <= 1
    pred = probe.run_probe(args, loaded, metadata, history, "prediction", "repeat")
    assert pred["excluded_terminal_prediction_rows"] == 4
    assert pred["train_rows"] + pred["validation_rows"] == 20
    assert np.isfinite(pred["validation"]["normalized_mse"])
    args.bc_action_key = "teacher_actions"
    with pytest.raises(ValueError, match="Missing BC label"):
        probe.run_probe(args, loaded, metadata, history, "bc", "current")
    # BC labels differ from behavior; prediction continues to use actual actions.
    loaded["teacher_actions"] = np.zeros_like(loaded["actions"])
    labeled = probe.run_probe(args, loaded, metadata, history, "bc", "current")
    assert labeled["config"]["bc_action_key"] == "teacher_actions"
    assert labeled["train"]["train_mean_baseline_normalized_mse"] == 0
    pred_labeled = probe.run_probe(args, loaded, metadata, history, "prediction", "repeat")
    assert pred_labeled["config"]["prediction_action_key"] == "actions"
    assert pred_labeled["validation"]["normalized_mse"] == pred["validation"]["normalized_mse"]
    args.bc_only_success = True
    skipped = probe.run_probe(args, loaded, metadata, history, "bc", "current")
    assert skipped["status"] == "skipped"


def test_state_fields_metadata_alias(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, **dataset(), metadata_json=json.dumps({"state_fields": probe.OBS_FIELDS}))
    assert probe.load_dataset(path)[1]["state_fields"] == probe.OBS_FIELDS
    np.savez(path, **dataset(), metadata_json=json.dumps({"state_fields": list(reversed(probe.OBS_FIELDS))}))
    with pytest.raises(ValueError, match="state_fields"):
        probe.load_dataset(path)


def test_current_actor_ignores_all_earlier_frames():
    torch.set_num_threads(1)
    config = {"task": "bc", "mode": "current", "input_dim": 162, "window": 8,
              "width": 16, "num_blocks": 1}
    model = probe.create_model(config)
    payload = {"config": config}
    history = torch.randn(8, 8, 162)
    changed = history.clone()
    changed[:, :-1] = torch.randn_like(changed[:, :-1]) * 1000
    expected = probe.predict_bc(model, payload, history)
    torch.testing.assert_close(probe.predict_bc(model, payload, changed), expected, rtol=0, atol=0)
    torch.testing.assert_close(probe.predict_bc(model, payload, history[:, -1:]), expected, rtol=0, atol=0)
