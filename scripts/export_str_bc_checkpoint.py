"""CPU-only BC Actor -> fresh native FlashSAC checkpoint bridge; no SAC updates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys


def export_checkpoint(bc_path: Path, output: Path, seed=0):
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo), str(repo / "scripts")]
    import gymnasium as gym
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from diagnose_str_memory import load_bc_checkpoint, predict_bc
    from flash_rl.agents import create_agent

    bc_path, output = bc_path.resolve(strict=True), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty export directory: {output}")
    source_model, payload = load_bc_checkpoint(bc_path, device="cpu")
    source_cfg = payload["config"]
    expected = {"mode": "current", "input_dim": 162, "obs_dim": 162,
                "action_dim": 29, "width": 128, "num_blocks": 2}
    if any(source_cfg.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Expected native current-observation BC Actor layout: {expected}")
    controls = payload.get("dataset_metadata", {}).get("resolved_config", {}).get("env", {}).get("task_cfg_overrides", {}).get("action", {})
    if controls and (controls.get("arm_moving_average") != .1 or controls.get("hand_moving_average") != .1):
        raise ValueError("BC teacher control settings are not the requested arm=.1 / hand=.1")
    OmegaConf.register_new_resolver("eval", lambda value: eval(value), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        cfg = hydra.compose(config_name="simtoolreal_state_teacher", overrides=[
            f"seed={seed}", "env.task_cfg_overrides.action.arm_moving_average=0.1",
            "env.task_cfg_overrides.action.hand_moving_average=0.1"])
    OmegaConf.resolve(cfg)
    training_config = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.clear_resolver("eval")
    cpu_config = OmegaConf.create(training_config["agent"])
    cpu_config.device_type = cpu_config.buffer_device_type = "cpu"
    cpu_config.buffer_max_length = cpu_config.buffer_min_length = cpu_config.sample_batch_size = 1
    cpu_config.use_compile = False
    # Enabled CPU GradScaler has the same serialized scalar fields as CUDA AMP.
    # No autocast, optimization or scaler update is performed in this exporter.
    cpu_config.use_amp = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(1, 324), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(1, 29), dtype=np.float32)
    env_info = {"actor_observation_size": (162,), "critic_observation_size": (162,),
                "critic_observation_offset": 162, "asymmetric_obs": True}
    agent = create_agent(obs_space, action_space, env_info, cpu_config)
    assert all(parameter.device.type == "cpu" for network in
               (agent._actor, agent._critic, agent._target_critic, agent._temperature)
               for parameter in network.network.parameters())
    assert agent._actor_observation_dim == agent._critic_observation_dim == 162
    assert agent._critic_observation_offset == 162 and agent._action_dim == 29
    actor = agent._actor.network
    actor.load_state_dict(payload["model_state_dict"], strict=True)
    actor.eval()
    if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("BC Actor contains nonfinite values")
    for name, value in payload["model_state_dict"].items():
        assert torch.equal(value.cpu(), actor.state_dict()[name]), name
    rng = np.random.default_rng(seed)
    probes = [np.zeros((4, 162), np.float32), np.ones((4, 162), np.float32),
              rng.normal(size=(128, 162)).astype(np.float32)]
    dataset_path = Path(payload.get("dataset_path", "__not_available__"))
    if dataset_path.is_file():
        with np.load(dataset_path, allow_pickle=False) as dataset:
            real = dataset["obs"]
            selected = rng.choice(len(real), size=min(512, len(real)), replace=False)
            probes.append(real[selected].astype(np.float32))
    observations = np.concatenate(probes)
    combined = np.concatenate([observations, observations], axis=-1)
    # Match the adapter's strided actor view so CPU BatchNorm takes the same
    # numerical kernel; contiguous versus strided inputs can differ by roundoff.
    expected_actions = predict_bc(source_model, payload, combined[:, None, :162]).cpu().numpy()
    contiguous_actions = predict_bc(source_model, payload, observations[:, None, :]).cpu().numpy()
    actions = agent.sample_actions(0, {"next_observation": combined}, training=False)
    assert np.array_equal(actions, expected_actions), f"BC -> Flash mismatch: {np.max(np.abs(actions - expected_actions))}"
    contiguous_error = float(np.max(np.abs(actions - contiguous_actions)))
    assert np.allclose(actions, contiguous_actions, rtol=0, atol=1e-4), contiguous_error
    assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1
    with torch.no_grad():
        _, std = actor.get_mean_and_std(torch.as_tensor(observations), training=False)
    assert torch.isfinite(std).all()
    for name, value in payload["model_state_dict"].items():
        assert torch.equal(value.cpu(), actor.state_dict()[name]), f"Probe mutated Actor/BN: {name}"
    critic_state = agent._critic.network.state_dict()
    for name, value in agent._target_critic.network.state_dict().items():
        assert torch.equal(value, critic_state[name]), f"Fresh target differs from initial Q: {name}"
    for network in (agent._actor, agent._critic, agent._temperature):
        assert not network.optimizer.state_dict()["state"], "Optimizer unexpectedly contains trained moments"
    assert agent._update_step == 0
    assert agent.reward_normalizer.G_r_max.item() == agent.reward_normalizer.G_rms.count.item() == 0
    # Native compiled training expects _orig_mod.* state-dict names. This eager
    # wrapper changes serialization names only; no Inductor/CUDA execution occurs.
    for network in (agent._actor, agent._critic, agent._target_critic, agent._temperature):
        network.network = torch.compile(network.network, backend="eager")
    agent.save(str(output))
    files = ["actor.pt", "critic.pt", "target_critic.pt", "temperature.pt", "reward_normalizer.pt", "agent_state.pt"]
    for name in files:
        assert (output / name).is_file()
    saved_actor = torch.load(output / "actor.pt", map_location="cpu", weights_only=True)
    assert all(key.startswith("_orig_mod.") for key in saved_actor["network_state_dict"])
    for name, value in payload["model_state_dict"].items():
        assert torch.equal(saved_actor["network_state_dict"]["_orig_mod." + name], value.cpu()), name
    for filename in ("actor.pt", "critic.pt", "temperature.pt"):
        saved = torch.load(output / filename, map_location="cpu", weights_only=True)
        assert not saved["optimizer_state_dict"]["state"] and saved["update_step"] == 0
    state = torch.load(output / "agent_state.pt", map_location="cpu", weights_only=True)
    assert state["update_step"] == 0 and state["grad_scaler_state_dict"]
    normalization = torch.load(output / "reward_normalizer.pt", map_location="cpu", weights_only=True)
    assert normalization["G_r_max"].item() == normalization["G_rms_count"].item() == 0
    assert normalization["G_rms_mean"].item() == 0 and normalization["G_rms_var"].item() == 1
    # Exercise the unmodified native six-file restore path on the CPU agent too.
    agent.load(str(output))
    restored_actions = agent.sample_actions(0, {"next_observation": combined}, training=False)
    assert np.array_equal(restored_actions, expected_actions)
    assert not torch.cuda.is_initialized(), "CPU exporter unexpectedly initialized CUDA"
    metadata = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "BC Actor warm-start with fresh SAC components; NOT SAC-trained",
        "source_bc_checkpoint": str(bc_path), "source_bc_sha256": hashlib.sha256(bc_path.read_bytes()).hexdigest(),
        "output": str(output), "seed": seed, "source_bc_config": source_cfg,
        "std_calibration": payload.get("std_calibration"),
        "training_config_reference": training_config,
        "cpu_export_only_overrides": {"device": "cpu", "replay_capacity": 1, "batch": 1,
                                      "initialization_compile": False, "serialization_wrapper": "torch.compile backend=eager"},
        "training_replay_capacity_unchanged": training_config["agent"]["buffer_max_length"],
        "observation_layout": {"replay": 324, "actor": 162, "critic": 162, "critic_offset": 162, "actions": 29},
        "controls": {"arm_moving_average": .1, "hand_moving_average": .1},
        "validation": {"actor_weights_and_BN_exact": True, "deterministic_actions_exact": True,
                       "probe_rows": len(observations), "max_action_abs_difference": 0,
                       "contiguous_vs_adapter_strided_input_max_difference": contiguous_error,
                       "contiguous_vs_strided_absolute_tolerance": 1e-4,
                       "fresh_optimizer_moments_empty": True, "fresh_critic_target_equal": True,
                       "native_six_file_load_passed": True, "cuda_initialized": False,
                       "fresh_normalizer_count": 0, "native_update_step": 0,
                       "pre_tanh_std_min_mean_max": [float(std.min()), float(std.mean()), float(std.max())]},
        "files": {name: {"size_bytes": (output / name).stat().st_size,
                          "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()} for name in files},
        "notes": ["Only the BC Actor weights and BN buffers were transferred. No old Critic, entropy temperature, reward statistics or optimizer moments were copied.",
                  ("The source std head was separately calibrated on expert observations; calibration metadata is copied above. Its fitted weights are preserved exactly, not SAC-trained; stochastic rollouts remain required."
                   if payload.get("std_calibration") else
                   "The source BC standard-deviation head was NOT trained by BC. Its random initialized weights are preserved exactly; stochastic rollouts must be tested before learning."),
                  "actor.pt already provides the small actor-only artifact consumed by diagnose_str_rollouts.py; no duplicate copy is needed.",
                  "Complete checkpoint uses native compiled _orig_mod.* names. Training must use agent.use_compile=true unless keys are explicitly converted.",
                  "Recommended training agent.load_optimizer=false retains freshly-created CUDA optimizer and AMP settings instead of the serialized CPU optimizer param groups (fused=false). All serialized moments are empty regardless.",
                  "The native trainer sets agent_loaded=true when agent_load_path is supplied, so after its first random step it acts with this Actor during replay collection instead of 100k random-action warmup.",
                  "This export does NOT add a Critic-only warmup or otherwise change when Actor/Critic updates begin. That design is separate."]}
    (output / "export_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "validation": metadata["validation"],
                      "files_bytes": sum(item["size_bytes"] for item in metadata["files"].values())}, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    export_checkpoint(args.bc_checkpoint, args.output_dir, args.seed)
