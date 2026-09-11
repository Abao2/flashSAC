"""Frozen-policy STR diagnostics; writes transitions and honest geometric proxies.

Run --help without Isaac. Defaults preserve the state-teacher task/reward/reset.
The raw entry bypasses wrapper.step, not the shared Isaac environment creation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def field_slices(names, sizes):
    fields, offset = {}, 0
    for name in names:
        fields[name] = slice(offset, offset + sizes[name])
        offset += sizes[name]
    return fields, offset


def extract_fields(state, fields, names):
    import numpy as np

    return np.concatenate([state[..., fields[name]] for name in names], axis=-1)


def terminal_next_observation(observation, done, info):
    """Never substitute an auto-reset initial observation for the final state."""
    import numpy as np

    result = np.asarray(observation).copy()
    if np.any(done):
        final = info.get("final_obs")
        if final is None or np.shape(final) != result.shape:
            raise RuntimeError("Missing or incorrectly shaped pre-reset final_obs")
        mask = info.get("_final_obs")
        if mask is not None and not np.all(np.asarray(mask)[done]):
            raise RuntimeError("final_obs validity mask excludes a done environment")
        result[done] = final[done]
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite transition next observation")
    return result


def geometric_metrics(state, fields, goal, table_top, object_base_size):
    """Eraser box geometry only: proximity and clearance are NOT contact tests."""
    import numpy as np

    quat = state[:, fields["object_rot"]]  # xyzw
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    palm = state[:, fields["palm_pos"]]
    obj = goal[:3] + state[:, fields["keypoints_rel_goal"]].reshape(-1, 4, 3).mean(axis=1)
    half_size = state[:, fields["object_scales"]] * object_base_size / 2
    x, y, z, w = quat.T
    rotation_z_row = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                               1 - 2 * (x * x + y * y)], axis=-1)
    bottom = obj[:, 2] - (np.abs(rotation_z_row) * half_size).sum(axis=-1)
    tips = palm[:, None, :] + state[:, fields["fingertip_pos_rel_palm"]].reshape(-1, 5, 3)
    # Inverse quaternion rotates world-relative fingertip locations into the box.
    delta = tips - obj[:, None, :]
    q_xyz = -quat[:, None, :3]
    local = delta + 2 * np.cross(q_xyz, np.cross(q_xyz, delta) + quat[:, None, 3:] * delta)
    tip_distance = np.linalg.norm(np.maximum(np.abs(local) - half_size[:, None, :], 0), axis=-1)
    return {"object_pos": obj, "palm_pos": palm,
            "object_z": obj[:, 2], "object_bottom_clearance_m": bottom - table_top,
            "palm_object_distance_m": np.linalg.norm(obj - palm, axis=-1),
            "tip_box_distance_m": tip_distance.min(axis=-1)}


def pre_action_phases(state, fields, goal, table_top, object_base_size, initial_z, steps, goal_tolerance):
    """Labels use ONLY s_t and its elapsed step count, never a_t/r_t/s_(t+1)."""
    import numpy as np

    geometry = geometric_metrics(state, fields, goal, table_top, object_base_size)
    phase = np.full(len(state), "far_from_object", dtype="U32")
    phase[geometry["tip_box_distance_m"] <= .02] = "near_object_proxy"
    off = geometry["object_bottom_clearance_m"] > .005
    phase[off] = "off_table_proxy"
    phase[off & (geometry["object_z"] - initial_z > .02)] = "lifted_proxy"
    phase[np.linalg.norm(geometry["object_pos"] - goal[:3], axis=-1) < goal_tolerance] = "near_goal_center_proxy"
    phase[np.asarray(steps) == 0] = "reset"
    return phase


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenPolicy:
    def __init__(self, args, cfg, env, info, state_fields, official_fields):
        import numpy as np
        import torch

        self.args, self.cfg = args, cfg
        self.state_fields, self.official_fields = state_fields, official_fields
        self.num_envs, self.device = args.num_envs, env.device
        self.player = self.agent = self.model = None
        self.bc_payload = self.bc_history = self.bc_predict = None
        self.teacher_actions = None
        self.teacher_hook = None
        self.fixed_noise = torch.zeros((args.num_envs, 29), device=self.device)
        self.noise_count = np.zeros(args.num_envs, dtype=int)
        if args.policy == "official":
            from deployment.rl_player import RlPlayer

            self.player = RlPlayer(140, 29, str(args.official_config), str(args.checkpoint),
                                   self.device, num_envs=args.num_envs)
            self.player.player.init_rnn()
            self.model = self.player.player.model
            player = self.player.player
            assert torch.all(player.actions_low == -1) and torch.all(player.actions_high == 1)
            self.teacher_hook = self.model.register_forward_hook(self.capture_teacher_actions)
        elif args.policy == "flash":
            from flash_rl.agents import create_agent

            self.agent = create_agent(env.observation_space, env.action_space, info, cfg.agent)
            saved = torch.load(args.checkpoint / "actor.pt", map_location="cpu", weights_only=True)
            weights = {key.removeprefix("_orig_mod."): value
                       for key, value in saved["network_state_dict"].items()}
            self.model = self.agent._actor.network
            self.model.load_state_dict(weights, strict=True)
        elif args.policy == "bc":
            from diagnose_str_memory import load_bc_checkpoint, predict_bc

            self.model, self.bc_payload = load_bc_checkpoint(args.checkpoint, device=self.device)
            self.bc_predict = predict_bc
        if self.model is not None:
            self.model.eval()
            self.model.requires_grad_(False)
        self.digest_before = model_digest(self.model) if self.model is not None else None

    def capture_teacher_actions(self, model, inputs, output):
        """Read the EXISTING official forward: never call model or advance RNN here."""
        player = self.player.player
        mean = output["mus"].detach()
        if player.clip_actions:
            mean = (mean.clamp(-1, 1) * (player.actions_high - player.actions_low) / 2
                    + (player.actions_high + player.actions_low) / 2)
        self.teacher_actions = mean.reshape(self.num_envs, 29).cpu().numpy().copy()

    def actions(self, obs, interaction_step):
        import numpy as np
        import torch

        if self.args.policy == "zero":
            return np.zeros((self.num_envs, 29), dtype=np.float32)
        with torch.no_grad():
            if self.player is not None:
                inputs = extract_fields(obs[:, :162], self.state_fields, self.official_fields)
                result = self.player.get_normalized_action(
                    torch.as_tensor(inputs, device=self.device),
                    deterministic_actions=self.args.sampling == "deterministic")
            elif self.bc_payload is not None:
                k = int(self.bc_payload["config"]["window"])
                current = obs[:, :162]
                if self.bc_history is None:
                    self.bc_history = np.repeat(current[:, None, :], k, axis=1)
                else:
                    self.bc_history = np.roll(self.bc_history, -1, axis=1)
                    self.bc_history[:, -1] = current
                    new_episode = self.noise_count == 0
                    self.bc_history[new_episode] = current[new_episode, None, :]
                self.noise_count += 1
                result = self.bc_predict(self.model, self.bc_payload, self.bc_history)
                if isinstance(result, np.ndarray):
                    return result
            elif self.args.sampling == "deterministic":
                return self.agent.sample_actions(interaction_step, {"next_observation": obs}, training=False)
            elif self.args.noise_repeat == 0:
                from flash_rl.agents.flashSAC.agent import _sample_flashsac_actions

                agent = self.agent
                agent._cached_noise, result, agent._cur_noise_repeat_count, agent._cur_noise_repeat_n = (
                    _sample_flashsac_actions(
                        actor=agent._actor, noise=agent._cached_noise,
                        observations=torch.as_tensor(obs[:, :162], device=self.device),
                        temperature=self.args.noise_multiplier,
                        cur_count=agent._cur_noise_repeat_count, cur_n=agent._cur_noise_repeat_n,
                        zeta_cdf=agent._zeta_cdf))
            else:
                mean, std = self.model.get_mean_and_std(
                    torch.as_tensor(obs[:, :162], device=self.device), training=False)
                refresh = self.noise_count % self.args.noise_repeat == 0
                self.fixed_noise[refresh] = torch.randn((int(refresh.sum()), 29), device=self.device)
                result = torch.tanh(mean + std * self.fixed_noise * self.args.noise_multiplier)
                self.noise_count += 1
            return result.detach().cpu().numpy()

    def reset_done(self, done):
        import torch

        self.noise_count[done] = 0
        if self.player is not None and self.player.player.is_rnn:
            ids = torch.as_tensor(done, device=self.device)
            for state in self.player.player.states:
                state[:, ids, :] = 0
        # Native Flash noise repetition deliberately matches training: a global
        # repeat counter is not reset when an individual environment terminates.


def numerical_settings(torch_module, enable=False):
    """Opt in to train.py's precision bundle; reading defaults never changes them."""
    if enable:
        torch_module.backends.cudnn.benchmark = True
        torch_module.backends.cuda.matmul.allow_tf32 = True
        torch_module.backends.cudnn.allow_tf32 = True
        torch_module.set_float32_matmul_precision("high")
    return {
        "cuda_matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
    }


def create_env(args, repo, str_root):
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from flash_rl.envs.isaaclab import make_isaaclab_env

    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        cfg = hydra.compose(config_name=args.config_name,
                            overrides=[f"num_train_envs={args.num_envs}", f"seed={args.seed}", *args.override])
    cfg.agent.buffer_max_length = cfg.agent.buffer_min_length = cfg.agent.sample_batch_size = 1
    cfg.agent.use_compile = cfg.agent.load_optimizer = cfg.agent.load_reward_normalizer = False
    OmegaConf.resolve(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.clear_resolver("eval")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.training_numerics:
        numerical_settings(torch, enable=True)
    env = make_isaaclab_env(
        env_name=cfg.env.env_name, num_envs=args.num_envs, seed=args.seed, headless=True,
        device=cfg.env.device, action_bounds=cfg.env.action_bounds,
        registration_modules=cfg.env.registration_modules,
        env_cfg_yaml_entry_point=cfg.env.env_cfg_yaml_entry_point,
        task_cfg_overrides=cfg.env.task_cfg_overrides,
        bootstrap_timeouts=cfg.env.bootstrap_timeouts, success_path=cfg.env.success_path)
    return env, cfg, resolved


def step_env(env, actions, entry):
    if entry == "wrapper":
        return env.step(actions)
    import numpy as np
    import torch
    from flash_rl.envs.isaaclab import recursive_to_numpy

    # Same normalized-action contract as the adapter, without its step/statistics.
    tensor = torch.as_tensor(actions, device=env.device)
    if env.action_bounds is not None:
        tensor = tensor.clamp(-1, 1) * env.action_bounds
    obs, reward, terminated, truncated, info = env.envs.step(tensor)
    converted = recursive_to_numpy(dict(info))
    if isinstance(converted.get("final_obs"), dict):
        final = converted["final_obs"]
        converted["final_obs"] = np.concatenate([final["policy"], final["critic"]], axis=-1)
    combined = torch.cat([obs["policy"], obs["critic"]], dim=-1)
    return (combined.cpu().numpy(), reward.cpu().numpy(), terminated.cpu().numpy(),
            truncated.cpu().numpy(), converted)


def rollout_action(policy, env, obs, vector_step, training_startup=False):
    """Training's first interaction is random, without advancing actor/noise state."""
    if training_startup and vector_step == 0:
        return env.action_space.sample()
    return policy.actions(obs, vector_step)


def self_test():
    import numpy as np
    import torch
    from types import SimpleNamespace

    fields, size = field_slices(["a", "b", "c"], {"a": 2, "b": 1, "c": 3})
    assert size == 6
    source = np.arange(12).reshape(2, 6)
    assert np.array_equal(extract_fields(source, fields, ["c", "a"]), source[:, [3, 4, 5, 0, 1]])
    done = np.array([True, False])
    final = source + 100
    actual = terminal_next_observation(source, done, {"final_obs": final, "_final_obs": done})
    assert np.array_equal(actual[0], final[0]) and np.array_equal(actual[1], source[1])
    try:
        terminal_next_observation(source, done, {})
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing final_obs accepted")
    geometry_fields, size = field_slices(
        ["object_rot", "palm_pos", "keypoints_rel_goal", "object_scales", "fingertip_pos_rel_palm"],
        {"object_rot": 4, "palm_pos": 3, "keypoints_rel_goal": 12,
         "object_scales": 3, "fingertip_pos_rel_palm": 15})
    state = np.zeros((2, size))
    state[:, geometry_fields["object_rot"]] = [0, 0, 0, 1]
    state[:, geometry_fields["palm_pos"]] = [0, 0, .65]
    state[:, geometry_fields["object_scales"]] = [4, 2, 1]  # .16 x .08 x .04 m
    goal = np.array([0, 0, .55, 1, 0, 0, 0])
    geom = geometric_metrics(state, geometry_fields, goal, .53, .04)
    assert np.allclose(geom["object_bottom_clearance_m"], 0)
    assert np.allclose(geom["tip_box_distance_m"], .08)
    # 90 degree rotation about X changes vertical half-extent .02 -> .04.
    state[1, geometry_fields["object_rot"]] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    geom = geometric_metrics(state, geometry_fields, goal, .53, .04)
    assert np.allclose(geom["object_bottom_clearance_m"], [0, -.02])
    assert np.allclose(geom["tip_box_distance_m"], [.08, .06])
    labels = pre_action_phases(state, geometry_fields, goal, .53, .04, .55, np.array([0, 7]), .03)
    assert labels.tolist() == ["reset", "near_goal_center_proxy"]
    probe = FrozenPolicy.__new__(FrozenPolicy)
    probe.num_envs = 2
    probe.player = SimpleNamespace(player=SimpleNamespace(
        clip_actions=True, actions_low=-torch.ones(29), actions_high=torch.ones(29)))

    class OneForward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, means):
            self.calls += 1
            return {"mus": means, "actions": means + .123}

    model = OneForward()
    model.register_forward_hook(probe.capture_teacher_actions)
    means = torch.linspace(-2, 2, 58).reshape(2, 29)
    output = model(means)
    assert model.calls == 1
    assert torch.equal(output["actions"], means + .123)
    assert np.array_equal(probe.teacher_actions, means.clamp(-1, 1).numpy())
    print("SELF_TEST_OK: fields/finalobs/geometry/phases and teacher hook preserves one forward/executed actions")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["official", "flash", "zero", "bc"], default="flash")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--official-config", type=Path)
    parser.add_argument("--sampling", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--noise-multiplier", type=float, default=1.0,
                        help="Flash pre-tanh standard-deviation multiplier; not SAC alpha")
    parser.add_argument("--noise-repeat", type=int, default=0,
                        help="0: native Flash zeta/global repetition; >0: fixed per-env noise persistence")
    parser.add_argument("--entry", choices=["wrapper", "raw"], default="wrapper")
    parser.add_argument("--training-startup", action="store_true",
                        help="Frozen Flash only: training's two randomized-horizon resets and one random first action")
    parser.add_argument("--training-numerics", action="store_true",
                        help="Opt in to train.py's TF32/high-matmul/cuDNN benchmark bundle; no compile or inference-mode change")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Maximum vector steps; 0 derives a bound from episodes and task horizon")
    parser.add_argument("--config-name", default="simtoolreal_state_teacher")
    parser.add_argument("--override", action="append", default=[], help="Explicit Hydra override, recorded in metadata")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-dataset", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.episodes < 1 or args.num_envs < 1 or args.max_steps < 0:
        parser.error("Positive episodes/num-envs and nonnegative max-steps required")
    if args.noise_multiplier < 0 or not math.isfinite(args.noise_multiplier) or args.noise_repeat < 0:
        parser.error("Noise multiplier/repeat must be nonnegative and finite")
    if args.policy != "flash" and (args.noise_repeat or args.noise_multiplier != 1):
        parser.error("Noise overrides apply only to Flash")
    if args.training_startup and (args.policy != "flash" or args.entry != "wrapper"):
        parser.error("--training-startup requires --policy flash --entry wrapper")
    if args.policy in ("flash", "bc") and args.checkpoint is None:
        parser.error("Flash/BC --checkpoint is required")
    if args.policy == "bc" and args.sampling != "deterministic":
        parser.error("BC inference currently supports deterministic actions only")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be absent or empty; refusing to overwrite")
    return args


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    repo = Path(__file__).resolve().parents[1]
    str_root = Path(os.environ.get("SIMTOOLREAL_ROOT", repo.parent / "simtoolreal")).resolve()
    sys.path[:0] = [str(repo), str(str_root), str(str_root / "rl_games")]
    args.output_dir = args.output_dir.expanduser().resolve()
    args.official_config = (args.official_config or str_root / "pretrained_policy/config.yaml").resolve()
    if args.policy == "official" and args.checkpoint is None:
        args.checkpoint = str_root / "pretrained_policy/model.pth"
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        check_file = args.checkpoint / "actor.pt" if args.policy == "flash" else args.checkpoint
        if not check_file.is_file():
            raise FileNotFoundError(check_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(str_root)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    import numpy as np
    import torch
    import yaml
    from eval_str_state_teacher import pose_errors

    started, started_iso = time.monotonic(), datetime.now(timezone.utc).isoformat()
    env, cfg, resolved = create_env(args, repo, str_root)
    try:
        from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import OBS_FIELD_SIZES, KEYPOINT_CORNERS

        obs, info = env.reset(random_start_init=args.training_startup)
        raw = env.envs.unwrapped
        fields, state_dim = field_slices(raw.cfg.obs.state_list, OBS_FIELD_SIZES)
        assert state_dim == 162 and tuple(info["actor_observation_size"]) == (162,)
        assert tuple(raw.cfg.obs.obs_list) == tuple(raw.cfg.obs.state_list)
        assert obs.shape == (args.num_envs, 324)
        assert cfg.env.bootstrap_timeouts, "Diagnostics require separate timeouts and true terminal states"
        assert not raw.cfg.domain_randomization.use_obs_delay
        assert not raw.cfg.domain_randomization.use_object_state_delay_noise
        with args.official_config.open() as stream:
            official_fields = yaml.safe_load(stream)["task"]["env"]["obsList"]
        assert sum(OBS_FIELD_SIZES[key] for key in official_fields) == 140
        policy = FrozenPolicy(args, cfg, env, info, fields, official_fields)
        startup = None
        if args.training_startup:
            # train.py resets once for env_info before agent creation, then again
            # immediately before collection. No extra physics tick or actor call.
            obs, info = env.reset(random_start_init=True)
            startup = {
                "random_action_vector_steps": [0],
                "initial_episode_ids": list(range(min(args.num_envs, args.episodes))),
                "initial_episode_length_buf": raw.episode_length_buf.detach().cpu().tolist(),
                "initial_observed_progress": obs[:, fields["progress"]].reshape(-1).tolist(),
                "action_space_rng_state_before_first_sample": env.action_space.np_random.bit_generator.state,
                "rng_note": "Same action_space.sample API as train.py. IsaacLab Box RNG is not explicitly seeded there; state is recorded, not reseeded. No claim of bitwise matching an existing training run.",
            }
        goal = np.asarray(raw.cfg.reset.fixed_goal_pose, dtype=float)
        initial_z = float(raw.cfg.reset.fixed_start_pose[2])
        assert goal.shape == (7,) and raw.cfg.termination.max_consecutive_successes == 1
        assert list(raw.cfg.assets.handle_head_types) == ["eraser"], "Geometry proxy assumes box eraser"
        offsets = np.asarray(KEYPOINT_CORNERS) * np.asarray(raw.cfg.reward.fixed_size) * raw.cfg.reward.keypoint_scale / 2
        assert raw.cfg.reward.fixed_size_keypoint_reward
        import xml.etree.ElementTree as ET

        table_collision = ET.parse(str_root / raw.cfg.assets.table_urdf).find(".//collision")
        table_size = [float(value) for value in table_collision.find("geometry/box").get("size").split()]
        table_origin = table_collision.find("origin")
        table_origin_z = 0 if table_origin is None else float(table_origin.get("xyz", "0 0 0").split()[2])
        table_top = float(raw.cfg.reset.table_reset_z + table_origin_z + table_size[2] / 2)
        object_base_size = float(raw.cfg.reward.object_base_size)
        dt = float(raw.step_dt)
        hold_steps = max(2, math.ceil(.2 / dt))
        max_steps = args.max_steps or math.ceil(args.episodes / args.num_envs) * int(raw.max_episode_length)
        metadata = {
            "started_utc": started_iso, "arguments": {key: str(value) if isinstance(value, Path) else value
                                                       for key, value in vars(args).items()},
            "resolved_config": resolved, "state_fields": list(raw.cfg.obs.state_list),
            "obs_fields": list(raw.cfg.obs.state_list),
            "state_field_slices": {name: [value.start, value.stop] for name, value in fields.items()},
            "next_obs_is_pre_reset": True, "successful_episode_ids": [],
            "training_startup": startup,
            "torch_numerics_after_env_init": numerical_settings(torch),
            "official_actor_fields": official_fields, "state_dim": 162, "action_dim": 29,
            "dt_seconds": dt, "table_top_m": table_top, "object_base_size_m": object_base_size,
            "initial_object_z_m": initial_z,
            "initial_actor_critic_max_abs_difference": float(np.abs(obs[:, :162] - obs[:, 162:]).max()),
            "phase_semantics": "Pre-action s_t geometry + step_in_episode only; no actions/reward/next_obs/episode outcome.",
            "phase_priority_low_to_high": ["far_from_object", "near_object_proxy", "off_table_proxy",
                                          "lifted_proxy", "near_goal_center_proxy", "reset"],
            "action_label_semantics": {
                "actions": "Actual policy action passed to env.step for this transition.",
                "teacher_actions": ("Official deterministic mean from the SAME single model forward and same RNN state "
                                    "that generated the executed action, with the original player's clamp/rescale; "
                                    "not a second policy call. Available for official policy only.")},
            "fixed_goal_pose_wxyz": goal.tolist(), "model_digest_before": policy.digest_before,
            "thresholds": {"near_object_proxy_m": .02, "off_table_box_clearance_m": .005,
                           "sustained_lift_near_palm_proxy": {"height_above_reset_m": .02,
                               "palm_center_distance_m": .2, "relative_world_displacement_per_step_m": .005,
                               "consecutive_steps": hold_steps},
                           "task_lift_flag": "sticky object-center lift >0.10 m above reset",
                           "success_keypoint_m": float(raw._current_success_tolerance * raw.cfg.reward.keypoint_scale)},
            "notes": ["No network/optimizer updates; this is frozen-policy sampling, not training.",
                      "Fixed reset is one task; episode noise seeds are one reproducible vector RNG stream.",
                      "Proximity/clearance/sustained-lift proxies are NOT verified contacts or stable grasp.",
                      "Dataset next_obs substitutes pre-reset final_obs at every terminated/truncated step.",
                      "Raw entry bypasses wrapper.step but uses the same Isaac environment initialization.",
                      "Native stochastic Flash uses original global zeta repetition; fixed repeat is per-env.",
                      "No extra reset physics tick is inserted; both entries begin from identical reset API timing."]}
        if startup is not None:
            metadata["notes"].append("Training-startup gate: randomized initial horizons plus one random vector action before frozen policy takeover; only the initial batch has this startup shift, not subsequent auto-resets.")
            metadata["action_label_semantics"]["actions"] = "Actual env.step action; vector step 0 is action_space.sample, all later steps are frozen-policy actions."
        if args.training_numerics:
            metadata["notes"].append("Training-numerics explicitly enables the train.py TF32/high-matmul/cuDNN benchmark bundle before environment creation; compile and inference training=False remain unchanged. This is a bundle intervention, not proof of one flag's causal effect.")
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))

        n = args.num_envs
        ids = np.arange(n, dtype=np.int64)
        ids[ids >= args.episodes] = -1
        next_id = min(n, args.episodes)
        lengths = np.zeros(n, dtype=np.int32)
        returns = np.zeros(n)
        discounted = np.zeros(n)
        ever_lift = np.zeros(n, dtype=bool)
        ever_near = np.zeros(n, dtype=bool)
        ever_off = np.zeros(n, dtype=bool)
        ever_hold = np.zeros(n, dtype=bool)
        dropped = np.zeros(n, dtype=bool)
        max_height = np.zeros(n)
        hold_count = np.zeros(n, dtype=np.int32)
        initial_geometry = geometric_metrics(obs[:, :162], fields, goal, table_top, object_base_size)
        previous_relative = initial_geometry["object_pos"] - initial_geometry["palm_pos"]
        previous_actions = np.zeros((n, 29), dtype=np.float32)
        reward_components = {}
        rows, chunks = [], {}
        action_abs, action_delta, action_clip, action_count = np.zeros(2), np.zeros(2), np.zeros(2), 0

        with (args.output_dir / "episodes.jsonl").open("w", buffering=1) as episode_file:
            for vector_step in range(max_steps):
                active = ids >= 0
                if not active.any():
                    break
                phase = pre_action_phases(obs[:, :162], fields, goal, table_top, object_base_size,
                                          initial_z, lengths,
                                          raw._current_success_tolerance * raw.cfg.reward.keypoint_scale)
                actions = np.asarray(rollout_action(policy, env, obs, vector_step, args.training_startup))
                if startup is not None and vector_step == 0:
                    startup["first_actions"] = actions.tolist()
                if not np.isfinite(actions).all():
                    raise RuntimeError("Nonfinite policy actions")
                result, reward, terminated, truncated, step_info = step_env(env, actions, args.entry)
                done = terminated | truncated
                nxt = terminal_next_observation(result, done, step_info)
                if startup is not None and vector_step == 0:
                    startup["first_step_final_observed_progress"] = nxt[:, fields["progress"]].reshape(-1).tolist()
                    startup["first_step_done"] = done.tolist()
                state = nxt[:, :162]
                if not np.isfinite(reward).all():
                    raise RuntimeError("Nonfinite rewards")
                geometry = geometric_metrics(state, fields, goal, table_top, object_base_size)
                height = geometry["object_z"] - initial_z
                near = geometry["tip_box_distance_m"] <= .02
                off = geometry["object_bottom_clearance_m"] > .005
                relative = geometry["object_pos"] - geometry["palm_pos"]
                stable_proxy = (off & (height > .02) & (geometry["palm_object_distance_m"] < .2)
                                & (np.linalg.norm(relative - previous_relative, axis=-1) < .005))
                hold_count = np.where(stable_proxy, hold_count + 1, 0)
                hold = hold_count >= hold_steps
                task_lift = state[:, fields["lifted_object"]][:, 0] > .5
                ever_lift |= task_lift
                ever_near |= near
                ever_off |= off
                ever_hold |= hold
                dropped |= ever_lift & (height < 0)
                max_height = np.maximum(max_height, height)
                returns += reward
                discounted += reward * float(cfg.agent.gamma) ** lengths
                lengths += 1
                for name, value in step_info.get("episode_cumulative", {}).items():
                    value = np.asarray(value)
                    if value.shape == (n,):
                        reward_components.setdefault(name, np.zeros(n))[:] += value
                if not args.no_dataset:
                    values = {"obs": obs[:, :162], "next_obs": state, "actions": actions,
                              "action_targets": state[:, fields["prev_action_targets"]],
                              "rewards": reward, "terminated": terminated, "truncated": truncated,
                              "episode_id": ids, "step_in_episode": lengths - 1,
                              "phase": phase,
                              "near_object_proxy": near, "off_table_proxy": off,
                              "sustained_lift_near_palm_proxy": hold, "task_lift_flag": task_lift,
                              **geometry}
                    if args.policy == "official":
                        if policy.teacher_actions is None:
                            raise RuntimeError("Official forward hook did not capture teacher actions")
                        values["teacher_actions"] = policy.teacher_actions
                    if startup is not None:
                        values["startup_random_action"] = np.full(n, vector_step == 0)
                    for name, value in values.items():
                        chunks.setdefault(name, []).append(np.asarray(value)[active].copy())
                for j, section in enumerate((slice(0, 7), slice(7, 29))):
                    selected = actions[active, section]
                    action_abs[j] += np.abs(selected).mean(axis=-1).sum()
                    action_delta[j] += np.abs(selected - previous_actions[active, section]).mean(axis=-1).sum()
                    action_clip[j] += (np.abs(selected) >= .999).mean(axis=-1).sum()
                action_count += int(active.sum())
                final_metrics = step_info.get("episode_final", {})
                for i in np.flatnonzero(done & active):
                    metrics = pose_errors(state[i], fields, goal, offsets, raw.cfg.obs.clamp_abs_observations)
                    success_values = final_metrics.get("all_goals_hit")
                    if success_values is None:
                        raise RuntimeError("Missing pre-reset episode_final.all_goals_hit")
                    row = {"episode_id": int(ids[i]), "env_id": int(i), "complete": True,
                           "return": float(returns[i]), "discounted_raw_return": float(discounted[i]),
                           "length": int(lengths[i]), "success": bool(success_values[i]),
                           "terminated": bool(terminated[i]), "truncated": bool(truncated[i]),
                           "ever_task_lift": bool(ever_lift[i]), "ever_near_object_proxy": bool(ever_near[i]),
                           "ever_off_table_proxy": bool(ever_off[i]),
                           "ever_sustained_lift_near_palm_proxy": bool(ever_hold[i]),
                           "dropped_below_reset_after_task_lift": bool(dropped[i]),
                           "max_object_center_height_above_reset_m": float(max_height[i]),
                           "position_error_m": metrics[0], "angular_error_rad": metrics[1],
                           "keypoint_error_m": metrics[2],
                           "reward_components": {name: float(value[i]) for name, value in reward_components.items()}}
                    rows.append(row)
                    if startup is not None:
                        row["startup_episode"] = row["episode_id"] < min(n, args.episodes)
                    episode_file.write(json.dumps(row, allow_nan=False) + "\n")
                    print("DIAG_EPISODE " + json.dumps(row, allow_nan=False), flush=True)
                    ids[i] = next_id if next_id < args.episodes else -1
                    next_id += int(next_id < args.episodes)
                policy.reset_done(done)
                for value in (lengths, returns, discounted, ever_lift, ever_near, ever_off,
                              ever_hold, dropped, max_height, hold_count):
                    value[done] = 0
                for value in reward_components.values():
                    value[done] = 0
                obs = result
                previous_relative = relative
                previous_actions = actions.copy()
                if done.any():
                    reset_geometry = geometric_metrics(obs[:, :162], fields, goal, table_top, object_base_size)
                    previous_relative[done] = (reset_geometry["object_pos"] - reset_geometry["palm_pos"])[done]
                    previous_actions[done] = 0
            for i in np.flatnonzero((ids >= 0) & (lengths > 0)):
                episode_file.write(json.dumps({"episode_id": int(ids[i]), "env_id": int(i),
                    "complete": False, "length": int(lengths[i]), "return": float(returns[i]),
                    "reason": "diagnostic vector-step budget exhausted"}) + "\n")

        digest_after = model_digest(policy.model) if policy.model is not None else None
        if digest_after != policy.digest_before:
            raise RuntimeError("Frozen rollout unexpectedly modified model parameters or buffers")
        metadata["successful_episode_ids"] = [row["episode_id"] for row in rows if row["success"]]
        metadata["completed_episode_ids"] = [row["episode_id"] for row in rows]
        metadata["torch_numerics_after_rollout"] = numerical_settings(torch)
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
        transitions = action_count
        if chunks:
            dataset = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
            dataset["success"] = np.isin(dataset["episode_id"], metadata["successful_episode_ids"])
            dataset["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
            np.savez_compressed(args.output_dir / "transitions.npz", **dataset)
            transitions = len(dataset["rewards"])
        summary = {"status": "complete" if len(rows) == args.episodes else "step_budget_exhausted",
                   "episodes_completed": len(rows), "episodes_requested": args.episodes,
                   "transitions": transitions, "wallclock_seconds": time.monotonic() - started,
                   "model_unchanged": True, "model_digest_after": digest_after,
                   "dataset": None if args.no_dataset else "transitions.npz",
                   "limitations": metadata["notes"]}
        for name in ("success", "ever_task_lift", "ever_near_object_proxy", "ever_off_table_proxy",
                     "ever_sustained_lift_near_palm_proxy", "dropped_below_reset_after_task_lift",
                     "return", "discounted_raw_return", "length", "position_error_m", "angular_error_rad",
                     "keypoint_error_m", "max_object_center_height_above_reset_m"):
            values = [row[name] for row in rows]
            summary[name] = {"mean": float(np.mean(values)) if values else None, "count": len(values)}
        summary["actions"] = {name: {"mean_abs": float(action_abs[j] / max(action_count, 1)),
                                      "mean_abs_delta": float(action_delta[j] / max(action_count, 1)),
                                      "fraction_abs_ge_0.999": float(action_clip[j] / max(action_count, 1))}
                              for j, name in enumerate(("arm_7", "hand_22"))}
        if startup is not None:
            summary["startup_cohorts"] = {}
            for name, initial in (("initial_batch", True), ("subsequent_episodes", False)):
                cohort = [row for row in rows if row["startup_episode"] == initial]
                summary["startup_cohorts"][name] = {
                    "episodes_completed": len(cohort),
                    "success_count": sum(row["success"] for row in cohort),
                    "success_rate": float(np.mean([row["success"] for row in cohort])) if cohort else None,
                    "mean_return": float(np.mean([row["return"] for row in cohort])) if cohort else None,
                    "mean_length": float(np.mean([row["length"] for row in cohort])) if cohort else None,
                }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        print("DIAG_RESULT " + json.dumps(summary, allow_nan=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
