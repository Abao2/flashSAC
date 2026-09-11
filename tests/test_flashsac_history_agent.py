"""Real CPU FlashSAC updates: history/replay/timeout/Q/checkpoint integration.

Spies observe real replay and update calls; no update or optimizer is bypassed.
Legacy unconditional torch.compile helpers run eagerly for this CPU-only test.
"""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from flash_rl.agents.flashSAC import agent as agent_module
from flash_rl.agents.flashSAC import update as update_module
from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig
from flash_rl.agents.flashSAC.history_network import FlashSACHistoryActor
from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.buffers.history_buffer import TorchHistoryBuffer
from flash_rl.buffers.torch_buffer import TorchUniformBuffer


def tiny_config():
    return FlashSACConfig(
        seed=0, normalize_reward=True, normalized_G_max=5., asymmetric_observation=True,
        device_type="cpu", buffer_max_length=64, buffer_min_length=4,
        buffer_device_type="cpu", sample_batch_size=4,
        learning_rate_init=.001, learning_rate_peak=.001, learning_rate_end=.001,
        learning_rate_warmup_rate=0., learning_rate_warmup_step=0,
        learning_rate_decay_rate=1., learning_rate_decay_step=100,
        actor_num_blocks=1, actor_hidden_dim=16, actor_bc_alpha=0.,
        actor_noise_zeta_mu=2., actor_noise_zeta_max=4, actor_update_period=1,
        critic_num_blocks=1, critic_hidden_dim=16, critic_num_bins=11,
        critic_min_v=-5., critic_max_v=5., critic_target_update_tau=.01,
        temp_initial_value=.01, temp_target_sigma=.15, temp_target_entropy=0.,
        gamma=.99, n_step=1, use_compile=False, compile_mode="default", use_amp=False,
        load_optimizer=True, load_reward_normalizer=True,
    )


def make_agent(history=True):
    cfg = tiny_config()
    if history:
        cfg = replace(cfg, actor_history_length=4, actor_lstm_hidden_dim=16)
    return FlashSACAgent(
        gym.spaces.Box(-np.inf, np.inf, shape=(2, 7), dtype=np.float32),
        gym.spaces.Box(-1., 1., shape=(2, 2), dtype=np.float32),
        {"actor_observation_size": (3,), "critic_observation_size": (4,),
         "critic_observation_offset": 3}, cfg,
    )


class HistoryAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.eager = torch.compiler.set_stance("force_eager")
        cls.eager.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.eager.__exit__(None, None, None)
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(31)

    def fill_without_sampling(self, agent):
        """Warmup path: every transition recorded, sample_actions never called."""
        current = torch.tensor([[-.6,.2,.7,-.1,.3,.2,.5], [.4,-.5,.1,.6,.8,-.2,.3]])
        past = [[], []]
        expected_windows, transitions = [], []
        for step in range(6):
            histories = torch.zeros(2, 4, 4)
            for env in range(2):
                frames = (past[env] + [current[env, :3].clone()])[-4:]
                histories[env, -len(frames):, :3] = torch.stack(frames)
                histories[env, -len(frames):, 3] = 1.
            following = current + torch.tensor([.03,-.02,.01,.02,-.01,.04,-.03])
            terminated, truncated = torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
            if step == 1:
                truncated[0] = True
                following[0] = torch.tensor([.7,-.4,.5,1.1,.6,.9,.8])  # Real timeout final_obs.
            if step == 3:
                terminated[1] = True
            transition = {"observation": current.clone(), "next_observation": following.clone(),
                "action": torch.tensor([[.1,-.2],[-.3,.4]]), "reward": torch.tensor([.2,.4])+step*.1,
                "terminated": terminated, "truncated": truncated}
            agent.process_transition(transition)
            self.assertEqual(float(agent.reward_normalizer.G_rms.count), 2*(step+1))
            if agent.cfg.actor_history_length > 1:
                replay = agent._replay_buffer.sample(np.array([2*step,2*step+1]))
                torch.testing.assert_close(replay["actor_history"], histories, rtol=0, atol=0)
                next_frame = torch.cat((following[:, :3], torch.ones(2,1)), dim=1)[:, None]
                expected_next = torch.cat((histories[:, 1:], next_frame), dim=1)
                torch.testing.assert_close(replay["actor_next_history"], expected_next, rtol=0, atol=0)
                online = histories[:, 1:].clone()
                online[terminated | truncated] = 0
                torch.testing.assert_close(agent._online_actor_history, online, rtol=0, atol=0)
            expected_windows.extend(histories.clone())
            transitions.append(transition)
            for env in range(2):
                past[env] = [] if terminated[env] or truncated[env] else (past[env]+[current[env,:3].clone()])[-3:]
            current = following.clone()
            if truncated[0]:
                current[0] = torch.tensor([-.8,-.7,.3,-.4,.4,.8,-.1])  # Auto-reset observation, not timeout target.
            if terminated[1]:
                current[1] = torch.tensor([.3,.9,-.5,.8,-.6,-.4,.2])
        return current, torch.stack(expected_windows), transitions

    def test_warmup_online_replay_and_timeout_boundaries(self):
        agent = make_agent()
        current, expected, transitions = self.fill_without_sampling(agent)
        self.assertIsInstance(agent._replay_buffer, TorchHistoryBuffer)
        replay = agent._replay_buffer.sample(np.arange(12))
        torch.testing.assert_close(replay["actor_history"], expected, rtol=0, atol=0)
        torch.testing.assert_close(replay["actor_next_history"][2,-1,:3], transitions[1]["next_observation"][0,:3])
        torch.testing.assert_close(replay["actor_history"][4,-1,:3], transitions[2]["observation"][0,:3])
        self.assertFalse(torch.equal(replay["actor_next_history"][2,-1,:3], replay["actor_history"][4,-1,:3]))
        torch.testing.assert_close(replay["actor_history"][4,:3], torch.zeros(3,4), rtol=0, atol=0)
        self.assertEqual(replay["terminated"][2].item(), 0)
        self.assertEqual(replay["truncated"][2].item(), 1)
        torch.testing.assert_close(replay["discount"], torch.full((12,), .99), rtol=0, atol=0)
        online_before = agent._online_actor_history.clone()
        expected_context = agent._actor_history_context(current[:,:3])
        with patch.object(agent._actor.network, "get_mean_and_std", wraps=agent._actor.network.get_mean_and_std) as spy:
            for _ in range(2):
                actions = agent.sample_actions(6, {"next_observation": current}, training=True)
                self.assertEqual(actions.shape, (2,2))
                self.assertTrue(np.isfinite(actions).all())
                torch.testing.assert_close(spy.call_args.kwargs["observations"], expected_context, rtol=0, atol=0)
                torch.testing.assert_close(agent._online_actor_history, online_before, rtol=0, atol=0)
        self.assertEqual(float(agent.reward_normalizer.G_rms.count), 12.)
        with self.assertRaisesRegex(RuntimeError, "generic eval is unsupported"):
            agent.sample_actions(6, {"next_observation": current}, training=False)

    def test_real_updates_Q_history_gamma_and_optimizer(self):
        agent = make_agent()
        self.fill_without_sampling(agent)
        self.assertTrue(agent.can_start_training())
        self.assertEqual(agent._critic_network_observation_dim, 4+4*(3+1))
        self.assertEqual(agent._critic.network.embedder.w.weight.shape[-1], 20+2)
        self.assertTrue(all(p.device.type == "cpu" for p in agent._actor.network.parameters()))
        indices = np.array([2,3,4,7])  # timeout, continuing, post-reset, true failure.
        sample = agent._replay_buffer.sample
        raw = sample(indices)
        before = agent._actor.network.lstm.weight_ih_l0.detach().clone()
        captured, td_targets, q_inputs = [], [], []
        actual_update = agent_module._update_networks
        actual_target = update_module._compute_categorical_td_target
        def record_update(*args, **kwargs):
            captured.append({k:v.detach().clone() for k,v in kwargs["batch"].items()})
            return actual_update(*args, **kwargs)
        def record_target(*args, **kwargs):
            td_targets.append({k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()})
            return actual_target(*args, **kwargs)
        hook = agent._target_critic.network.register_forward_pre_hook(
            lambda module, args, kwargs: q_inputs.append(kwargs["observations"].detach().clone()), with_kwargs=True)
        try:
            with patch.object(agent._replay_buffer, "sample", side_effect=lambda: sample(indices)), \
                 patch.object(agent_module, "_update_networks", side_effect=record_update), \
                 patch.object(update_module, "_compute_categorical_td_target", side_effect=record_target):
                for _ in range(4):
                    info = agent.update()
                    self.assertIn("actor/loss", info)
                    self.assertIn("critic/loss", info)
                    self.assertTrue(all(np.isfinite(value) for value in info.values()))
        finally:
            hook.remove()
        self.assertEqual(agent._update_step, 4)
        self.assertGreater(float((before-agent._actor.network.lstm.weight_ih_l0).abs().sum()), 0.)
        self.assertEqual(float(agent.reward_normalizer.G_rms.count), 12.)
        self.assertEqual(len(td_targets), 4)
        for batch, target, q_input in zip(captured, td_targets, q_inputs):
            for label, raw_label in (("actor_observation","actor_history"), ("actor_next_observation","actor_next_history")):
                torch.testing.assert_close(batch[label], raw[raw_label], rtol=0, atol=0)
            for label, history_label in (("observation","actor_history"), ("next_observation","actor_next_history")):
                expected = torch.cat((raw[label][:,3:7], raw[history_label].flatten(1)), dim=1)
                torch.testing.assert_close(batch[label], expected, rtol=0, atol=0)
            torch.testing.assert_close(q_input, torch.cat((batch["observation"],batch["next_observation"])), rtol=0, atol=0)
            torch.testing.assert_close(target["gamma"], torch.full((4,),.99), rtol=0, atol=0)
            torch.testing.assert_close(target["done"], raw["terminated"], rtol=0, atol=0)
            self.assertEqual(target["done"].tolist(), [0.,0.,0.,1.])
        adam_steps = [float(state["step"]) for state in agent._actor.optimizer.state.values()]
        self.assertTrue(adam_steps and all(step == 4 for step in adam_steps))

    def test_history_checkpoint_metadata_and_roundtrip(self):
        agent = make_agent()
        self.fill_without_sampling(agent)
        agent.update()
        inputs = agent._replay_buffer.sample(np.array([2,3,4,7]))["actor_history"]
        expected = agent._actor.network.get_mean_and_std(inputs, False)
        with tempfile.TemporaryDirectory() as folder:
            agent.save(folder)
            state = torch.load(Path(folder)/"agent_state.pt", weights_only=True)
            self.assertEqual(state["history_policy"], {"history_length":4,"lstm_hidden_dim":16,
                "actor_observation_dim":3,"critic_current_observation_dim":4,"critic_network_observation_dim":20})
            restored = make_agent()
            restored._online_actor_history = torch.ones(2,3,4)
            restored.load(folder)
            self.assertIsNone(restored._online_actor_history)
            self.assertEqual(restored._update_step, agent._update_step)
            self.assertEqual(len(restored._replay_buffer), 0)
            self.assertEqual(float(restored.reward_normalizer.G_rms.count), 12.)
            actual = restored._actor.network.get_mean_and_std(inputs, False)
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            for key,value in agent._actor.network.state_dict().items():
                torch.testing.assert_close(value,restored._actor.network.state_dict()[key],rtol=0,atol=0)

    def test_default_feedforward_old_schema_unchanged(self):
        agent = make_agent(history=False)
        self.assertEqual(agent.cfg.actor_history_length, 1)
        self.assertIs(type(agent._actor.network), FlashSACActor)
        self.assertIs(type(agent._replay_buffer), TorchUniformBuffer)
        self.assertEqual(agent._critic_network_observation_dim, 4)
        current, _, _ = self.fill_without_sampling(agent)
        self.assertIsNone(agent._online_actor_history)
        self.assertNotIn("actor_history", agent._replay_buffer.sample())
        agent.update()
        expected = agent.sample_actions(6, {"next_observation":current}, training=False)
        with tempfile.TemporaryDirectory() as folder:
            agent.save(folder)
            state = torch.load(Path(folder)/"agent_state.pt", weights_only=True)
            self.assertEqual(set(state), {"update_step","grad_scaler_state_dict"})
            actor_checkpoint = torch.load(Path(folder)/"actor.pt", weights_only=True)
            self.assertEqual(set(actor_checkpoint), {"network_state_dict","optimizer_state_dict","scheduler_state_dict","update_step"})
            self.assertFalse(any(key.startswith(("lstm.","head.","frame_norm.")) for key in actor_checkpoint["network_state_dict"]))
            restored = make_agent(history=False)
            restored.load(folder)  # Exact old schema, no history fields required.
            actual = restored.sample_actions(6, {"next_observation":current}, training=False)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(restored._update_step, 1)


if __name__ == "__main__":
    unittest.main()
