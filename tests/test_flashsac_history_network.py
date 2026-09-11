"""CPU checks for finite-window semantics, gradient flow, and FlashSAC interfaces."""

import copy
import io
import unittest

import torch

from flash_rl.agents.flashSAC.history_network import FlashSACHistoryActor
from flash_rl.agents.flashSAC.network import FlashSACActor


class HistoryActorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.actor = FlashSACHistoryActor(1, 5, 16, 3, history_length=4, lstm_hidden_dim=7)
        self.obs = torch.randn(4, 4, 6)
        self.lengths = [1, 2, 4, 3]
        for row, length in enumerate(self.lengths):
            self.obs[row, :, -1] = 0
            self.obs[row, -length:, -1] = 1

    def test_padding_never_affects_memory_head_or_batchnorm(self):
        altered = self.obs.clone()
        altered[..., :5][altered[..., -1] == 0] = float("nan")
        reference = copy.deepcopy(self.actor)
        for training in (False, True):
            actual = self.actor.get_mean_and_std(altered, training)
            expected = reference.get_mean_and_std(self.obs, training)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key, value in self.actor.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)

    def test_unpadded_reference_for_every_valid_length_and_batch_independence(self):
        memory = self.actor.encode_history(self.obs)
        means, stds = self.actor.get_mean_and_std(self.obs, False)
        for row, length in enumerate(self.lengths):
            raw = self.obs[row:row+1, -length:, :5]
            sequence, _ = self.actor.lstm(self.actor.frame_norm(raw))
            torch.testing.assert_close(memory[row], sequence[0, -1], atol=2e-7, rtol=2e-6)
            inputs = torch.cat((raw[:, -1], sequence[:, -1]), dim=-1)
            expected = self.actor.head.get_mean_and_std(inputs, False)
            singleton = self.actor.get_mean_and_std(self.obs[row:row+1], False)
            for batched, single, ref in zip((means[row:row+1], stds[row:row+1]), singleton, expected):
                torch.testing.assert_close(batched, single, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(single, ref, atol=2e-6, rtol=2e-5)
        # No hidden cache: an unrelated call cannot change a later identical call.
        self.actor.get_mean_and_std(self.obs.flip(0), False)
        again = self.actor.get_mean_and_std(self.obs, False)
        torch.testing.assert_close(again[0], means, rtol=0, atol=0)

    def test_history_changes_action_with_current_frame_fixed(self):
        obs = self.obs[2:3].repeat(2, 1, 1)
        obs[1, :-1, :5] = torch.randn_like(obs[1, :-1, :5]) * 3
        torch.testing.assert_close(obs[0, -1], obs[1, -1], rtol=0, atol=0)
        mean, _ = self.actor.get_mean_and_std(obs, False)
        self.assertGreater(float((mean[0].tanh() - mean[1].tanh()).abs().max()), 1e-5)

    def test_full_window_gradients_and_optimizer_update(self):
        obs = self.obs.clone().requires_grad_()
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=.001)
        before = self.actor.lstm.weight_ih_l0.detach().clone()
        mean, std = self.actor.get_mean_and_std(obs, True)
        (mean.square().mean() + std.log().square().mean()).backward()
        self.assertTrue(torch.isfinite(obs.grad).all())
        for row, length in enumerate(self.lengths):
            self.assertEqual(float(obs.grad[row, :4-length, :5].abs().sum()), 0.)
            if length > 1:
                self.assertTrue((obs.grad[row, -length:-1, :5].abs().sum(dim=-1) > 0).all())
        self.assertGreater(float(self.actor.lstm.weight_ih_l0.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertGreater(float((self.actor.lstm.weight_ih_l0 - before).abs().sum()), 0.)

    def test_forward_interface_and_checkpoint_roundtrip(self):
        checkpoint = io.BytesIO()
        torch.save(self.actor.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = FlashSACHistoryActor(1, 5, 16, 3, 4, 7)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.manual_seed(123)
        actions, info = self.actor(self.obs, False)
        torch.manual_seed(123)
        loaded_actions, loaded_info = restored(self.obs, False)
        self.assertEqual(actions.shape, (4, 3))
        self.assertEqual(info["log_prob"].shape, (4,))
        self.assertTrue(torch.isfinite(actions).all() and torch.isfinite(info["log_prob"]).all())
        self.assertLessEqual(float(actions.abs().max()), 1.)
        torch.testing.assert_close(actions, loaded_actions, rtol=0, atol=0)
        torch.testing.assert_close(info["log_prob"], loaded_info["log_prob"], rtol=0, atol=0)

    def test_default_parameter_count_and_single_frame_window(self):
        actor = FlashSACHistoryActor(2, 140, 128, 29)
        head_count = sum(p.numel() for p in FlashSACActor(2, 268, 128, 29).parameters())
        expected = head_count + 2*140 + 4*128*(140+128+2)
        self.assertEqual(sum(p.numel() for p in actor.parameters()), expected)
        self.assertEqual(expected, 445674)
        single = FlashSACHistoryActor(1, 5, 16, 3, history_length=1, lstm_hidden_dim=7)
        obs = self.obs[:, -1:]
        reference, _ = single.lstm(single.frame_norm(obs[..., :5]))
        torch.testing.assert_close(single.encode_history(obs), reference[:, -1], rtol=0, atol=0)

    def test_invalid_shape_and_masks_are_rejected(self):
        with self.assertRaises(ValueError):
            self.actor.encode_history(self.obs[:, :-1])
        for mask in ([0, 0, 0, 0], [0, 1, 0, 1], [0, .5, 1, 1]):
            bad = self.obs.clone()
            bad[0, :, -1] = torch.tensor(mask)
            with self.assertRaises(RuntimeError):
                self.actor.encode_history(bad)


if __name__ == "__main__":
    unittest.main()
