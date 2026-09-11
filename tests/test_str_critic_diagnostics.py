"""CPU checks for the frozen STR critic probe; no simulator required."""
import copy

import numpy as np
import pytest
import torch

from scripts.diagnose_str_critic import diagnose, reward_denominator, select_transition_ids, target_metrics
from flash_rl.agents.flashSAC.network import FlashSACActor, FlashSACDoubleCritic
from flash_rl.agents.flashSAC.update import _compute_categorical_td_target


def test_terminal_mask_entropy_and_unprojected_mass():
    bins = torch.tensor([-1., 0., 1.])
    prob = torch.tensor([[.1, .2, .7], [.1, .2, .7], [.1, .2, .7]])
    reward = torch.tensor([.5, .5, -2.])
    # Second row is a TIMEOUT, so terminated=False and it must bootstrap.
    terminal = torch.tensor([True, False, False])
    alpha_logp = torch.tensor([-.4, -.4, -.4])
    out = target_metrics(prob, bins, reward, terminal, alpha_logp, 1.)
    torch.testing.assert_close(out["target_unprojected_mean"], torch.tensor([.5, 1.5, -1.]))
    torch.testing.assert_close(out["target_mass_above_support"], torch.tensor([0., .7, 0.]))
    torch.testing.assert_close(out["target_mass_below_support"], torch.tensor([0., 0., .3]))
    native = getattr(_compute_categorical_td_target, "_torchdynamo_orig_callable", _compute_categorical_td_target)
    projected = native(prob.log(), reward, terminal.float(), alpha_logp, 1., 3, -1., 1.)
    torch.testing.assert_close(out["target_clipped_mean"], (projected * bins).sum(-1))
    assert reward_denominator({"G_rms_var": torch.tensor([4.]), "G_r_max": torch.tensor([30.])}, 5.) == 6.


def test_cpu_probe_preserves_models_and_separates_phase_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    actor = FlashSACActor(1, 162, 8, 29)
    critic = FlashSACDoubleCritic(1, 191, 8, 11, -5., 5.).requires_grad_(False)
    target = copy.deepcopy(critic)
    snapshots = [copy.deepcopy(model.state_dict()) for model in (actor, critic, target)]
    rng = np.random.default_rng(0)
    data = dict(obs=rng.normal(size=(8, 162)).astype("float32"),
                next_obs=rng.normal(size=(8, 162)).astype("float32"),
                actions=rng.uniform(-1, 1, size=(8, 29)).astype("float32"),
                rewards=np.ones(8), terminated=np.array([0, 0, 1, 0, 0, 0, 0, 0]),
                truncated=np.array([0, 0, 0, 1, 0, 0, 0, 0]),
                episode_id=np.arange(8), phase=np.array(["reset"] * 4 + ["contact"] * 4))
    report = diagnose(actor, critic, target, .01, 3., data, random_actions=3,
                      batch_size=4, gradient_repeats=2)
    assert set(report) == {"reset", "contact"}
    assert report["reset"]["truncated"] == 1
    assert report["contact"]["sampled_transitions"] == 4
    assert report["contact"]["actor_gradient"]["total_loss_gradient_l2"]["mean"] > 0
    assert report["contact"]["metrics"]["uniform_action_q_range"]["mean"] > 0
    for model, before in zip((actor, critic, target), snapshots):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert all(p.grad is None for p in model.parameters())


def test_selector_keeps_every_high_reward_event_without_duplicate_rows():
    phases = np.array(["approach"] * 50 + ["lift"] * 50)
    rewards = np.zeros(100)
    rewards[[2, 8, 20, 61, 89]] = 300.
    rewards[10] = 200.  # Strictly greater than threshold, not greater-or-equal.
    ids, counts, info = select_transition_ids(phases, rewards, np.random.default_rng(7), 2)
    assert counts == {"approach": 50, "lift": 50}
    assert set(np.flatnonzero(rewards > 200)).issubset(ids)
    assert len(np.unique(ids)) == len(ids)
    assert info["reward_events_available"] == info["reward_events_selected"] == 5
    assert info["phase_balanced_base_transitions"] == 4
    assert info["reward_events_added_beyond_phase_sample"] == len(ids) - 4
    ids_again, _, _ = select_transition_ids(phases, rewards, np.random.default_rng(7), 2)
    np.testing.assert_array_equal(ids, ids_again)
    with pytest.raises(ValueError, match="No reward events were silently discarded"):
        select_transition_ids(phases, rewards, np.random.default_rng(7), 2, max_reward_events=4)
