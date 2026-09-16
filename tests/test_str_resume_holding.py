"""CPU-only native-resume hooks; no simulator or training is launched."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flash_rl.agents.utils.reward_normalization import RewardNormalizer
from scripts.check_str_full_nodr import load_config
from scripts.train_str_holding import CHECKPOINT_NAMES, install_resume_hooks, resume_overrides


def resume_args(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in CHECKPOINT_NAMES:
        (checkpoint / f"{name}.pt").write_bytes(b"validation fixture")
    return SimpleNamespace(resume_checkpoint=checkpoint, resume_tolerance=.075 * .9 ** 8,
                           env_step_offset=100003840, output_root=tmp_path)


def test_resume_validation_and_native_load_overrides(tmp_path):
    args = resume_args(tmp_path)
    overrides = resume_overrides(args)
    assert overrides == [f'agent_load_path="{args.resume_checkpoint}"',
                         f'buffer_load_path="{args.resume_checkpoint}"', "require_agent_load=true",
                         "agent.load_optimizer=true", "agent.load_reward_normalizer=true"]
    (args.resume_checkpoint / "replay_buffer.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="replay_buffer"):
        resume_overrides(args)


@pytest.mark.parametrize("changes", [
    {"resume_checkpoint": Path("relative")}, {"env_step_offset": 0}, {"env_step_offset": -1},
    {"resume_tolerance": None}, {"resume_tolerance": float("nan")}, {"resume_tolerance": -1},
    {"resume_checkpoint": None},
])
def test_resume_rejects_incoherent_options(tmp_path, changes):
    args = resume_args(tmp_path)
    vars(args).update(changes)
    with pytest.raises(ValueError):
        resume_overrides(args)


def test_fresh_path_leaves_factories_untouched():
    args = SimpleNamespace(resume_checkpoint=None, resume_tolerance=None, env_step_offset=0)
    train = SimpleNamespace(create_envs=object(), create_agent=object(), create_logger=object())
    before = vars(train).copy()
    assert resume_overrides(args) == []
    install_resume_hooks(train, args)()
    assert vars(train) == before


def test_legacy_continuation_preserves_cumulative_task_agent_and_additional_budget():
    baseline, _, baseline_task = load_config("simtoolreal_full_arm1_nstep3")
    candidate, _, candidate_task = load_config("simtoolreal_full_nstep3_continue")
    assert baseline_task["termination"]["force_consecutive_near_goal_steps"] is True
    assert candidate_task["termination"]["force_consecutive_near_goal_steps"] is False
    # The sole task difference is intentional preservation of the old replay's
    # cumulative success semantics; all other task and learner settings match.
    candidate_task["termination"]["force_consecutive_near_goal_steps"] = True
    assert candidate_task == baseline_task and candidate.agent == baseline.agent
    assert candidate.num_train_envs == 2048 and candidate.updates_per_interaction_step == 4
    assert candidate.num_env_steps == 400015360 and candidate.num_interaction_steps == 195320
    assert candidate.resume_env_step_offset == 100003840
    assert candidate.resume_env_step_offset + candidate.num_env_steps == 500019200
    assert candidate.resume_tolerance == .03228504075
    assert candidate.save_checkpoint_per_interaction_step == 9766
    assert candidate.save_buffer_per_interaction_step == 48830


def test_hooks_restore_tolerance_offset_and_replay_without_resetting_learned_statistics(tmp_path):
    args = resume_args(tmp_path)
    term = SimpleNamespace(success_tolerance=.075, target_success_tolerance=.01, eval_success_tolerance=None)
    raw = SimpleNamespace(cfg=SimpleNamespace(termination=term), _current_success_tolerance=.075,
                          _frame_counter=99, _last_curriculum_update=55,
                          _prev_episode_successes=torch.ones(2))
    environments = (SimpleNamespace(envs=SimpleNamespace(unwrapped=raw)), None, None)
    normalizer = RewardNormalizer(.99, 5., True, torch.device("cpu"))
    normalizer.G_r = torch.tensor([3., 4.])
    normalizer.G_r_max.fill_(9)
    normalizer.G_rms.mean.fill_(2)
    normalizer.G_rms.var.fill_(7)
    normalizer.G_rms.count.fill_(100003840)
    preserved = {name: value.clone() for name, value in dict(
        maximum=normalizer.G_r_max, mean=normalizer.G_rms.mean,
        var=normalizer.G_rms.var, count=normalizer.G_rms.count).items()}
    calls, logged = [], []

    class Buffer:
        _current_idx = 3
        def __len__(self):
            return 7

    agent = SimpleNamespace(reward_normalizer=normalizer, _update_step=195120, _replay_buffer=Buffer(),
                            load_replay_buffer=lambda path: calls.append(path), scheduler=object())
    scheduler = agent.scheduler
    logger = SimpleNamespace(log_metric=lambda step: logged.append(step))
    train = SimpleNamespace(create_envs=lambda: environments, create_agent=lambda: agent, create_logger=lambda: logger)
    originals = vars(train).copy()
    restore = install_resume_hooks(train, args)
    try:
        assert train.create_envs() is environments
        assert raw._current_success_tolerance == args.resume_tolerance
        assert term.success_tolerance == .075 and term.eval_success_tolerance is None
        assert raw._frame_counter == raw._last_curriculum_update == 0
        assert not raw._prev_episode_successes.any()
        wrapped_logger = train.create_logger()
        wrapped_logger.log_metric(0)
        wrapped_logger.log_metric(102400)
        assert logged == [100003840, 100106240]
        train.create_agent().load_replay_buffer(str(args.resume_checkpoint))
        assert calls == [str(args.resume_checkpoint)] and not normalizer.G_r.any()
        for name, value in dict(maximum=normalizer.G_r_max, mean=normalizer.G_rms.mean,
                                var=normalizer.G_rms.var, count=normalizer.G_rms.count).items():
            torch.testing.assert_close(value, preserved[name], rtol=0, atol=0)
        assert agent._update_step == 195120 and agent.scheduler is scheduler
        proof = json.loads((tmp_path / "resume_loaded.json").read_text())
        assert (proof["replay_length"], proof["replay_current_idx"], proof["update_step"]) == (7, 3, 195120)
        assert proof["cleared_G_r_elements"] == 2 and not proof["exact_environment_resume"]
        args.resume_tolerance = .08
        with pytest.raises(ValueError, match="bounds"):
            train.create_envs()
        args.resume_tolerance = .005
        with pytest.raises(ValueError, match="bounds"):
            train.create_envs()
        args.resume_tolerance = .03
        term.eval_success_tolerance = .075
        with pytest.raises(ValueError, match="pin"):
            train.create_envs()
    finally:
        restore()
    assert vars(train) == originals
