import numpy as np
import pytest
from pathlib import Path

from scripts.diagnose_str_rollouts import terminal_next_observation
from scripts.eval_str_full_nodr import Episodes, freeze_tolerance


def test_frozen_eval_real_terminal_metrics_and_budget():
    episodes = Episodes(3, 2, 50)
    assert episodes.summary()["all_goals_hit"]["mean"] is None
    assert episodes.summary()["status"] == "incomplete"
    false = np.array([False, False])
    first_done = np.array([True, False])
    info = {"final_obs": np.array([[1.0, 9.0], [0.0, 0.0]]), "_final_obs": first_done}
    state = terminal_next_observation(np.zeros((2, 2)), first_done, info)
    final = {"successes": np.array([50, 0]), "all_goals_hit": first_done,
             "done_timeout": false}
    rows = episodes.step(np.array([2.0, 3.0]), first_done, false, final, state, 0)
    assert rows[0]["all_goals_hit"] and rows[0]["lifted"]
    assert episodes.final_observations[0].tolist() == [1.0, 9.0]
    assert episodes.returns.tolist() == [0.0, 3.0]
    assert episodes.ids.tolist() == [2, 1]
    both = np.array([True, True])
    final = {"successes": np.array([1, 0]), "all_goals_hit": false, "done_timeout": both}
    rows = episodes.step(np.array([5.0, 7.0]), false, both, final, np.zeros((2, 2)), 0)
    assert [row["return"] for row in rows] == [5.0, 10.0]
    assert [row["length"] for row in rows] == [1, 2]
    assert not rows[0]["all_goals_hit"]  # One goal is not the 50-goal chain.
    summary = episodes.summary()
    assert summary["status"] == "complete" and summary["completed_episodes"] == 3
    assert summary["counts"] == {"all_goals_hit": 1, "lifted": 1, "timeout": 2, "terminated": 1}
    assert summary["observed_auto_resets_per_env"] == [2, 1]
    assert episodes.ids.tolist() == [-1, -1]
    assert episodes.step(np.ones(2), false, both, final, np.zeros((2, 2)), 0) == []


def test_missing_or_single_goal_terminal_success_rejected():
    with pytest.raises(RuntimeError, match="pre-reset final_obs"):
        terminal_next_observation(np.zeros((1, 1)), np.array([True]), {})
    episodes = Episodes(1, 1, 50)
    with pytest.raises(RuntimeError, match="whole-chain"):
        episodes.step(np.ones(1), np.array([True]), np.array([False]),
                      {"successes": [1], "all_goals_hit": [1], "done_timeout": [0]},
                      np.zeros((1, 1)), 0)


@pytest.fixture
def isaaclab_config_updater():
    """Run the installed scalar type checker, without Isaac's package bootstrap."""
    import ast
    import importlib.util
    from collections.abc import Iterable, Mapping, Sized
    from typing import Any

    spec = importlib.util.find_spec("isaaclab")
    if spec is None:
        pytest.skip("Isaac Lab is required for the real config type-check regression")
    package = Path(spec.origin).parent
    path = package / "utils/dict.py"
    if not path.is_file():
        path = package / "source/isaaclab/isaaclab/utils/dict.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "update_class_from_dict")
    namespace = {"Iterable": Iterable, "Mapping": Mapping, "Sized": Sized, "Any": Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["update_class_from_dict"]


@pytest.mark.parametrize("requested,expected", [(None, .075), (.075, .075), (.01, .01)])
@pytest.mark.parametrize("previous_eval_override", [None, .03])
def test_actual_sparse_full_config_freezes_initial_and_floor(
        isaaclab_config_updater, requested, expected, previous_eval_override):
    import hydra
    import torch
    from omegaconf import OmegaConf
    from runpy import run_path
    from types import SimpleNamespace
    from scripts.check_str_full_nodr import STR_ROOT, source_defaults

    repo = Path(__file__).resolve().parents[1]
    task = STR_ROOT / "isaacsimenvs/cfg/task/SimToolReal.yaml"
    runtime = SimpleNamespace(termination=SimpleNamespace(**source_defaults()["termination"]))
    isaaclab_config_updater(runtime, {"termination": OmegaConf.to_container(
        OmegaConf.load(task).termination, resolve=True)})
    # This is the actual failure hidden by the earlier OmegaConf-only check.
    with pytest.raises(ValueError, match="/termination/eval_success_tolerance.*NoneType.*float"):
        isaaclab_config_updater(runtime, {"termination": {"eval_success_tolerance": expected}})

    OmegaConf.register_new_resolver("eval", lambda expression: eval(expression), replace=True)
    try:
        with hydra.initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
            cfg = hydra.compose(config_name="simtoolreal_full_nodr")
        if previous_eval_override is not None:
            OmegaConf.update(cfg, "env.task_cfg_overrides.termination.eval_success_tolerance",
                             previous_eval_override, force_add=True)
        initial, actual = freeze_tolerance(cfg, task, requested)
        assert initial == .075 and actual == expected
        OmegaConf.resolve(cfg)
        isaaclab_config_updater(runtime, {"termination": OmegaConf.to_container(
            cfg.env.task_cfg_overrides.termination, resolve=True)})
        term = runtime.termination
        assert term.success_tolerance == term.target_success_tolerance == expected
        assert term.eval_success_tolerance is None
        assert term.tolerance_curriculum_interval > 600 * 50 * 600
        assert term.max_consecutive_successes == 50
        update_curriculum = run_path(str(
            STR_ROOT / "isaacsimenvs/tasks/simtoolreal/utils/termination_utils.py"))["update_tolerance_curriculum"]
        state = SimpleNamespace(cfg=runtime, _frame_counter=term.tolerance_curriculum_interval,
                                _last_curriculum_update=0, _prev_episode_successes=torch.tensor([50.]),
                                _current_success_tolerance=term.success_tolerance)
        update_curriculum(state)  # Even an eligible curriculum update cannot move equal bounds.
        assert state._current_success_tolerance == expected
    finally:
        OmegaConf.clear_resolver("eval")


def test_partial_object_coverage_is_not_reported_as_full_distribution():
    episodes = Episodes(1, 1, 50, [{"asset_idx": 0, "asset_type": "hammer"}],
                        ["hammer", "eraser"])
    episodes.step(np.ones(1), np.array([False]), np.array([True]),
                  {"successes": [0], "all_goals_hit": [0], "done_timeout": [1]},
                  np.zeros((1, 1)), 0)
    summary = episodes.summary()
    assert summary["complete_budget"]
    assert not summary["coverage"]["full_asset_pool_covered"]
    assert not summary["coverage"]["equal_completed_type_counts"]
    assert summary["coverage"]["completed_episode_type_counts"] == {"eraser": 0, "hammer": 1}
