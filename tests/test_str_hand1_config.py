"""The hand-response experiment must remain a single task-variable change."""
from scripts.check_str_full_nodr import load_config, observation_sizes


def test_hand1_preserves_full_task_algorithm_and_budget():
    baseline_cfg, _, baseline = load_config('simtoolreal_full_arm1')
    candidate_cfg, _, candidate = load_config('simtoolreal_full_arm1_hand1')
    assert baseline['action']['hand_moving_average'] == .1
    assert candidate['action']['hand_moving_average'] == 1.
    candidate['action']['hand_moving_average'] = .1
    assert candidate == baseline
    assert candidate_cfg.agent == baseline_cfg.agent
    for key in ('num_train_envs', 'num_env_steps', 'updates_per_interaction_step',
                'gamma', 'n_step', 'save_checkpoint_per_interaction_step'):
        assert candidate_cfg[key] == baseline_cfg[key]
    assert candidate_cfg.num_train_envs == 2048
    assert candidate_cfg.num_env_steps == 100003840
    assert candidate_cfg.save_buffer_per_interaction_step == 48830
    assert candidate_cfg.agent_load_path is None and candidate_cfg.buffer_load_path is None
    sizes = observation_sizes()
    assert sum(sizes[name] for name in candidate['obs']['obs_list']) == 140
    assert sum(sizes[name] for name in candidate['obs']['state_list']) == 162
