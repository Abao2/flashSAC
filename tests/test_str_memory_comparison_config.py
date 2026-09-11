from pathlib import Path

import hydra
from omegaconf import OmegaConf


def test_three_groups_change_only_information_and_opt_in_memory():
    repo = Path(__file__).resolve().parents[1]
    OmegaConf.register_new_resolver('eval', lambda expression: eval(expression), replace=True)
    try:
        with hydra.initialize_config_dir(version_base=None, config_dir=str(repo / 'configs')):
            groups = [hydra.compose(config_name=name) for name in (
                'simtoolreal_full_nodr', 'simtoolreal_full_state162', 'simtoolreal_full_lstm32')]
        data = [OmegaConf.to_container(cfg, resolve=True) for cfg in groups]
        task = OmegaConf.load('/home/abao/simtoolreal/isaacsimenvs/cfg/task/SimToolReal.yaml')
        resolved = [OmegaConf.merge(task, cfg.env.task_cfg_overrides) for cfg in groups]
        assert resolved[1].obs.obs_list == resolved[1].obs.state_list
        assert resolved[0] == resolved[2]
        resolved[1].obs.obs_list = resolved[0].obs.obs_list
        assert resolved[0] == resolved[1]
        assert data[2]['agent'].pop('actor_history_length') == 32
        assert data[2]['agent'].pop('actor_lstm_hidden_dim') == 128
        data[1]['env']['task_cfg_overrides'].pop('obs')
        for cfg in data:
            cfg.pop('group_name')
            cfg.pop('exp_name')
            cfg.pop('save_path')  # Derived experiment-specific artifact location.
        assert data[0] == data[1] == data[2]
        assert data[0]['num_env_steps'] == 100003840
        assert data[0]['num_train_envs'] == 2048
        assert data[0]['updates_per_interaction_step'] == 4
        assert data[0]['agent']['buffer_max_length'] == 10000000
        assert data[0]['num_eval_episodes'] == 0
        assert resolved[0].action.arm_moving_average == resolved[0].action.hand_moving_average == .1
    finally:
        OmegaConf.clear_resolver('eval')
