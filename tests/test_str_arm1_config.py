from pathlib import Path

import hydra
from omegaconf import OmegaConf


def test_full_arm1_changes_only_arm_control_and_artifact_labels():
    repo = Path(__file__).resolve().parents[1]
    OmegaConf.register_new_resolver('eval', lambda expression: eval(expression), replace=True)
    try:
        with hydra.initialize_config_dir(version_base=None, config_dir=str(repo / 'configs')):
            baseline = hydra.compose(config_name='simtoolreal_full_nodr')
            candidate = hydra.compose(config_name='simtoolreal_full_arm1')
        task = OmegaConf.load('/home/abao/simtoolreal/isaacsimenvs/cfg/task/SimToolReal.yaml')
        task_a = OmegaConf.merge(task, baseline.env.task_cfg_overrides)
        task_b = OmegaConf.merge(task, candidate.env.task_cfg_overrides)
        assert task_a.action.arm_moving_average == .1
        assert task_b.action.arm_moving_average == 1.0
        assert task_a.action.hand_moving_average == task_b.action.hand_moving_average == .1
        task_b.action.arm_moving_average = task_a.action.arm_moving_average
        assert task_a == task_b
        a = OmegaConf.to_container(baseline, resolve=True)
        b = OmegaConf.to_container(candidate, resolve=True)
        assert b['env']['task_cfg_overrides'].pop('action') == {'arm_moving_average': 1.0}
        for cfg in (a, b):
            for key in ('group_name', 'exp_name', 'save_path'):
                cfg.pop(key)
        assert a == b
        assert b['seed'] == 0
        assert b['num_env_steps'] == 100003840
        assert b['num_train_envs'] == 2048
        assert b['agent_load_path'] is None and b['buffer_load_path'] is None
        assert b['num_eval_episodes'] == 0
    finally:
        OmegaConf.clear_resolver('eval')
