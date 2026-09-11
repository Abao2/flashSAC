"""CPU checks of real STR helpers; no simulator, training, or policy evaluation.

The flat eraser example is illustrative, not a measured full-run trajectory.
"""

import json
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import torch


torch.set_num_threads(2)
root = Path('/home/abao/simtoolreal/isaacsimenvs/tasks/simtoolreal/utils')
apply_action = run_path(str(root / 'action_utils.py'))['apply_action_pipeline']
rewards = run_path(str(root / 'reward_utils.py'))


def arm_increment(smoothing):
    # Mid-range joints, identity ordering, no clipping or delay. The helper is
    # production code; this fixture isolates its action-to-target transform.
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            action=SimpleNamespace(dof_speed_scale=1.5,
                                   arm_moving_average=smoothing, hand_moving_average=.1),
            domain_randomization=SimpleNamespace(use_action_delay=False)),
        step_dt=1 / 60, device='cpu',
        _perm_canon_to_lab=torch.arange(29),
        _arm_joint_ids=torch.arange(7), _hand_joint_ids=torch.arange(7, 29),
        _prev_targets=torch.zeros(1, 29), _cur_targets=torch.zeros(1, 29),
        _arm_lower=torch.full((7,), -3.), _arm_upper=torch.full((7,), 3.),
        _hand_lower=torch.full((22,), -1.), _hand_upper=torch.full((22,), 1.))
    action = torch.zeros(1, 29)
    action[0, 0] = 1
    apply_action(env, action)
    return float(env._cur_targets[0, 0])


slow, fast = arm_increment(.1), arm_increment(1.)
assert abs(fast / slow - 10) < 1e-5

# Same flat eraser and table as the successful fixed task, after ideal settling.
table_top, eraser_half_height = .53, .049932924209852 / 2
settled_z = table_top + eraser_half_height
simple_init_z, full_example_init_z = settled_z + .001, .63
heights = torch.tensor([settled_z, settled_z + .01, settled_z + .03])
false = torch.zeros(3, dtype=torch.bool)
lift = rewards['lifting_reward']
simple_dense = lift(heights, torch.full((3,), simple_init_z), false, .15, 300., 20.)[0]
full_dense = lift(heights, torch.full((3,), full_example_init_z), false, .15, 300., 20.)[0]
assert simple_dense[1] > simple_dense[0]
assert full_dense[0] == full_dense[1] == 0 and full_dense[2] > 0

# Identical visible geometry can have different progress reward due to the
# historical distance tracker omitted from the 140D actor (present in critic).
kp_reward, _ = rewards['keypoint_reward'](
    torch.tensor([.2, .2]), torch.tensor([.25, .2]),
    torch.tensor([True, True]), 200.)
assert torch.allclose(kp_reward, torch.tensor([10., 0.]))

print(json.dumps({
    'status': 'PASS', 'simulator_started': False,
    'arm_target_increment_rad': {'full_smoothing_0.1': slow, 'simple_smoothing_1': fast},
    'arm_target_max_rate_rad_per_s': {'full': slow * 60, 'simple': fast * 60},
    'gamma_0.99_discount_at_policy_steps': {n: .99 ** n for n in [30, 60, 300, 600]},
    'illustrative_flat_eraser': {
        'settled_root_z': settled_z,
        'full_example_spawn_z': full_example_init_z,
        'tested_lifts_from_settled_m': [0, .01, .03],
        'simple_dense_reward': simple_dense.tolist(), 'full_dense_reward': full_dense.tolist(),
        'simple_height_above_table_rest_for_bonus_m': simple_init_z + .1 - settled_z,
        'full_example_height_above_table_rest_for_bonus_m': full_example_init_z + .1 - settled_z,
        'not_a_measured_rollout': True},
    'same_geometry_different_tracker_keypoint_rewards': kp_reward.tolist(),
    'limits': ['No causal training comparison; no measured action or Q statistics.',
               'Target increments are not measured physical joint speeds.',
               'History dependence does not prove LSTM is necessary.']
}, indent=2))
