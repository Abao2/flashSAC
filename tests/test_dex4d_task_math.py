import unittest

import torch

from robotics_tasks.dex4d.task_math import (
    LEGACY_JOINT_NAMES,
    OBS_DIM,
    OBS_SLICES,
    build_observation,
    compute_action_targets,
    compute_metrics,
    compute_reward,
    observation_dim,
    quat_wxyz_to_xyzw,
    transform_keypoints,
)


class Dex4DTaskMathTest(unittest.TestCase):
    def test_layout_and_legacy_joint_order(self) -> None:
        self.assertEqual(len(LEGACY_JOINT_NAMES), 22)
        self.assertEqual(observation_dim(22, 4, 128), OBS_DIM)
        self.assertEqual(LEGACY_JOINT_NAMES[6:], ("1", "0", "2", "3", "12", "13", "14", "15", "5", "4", "6", "7", "9", "8", "10", "11"))
        self.assertEqual(
            {name: (value.start, value.stop) for name, value in OBS_SLICES.items()},
            {
                "joint_pos": (0, 22), "joint_vel": (22, 44), "joint_force": (44, 66),
                "fingertip_states": (66, 118), "fingertip_wrenches": (118, 142),
                "palm_state": (142, 155), "actions": (155, 177), "object_pos": (177, 180),
                "object_quat": (180, 184), "object_linvel": (184, 187), "object_angvel": (187, 190),
                "goal_delta_pos": (190, 193), "goal_delta_quat": (193, 197),
                "object_keypoints": (197, 581), "goal_keypoints": (581, 965),
                "visual_features": (965, 1029), "fingertip_object_vectors": (1029, OBS_DIM),
            },
        )

    def test_wxyz_conversion_and_keypoint_transform(self) -> None:
        root_half = 2.0**-0.5
        quaternion = torch.tensor([[root_half, 0.0, 0.0, root_half]])
        self.assertTrue(torch.allclose(quat_wxyz_to_xyzw(quaternion), torch.tensor([[0.0, 0.0, root_half, root_half]])))
        point = torch.tensor([[[1.0, 0.0, 0.0]]])
        transformed = transform_keypoints(point, torch.tensor([[1.0, 2.0, 3.0]]), quaternion)
        self.assertTrue(torch.allclose(transformed, torch.tensor([[[1.0, 3.0, 3.0]]]), atol=1e-6))

    def test_observation_offsets(self) -> None:
        batch = 1
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        tip_states = torch.zeros(batch, 4, 13)
        tip_states[..., 3] = 1.0
        palm_state = torch.zeros(batch, 13)
        palm_state[..., 3] = 1.0
        palm_state[..., 10:13] = 5.0
        actions = torch.arange(22, dtype=torch.float32).unsqueeze(0)
        object_pos = torch.tensor([[1.0, 2.0, 3.0]])
        goal_pos = torch.tensor([[1.0, 2.0, 4.0]])
        observation = build_observation(
            joint_pos=torch.zeros(batch, 22),
            joint_vel=torch.ones(batch, 22),
            joint_force=torch.ones(batch, 22),
            joint_lower=-torch.ones(22),
            joint_upper=torch.ones(22),
            fingertip_states_wxyz=tip_states,
            fingertip_wrenches=torch.ones(batch, 4, 6),
            fingertip_positions=torch.zeros(batch, 4, 3),
            palm_state_wxyz=palm_state,
            actions=actions,
            object_pos=object_pos,
            object_quat_wxyz=identity,
            object_linvel=torch.full((batch, 3), 2.0),
            object_angvel=torch.full((batch, 3), 5.0),
            goal_pos=goal_pos,
            goal_quat_wxyz=identity,
            local_keypoints=torch.zeros(batch, 128, 3),
            visual_features=torch.ones(batch, 64),
        )
        self.assertEqual(observation.shape, (batch, OBS_DIM))
        self.assertTrue(torch.equal(observation[:, OBS_SLICES["actions"]], actions))
        self.assertTrue(torch.allclose(observation[:, OBS_SLICES["joint_vel"]], torch.full((1, 22), 0.2)))
        self.assertTrue(torch.allclose(observation[:, OBS_SLICES["joint_force"]], torch.full((1, 22), 10.0)))
        self.assertTrue(torch.allclose(observation[:, OBS_SLICES["object_quat"]], torch.tensor([[0.0, 0.0, 0.0, 1.0]])))
        self.assertTrue(torch.allclose(observation[:, OBS_SLICES["goal_delta_pos"]], torch.tensor([[0.0, 0.0, 1.0]])))
        self.assertTrue(torch.allclose(observation[:, OBS_SLICES["visual_features"]], torch.full((1, 64), 0.1)))
        self.assertTrue(torch.allclose(observation[:, OBS_SLICES["fingertip_object_vectors"]], object_pos.repeat(1, 4)))
        self.assertTrue(torch.allclose(observation[:, OBS_SLICES["palm_state"]][:, 10:13], torch.ones(1, 3)))

    def test_reward_terms(self) -> None:
        object_keypoints = torch.zeros(1, 128, 3)
        goal_keypoints = object_keypoints.clone()
        goal_keypoints[..., 2] = 0.02
        metrics = compute_metrics(
            object_keypoints=object_keypoints,
            goal_keypoints=goal_keypoints,
            object_pos=torch.tensor([[0.0, 0.0, 0.7]]),
            palm_pos=torch.tensor([[0.0, 0.0, 0.8]]),
            fingertip_positions=torch.tensor([[[0.1, 0.0, 0.7]]]).repeat(1, 4, 1),
            hand_joint_pos=torch.zeros(1, 16),
        )
        weights = {name: 1.0 for name in (
            "obj_finger", "obj_hand", "goal_obj", "success_bonus", "terminal_bonus",
            "finger_curl_reg", "table_collision_penalty", "action_penalty",
        )}
        reward, terms = compute_reward(metrics, torch.zeros(1, 22), torch.tensor([30]), weights)
        self.assertTrue(metrics["grasped"].item())
        self.assertAlmostEqual(terms["obj_finger"].item(), -0.2, places=6)
        self.assertAlmostEqual(terms["obj_hand"].item(), -0.05, places=6)
        self.assertAlmostEqual(terms["goal_obj"].item(), 1.34, places=6)
        self.assertAlmostEqual(terms["success_bonus"].item(), 5.0 / 1.2, places=6)
        self.assertEqual(terms["terminal_bonus"].item(), 10.0)
        self.assertAlmostEqual(reward.item(), -0.25 + 1.34 + 5.0 / 1.2 + 10.0, places=5)

    def test_action_targets(self) -> None:
        lower = torch.cat((-torch.full((6,), 10.0), torch.zeros(16)))
        upper = torch.cat((torch.full((6,), 10.0), torch.full((16,), 2.0)))
        targets = compute_action_targets(
            torch.cat((torch.ones(1, 6), torch.zeros(1, 16)), dim=-1),
            torch.zeros(1, 22),
            lower,
            upper,
            speed_scale=10.0,
        )
        self.assertTrue(torch.allclose(targets[:, :6], torch.full((1, 6), 1.0 / 6.0)))
        self.assertTrue(torch.equal(targets[:, 6:], torch.ones(1, 16)))

    def test_wuji_m6_dimensions_and_controls(self) -> None:
        batch, num_dofs, num_fingertips, num_keypoints = 1, 27, 5, 128
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        tip_states = torch.zeros(batch, num_fingertips, 13)
        tip_states[..., 3] = 1.0
        palm_state = torch.zeros(batch, 13)
        palm_state[..., 3] = 1.0
        observation = build_observation(
            joint_pos=torch.zeros(batch, num_dofs),
            joint_vel=torch.zeros(batch, num_dofs),
            joint_force=torch.zeros(batch, num_dofs),
            joint_lower=-torch.ones(num_dofs),
            joint_upper=torch.ones(num_dofs),
            fingertip_states_wxyz=tip_states,
            fingertip_wrenches=torch.zeros(batch, num_fingertips, 6),
            fingertip_positions=torch.zeros(batch, num_fingertips, 3),
            palm_state_wxyz=palm_state,
            actions=torch.zeros(batch, num_dofs),
            object_pos=torch.zeros(batch, 3),
            object_quat_wxyz=identity,
            object_linvel=torch.zeros(batch, 3),
            object_angvel=torch.zeros(batch, 3),
            goal_pos=torch.zeros(batch, 3),
            goal_quat_wxyz=identity,
            local_keypoints=torch.zeros(batch, num_keypoints, 3),
            visual_features=torch.zeros(batch, 64),
        )
        self.assertEqual(observation_dim(num_dofs, num_fingertips, num_keypoints), 1083)
        self.assertEqual(observation.shape, (batch, 1083))

        lower = torch.cat((-torch.full((7,), 10.0), torch.zeros(20)))
        upper = torch.cat((torch.full((7,), 10.0), torch.full((20,), 2.0)))
        targets = compute_action_targets(
            torch.cat((torch.ones(1, 7), torch.zeros(1, 20)), dim=-1),
            torch.zeros(1, num_dofs),
            lower,
            upper,
            speed_scale=10.0,
            moving_average=0.5,
            num_arm_dofs=7,
            hand_moving_average=0.25,
        )
        self.assertTrue(torch.allclose(targets[:, :7], torch.full((1, 7), 1.0 / 12.0)))
        self.assertTrue(torch.equal(targets[:, 7:], torch.full((1, 20), 0.25)))

        metrics = compute_metrics(
            object_keypoints=torch.zeros(batch, num_keypoints, 3),
            goal_keypoints=torch.zeros(batch, num_keypoints, 3),
            object_pos=torch.zeros(batch, 3),
            palm_pos=torch.zeros(batch, 3),
            fingertip_positions=torch.tensor([[[0.1, 0.0, 0.0]]]).repeat(batch, num_fingertips, 1),
            hand_joint_pos=torch.zeros(batch, 20),
            finger_distance_scale=4.0 / 5.0,
        )
        self.assertAlmostEqual(metrics["obj_finger_dist"].item(), 0.4, places=6)
        self.assertTrue(metrics["grasped"].item())


if __name__ == "__main__":
    unittest.main()
