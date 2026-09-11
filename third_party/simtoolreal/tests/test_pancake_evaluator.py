import json
import math
import unittest
from pathlib import Path

import torch

from dextoolbench.functional_eval import (
    FailureCode,
    PancakeEvaluatorCfg,
    PancakeFunctionalEvaluator,
    is_stable,
    normal_angle_deg,
    quat_rotate_wxyz,
    target_contains,
    update_hold_counter,
)


def _x_rotation(degrees):
    radians = math.radians(degrees)
    return torch.tensor(
        [math.cos(radians / 2.0), math.sin(radians / 2.0), 0.0, 0.0],
        dtype=torch.float32,
    )


def _state(num_envs=1):
    pancake_pos = torch.tensor([[0.0, 0.0, 0.55]]).repeat(num_envs, 1)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)
    return {
        "pancake_pos": pancake_pos,
        "pancake_quat": identity.clone(),
        "pancake_lin_vel": torch.zeros(num_envs, 3),
        "pancake_ang_vel": torch.zeros(num_envs, 3),
        "spatula_pos": pancake_pos
        - torch.tensor([[0.166, 0.0, 0.0]]).repeat(num_envs, 1),
        "spatula_quat": identity.clone(),
    }


class PancakeGeometryTest(unittest.TestCase):
    def _assert_goals_reproduce_desired_blade_path(
        self, filename, max_xy_distance=0.02
    ):
        path = (
            Path(__file__).resolve().parents[1]
            / "dextoolbench/trajectories/spatula/flat_spatula"
            / filename
        )
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        goals = torch.tensor(trajectory["goals"], dtype=torch.float32)
        desired = torch.tensor(trajectory["blade_centers"], dtype=torch.float32)
        offset = torch.tensor(trajectory["spatula_blade_local_offset"]).expand(
            len(goals), -1
        )
        quaternion_wxyz = goals[:, [6, 3, 4, 5]]
        reconstructed = goals[:, :3] + quat_rotate_wxyz(quaternion_wxyz, offset)

        self.assertEqual(len(goals), len(desired))
        torch.testing.assert_close(reconstructed, desired)
        pancake_center_xy = torch.tensor(
            trajectory.get("pancake_center_xy", [0.0, 0.05])
        )
        distance_to_pancake_xy = torch.linalg.vector_norm(
            desired[:, :2] - pancake_center_xy, dim=-1
        )
        self.assertLessEqual(distance_to_pancake_xy.min().item(), max_xy_distance)
        return trajectory, desired

    def test_aligned_scoop_goals_reproduce_desired_blade_path(self):
        _, desired = self._assert_goals_reproduce_desired_blade_path(
            "scoop_and_lift.json"
        )
        self.assertEqual(len(desired), 18)
        self.assertAlmostEqual(desired[:, 2].min().item(), 0.545, places=6)
        self.assertAlmostEqual(
            desired[-1, 2].item() - desired[:, 2].min().item(),
            0.075,
            places=6,
        )

    def test_pitched_scoop_lowers_leading_edge_then_lifts(self):
        trajectory, desired = self._assert_goals_reproduce_desired_blade_path(
            "scoop_and_lift_pitched.json"
        )
        self.assertEqual(len(desired), 18)
        self.assertEqual(max(trajectory["blade_pitch_degrees"]), 15)
        self.assertAlmostEqual(desired[:, 2].min().item(), 0.542, places=6)
        self.assertAlmostEqual(
            desired[-1, 2].item() - desired[:, 2].min().item(),
            0.078,
            places=6,
        )

    def test_pedestal_scoop_lifts_before_reaching_support(self):
        trajectory, desired = self._assert_goals_reproduce_desired_blade_path(
            "scoop_from_pedestal.json", max_xy_distance=0.066
        )
        self.assertAlmostEqual(trajectory["recommended_support_radius"], 0.005)
        self.assertAlmostEqual(
            trajectory["recommended_support_half_spacing"], 0.060
        )
        self.assertEqual(trajectory["scoop_start_goal_index"], 8)
        self.assertEqual(len(desired), 25)
        torch.testing.assert_close(
            desired[:4, 2], torch.full_like(desired[:4, 2], 0.620)
        )
        torch.testing.assert_close(
            desired[:4, 0], torch.tensor([0.00, 0.05, 0.10, 0.15])
        )
        torch.testing.assert_close(
            desired[:4, 1], torch.tensor([0.03, 0.07, 0.12, 0.17])
        )
        self.assertAlmostEqual(desired[12, 0].item(), 0.065, places=6)
        torch.testing.assert_close(desired[13:, 0], torch.zeros_like(desired[13:, 0]))
        self.assertGreater(desired[-1, 2].item(), desired[12, 2].item() + 0.08)

    def test_normal_rotation_gives_zero_half_and_full_flip_progress(self):
        quaternions = torch.stack(
            [_x_rotation(0.0), _x_rotation(90.0), _x_rotation(180.0)]
        )
        local_normal = torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1)
        normals = quat_rotate_wxyz(quaternions, local_normal)
        angles = normal_angle_deg(local_normal, normals)
        torch.testing.assert_close(angles, torch.tensor([0.0, 90.0, 180.0]))
        torch.testing.assert_close(angles / 180.0, torch.tensor([0.0, 0.5, 1.0]))

    def test_target_containment_and_stability_reject_outside_or_fast_state(self):
        positions = torch.tensor([[0.04, 0.00], [0.06, 0.00]])
        center = torch.zeros(2, 2)
        self.assertEqual(
            target_contains(positions, center, 0.05).tolist(), [True, False]
        )

        linear = torch.tensor([[0.01, 0.00, 0.00], [0.06, 0.00, 0.00]])
        angular = torch.zeros(2, 3)
        self.assertEqual(is_stable(linear, angular, 0.05, 0.5).tolist(), [True, False])

    def test_hold_counter_requires_consecutive_frames(self):
        counter = torch.zeros(2, dtype=torch.long)
        counter = update_hold_counter(torch.tensor([True, True]), counter)
        counter = update_hold_counter(torch.tensor([True, False]), counter)
        self.assertEqual(counter.tolist(), [2, 0])


class ScoopAndLiftEvaluatorTest(unittest.TestCase):
    def test_success_requires_height_blade_proximity_drop_check_and_hold(self):
        cfg = PancakeEvaluatorCfg(lift_hold_steps=3, drop_height=0.50)
        evaluator = PancakeFunctionalEvaluator(cfg, num_envs=2)
        initial = _state(2)
        initial["pancake_pos"][1, 2] = 0.45
        initial["spatula_pos"][1, 2] = 0.45
        evaluator.reset([0, 1], initial)

        current = _state(2)
        current["pancake_pos"][0, 2] = 0.59
        current["spatula_pos"][0, 2] = 0.59
        current["pancake_pos"][1, 2] = 0.48
        current["spatula_pos"][1, 2] = 0.48

        for _ in range(2):
            metrics = evaluator.update(current)
        self.assertFalse(metrics["functional_success"][0].item())
        self.assertFalse(metrics["instant_success"][1].item())
        self.assertEqual(
            metrics["failure_code"][1].item(), int(FailureCode.PANCAKE_DROPPED)
        )

        metrics = evaluator.update(current)
        self.assertTrue(metrics["functional_success"][0].item())
        self.assertTrue(metrics["success_ever"][0].item())
        self.assertAlmostEqual(metrics["functional_progress"][0].item(), 1.0)
        self.assertAlmostEqual(metrics["max_lift_height"][0].item(), 0.04, places=5)
        self.assertEqual(metrics["near_blade_steps"][0].item(), 3)
        self.assertFalse(metrics["spatula_never_approached"][0].item())

    def test_far_blade_resets_hold_and_selected_reset_preserves_other_env(self):
        cfg = PancakeEvaluatorCfg(lift_hold_steps=2)
        evaluator = PancakeFunctionalEvaluator(cfg, num_envs=2)
        initial = _state(2)
        evaluator.reset([0, 1], initial)

        current = _state(2)
        current["pancake_pos"][:, 2] += 0.04
        current["spatula_pos"][:, 2] += 0.04
        evaluator.update(current)
        metrics = evaluator.update(current)
        self.assertEqual(metrics["success_ever"].tolist(), [True, True])

        far = {name: value.clone() for name, value in current.items()}
        far["spatula_pos"][0, 0] += 1.0
        metrics = evaluator.update(far)
        self.assertEqual(metrics["success_counter"].tolist(), [0, 3])
        self.assertFalse(metrics["instant_success"][0].item())
        self.assertTrue(metrics["success_ever"][0].item())

        evaluator.reset([0], initial)
        self.assertFalse(evaluator.success_ever[0].item())
        self.assertTrue(evaluator.success_ever[1].item())
        self.assertTrue(torch.isinf(evaluator.min_blade_distance[0]).item())

    def test_never_approached_diagnostic(self):
        evaluator = PancakeFunctionalEvaluator(PancakeEvaluatorCfg(), num_envs=1)
        initial = _state()
        evaluator.reset([0], initial)
        far = {name: value.clone() for name, value in initial.items()}
        far["spatula_pos"][0, 0] += 1.0
        metrics = evaluator.update(far)
        self.assertTrue(metrics["spatula_never_approached"][0].item())
        self.assertEqual(
            metrics["failure_code"][0].item(),
            int(FailureCode.NEVER_CONTACTED_OR_LIFTED),
        )


if __name__ == "__main__":
    unittest.main()
