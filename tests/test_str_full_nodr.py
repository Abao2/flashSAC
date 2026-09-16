import copy
import json
from types import SimpleNamespace

import pytest

from scripts import check_str_full_nodr as probe


def test_restored_config_cpu_and_fresh_output(tmp_path):
    output = tmp_path / "preflight.json"
    probe.main(["--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "PASS" and report["mode"] == "cpu_config"
    assert report["cpu"]["combined_dim"] == 302
    assert report["cpu"]["official_consecutive_success_enabled"] is True
    assert report["cpu"]["success_steps"] == 10
    assert "gpu" not in report
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        probe.main(["--output", str(output)])
    assert output.read_bytes() == saved


@pytest.mark.parametrize("section,key,value", [
    ("action", "arm_moving_average", 1.),
    ("reward", "reach_goal_bonus", 10.),
    ("reset", "fixed_goal_pose", [0., 0., .8, 1., 0., 0., 0.]),
    ("termination", "max_consecutive_successes", 1),
    ("termination", "force_consecutive_near_goal_steps", False),
    ("termination", "success_steps", 1),
    ("termination", "success_steps", 9),
    ("domain_randomization", "joint_velocity_obs_noise_std", .1),
    ("domain_randomization", "object_friction_scale_range", [.8, 1.2]),
])
def test_task_mutations_are_rejected(section, key, value):
    _, original, task = probe.load_config()
    changed = copy.deepcopy(task)
    changed[section][key] = value
    with pytest.raises(AssertionError):
        probe.validate_task(changed, original, probe.observation_sizes())


def test_asset_selection_checks_real_indices_not_only_configuration():
    import torch

    counts = probe.asset_pool_counts()
    paths = [f"{i:03d}_{name}_handle.urdf" for name, count in counts.items() for i in range(count)]
    assert counts == {"hammer": 200, "screwdriver": 200, "marker": 100,
                      "spatula": 200, "eraser": 100, "brush": 400}
    raw = SimpleNamespace(num_envs=1200, _object_urdf_paths=paths,
                          _object_asset_index_per_env=torch.arange(1200))
    assert probe.selected_assets(raw)["all_pool_assets_selected"] == "PASS"
    raw._object_asset_index_per_env = torch.zeros(1200, dtype=torch.long)
    with pytest.raises(AssertionError):
        probe.selected_assets(raw)


def test_success_report_is_durable_before_simulator_exit(tmp_path, monkeypatch):
    output = tmp_path / "gpu.json"

    def exits_after_save(cfg, original, sizes, steps, report, save_report):
        report.update(status="PASS", gpu={"status": "PASS"})
        save_report()
        assert json.loads(output.read_text())["status"] == "PASS"
        raise SystemExit(0)

    monkeypatch.setattr(probe, "gpu_smoke", exits_after_save)
    with pytest.raises(SystemExit):
        probe.main(["--gpu", "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["status"] == "PASS" and "error" not in result
