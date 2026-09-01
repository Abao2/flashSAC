import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np

from robotics_tasks.dex4d.data import (
    ObjectSpec,
    assign_object_specs,
    load_keypoints,
    load_object_observation_data,
    load_object_specs,
    load_pc_feature,
    prepare_sanitized_robot_urdf,
    validate_object_specs,
)


class Dex4DDataTest(unittest.TestCase):
    def _root(self, directory: str) -> Path:
        root = Path(directory)
        cfg = root / "dex4d_policy/dex4d/cfg"
        cfg.mkdir(parents=True)
        cfg.joinpath("train_set.yaml").write_text(
            """object_code_dict: {
  'sem/Bottle-a':[0.06],
  'core/mug-b':[0.08, 0.10],
  'sem/Bottle-a':[0.08, 0.06],
  'sem/Camera-c':[0.12],
}
""",
            encoding="utf-8",
        )
        return root

    def test_catalog_merges_duplicate_keys_filters_and_keeps_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            with patch.dict(os.environ, {"DEX4D_ROOT": str(root)}):
                specs = load_object_specs()
            self.assertEqual(
                [(spec.code, spec.scale_str) for spec in specs],
                [
                    ("sem/Bottle-a", "006"),
                    ("sem/Bottle-a", "008"),
                    ("core/mug-b", "008"),
                    ("core/mug-b", "010"),
                    ("sem/Camera-c", "012"),
                ],
            )
            self.assertTrue(specs[0].urdf_path.is_absolute())
            self.assertEqual(specs[0].mesh_path.name, "decomposed_006.obj")
            self.assertEqual(specs[0].feature_path.name, "pc_feat_006.npy")
            self.assertEqual(
                [(spec.code, spec.scale_str) for spec in load_object_specs(root, include_classes=["MUG"])],
                [("core/mug-b", "008"), ("core/mug-b", "010")],
            )
            self.assertEqual(
                load_object_specs(root, include_classes=["bottle"], exclude_classes=["BOTTLE"]), []
            )
            self.assertEqual(
                [(spec.code, spec.scale_str) for spec in load_object_specs(root, object_count=2)],
                [("sem/Bottle-a", "006"), ("core/mug-b", "008")],
            )

    def test_official_catalog_regression_when_available(self) -> None:
        root = os.environ.get("DEX4D_ROOT")
        if not root:
            self.skipTest("DEX4D_ROOT is not set")
        specs = load_object_specs(root)
        self.assertEqual(len({spec.code for spec in specs}), 2145)
        self.assertEqual(len(specs), 3200)
        bottles = load_object_specs(root, include_classes=["bottle"])
        self.assertEqual(len({spec.code for spec in bottles}), 418)
        self.assertEqual(len(bottles), 735)

    def test_official_code_stratified_assignment_and_pc_feature(self) -> None:
        specs = [
            ObjectSpec(f"core/test-{index}", 0.06, "006", Path("a"), Path("b"), Path("c"))
            for index in range(3)
        ]
        indices, assigned = assign_object_specs(7, specs)
        np.testing.assert_array_equal(indices, [0, 0, 0, 1, 1, 2, 2])
        self.assertEqual([spec.code for spec in assigned], [specs[index].code for index in indices])
        multi_scale = [
            ObjectSpec("core/a", 0.06, "006", Path("a"), Path("b"), Path("c")),
            ObjectSpec("core/a", 0.08, "008", Path("a"), Path("b"), Path("c")),
            ObjectSpec("core/b", 0.06, "006", Path("a"), Path("b"), Path("c")),
        ]
        multi_indices, _ = assign_object_specs(4, multi_scale)
        np.testing.assert_array_equal(multi_indices, [0, 1, 2, 2])
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "feature.npy"
            np.save(feature_path, np.arange(64, dtype=np.float64))
            spec = ObjectSpec("core/mug-a", 0.06, "006", Path("a"), Path("b"), feature_path)
            feature = load_pc_feature(spec)
            self.assertEqual(feature.shape, (64,))
            self.assertEqual(feature.dtype, np.float32)

    def test_assigned_asset_preflight_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("object.urdf", "object.obj", "feature.npy")]
            spec = ObjectSpec("core/test-a", 0.06, "006", *paths)
            with self.assertRaisesRegex(FileNotFoundError, "3 missing"):
                validate_object_specs([spec, spec])
            for path in paths:
                path.touch()
            validate_object_specs([spec])

    def test_keypoints_are_deterministic_and_cache_is_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "tetra.obj"
            mesh.write_text(
                """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
""",
                encoding="utf-8",
            )
            spec = ObjectSpec("core/test-a", 0.06, "006", root / "x.urdf", mesh, root / "x.npy")
            np.save(spec.feature_path, np.arange(64, dtype=np.float32))
            cache = root / "cache"
            first = load_keypoints(spec, num_points=8, seed=7, cache_root=cache)
            second = load_keypoints(spec, num_points=8, seed=7, cache_root=cache)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, (8, 3))
            self.assertEqual(len(list(cache.rglob("*.npy"))), 1)
            load_keypoints(spec, num_points=9, seed=7, cache_root=cache)
            load_keypoints(spec, num_points=8, seed=8, cache_root=cache)
            self.assertEqual(len(list(cache.rglob("*.npy"))), 3)
            mesh.write_text(mesh.read_text(encoding="utf-8") + "# cache invalidation\n", encoding="utf-8")
            load_keypoints(spec, num_points=8, seed=7, cache_root=cache)
            self.assertEqual(len(list(cache.rglob("*.npy"))), 4)
            features, keypoints = load_object_observation_data(
                [spec], [0, 0], num_keypoints=8, seed=7, cache_root=cache
            )
            self.assertEqual(features.shape, (2, 64))
            self.assertEqual(keypoints.shape, (2, 8, 3))
            np.testing.assert_array_equal(keypoints[0], keypoints[1])

    def test_robot_urdf_is_sanitized_in_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf_dir = root / "dex4d_policy/assets/urdf/xarm6_leap_description"
            mesh_dir = urdf_dir / "meshes"
            mesh_dir.mkdir(parents=True)
            mesh_dir.joinpath("part.stl").write_text("mesh", encoding="utf-8")
            joints = "\n".join(f'<joint name="{index}" type="fixed" />' for index in range(16))
            source = urdf_dir / "xarm6_leap_right_2023.urdf"
            original = f'<robot name="test">{joints}<link name="x"><visual><geometry><mesh filename="meshes/part.stl" /></geometry></visual></link></robot>'
            source.write_text(original, encoding="utf-8")

            output = prepare_sanitized_robot_urdf(root, cache_root=root / "cache")
            self.assertNotEqual(output, source)
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            tree = ET.parse(output)
            names = [joint.get("name") for joint in tree.getroot().findall(".//joint")]
            self.assertEqual(set(names), {f"leap_joint_{index}" for index in range(16)})
            self.assertFalse(any((name or "").isdigit() for name in names))
            meshes = [mesh.get("filename") for mesh in tree.getroot().findall(".//mesh")]
            self.assertTrue(meshes and all(Path(filename).is_absolute() for filename in meshes if filename))
            self.assertEqual(prepare_sanitized_robot_urdf(root, cache_root=root / "cache"), output)
            source.write_text(original + "\n", encoding="utf-8")
            self.assertNotEqual(prepare_sanitized_robot_urdf(root, cache_root=root / "cache"), output)


if __name__ == "__main__":
    unittest.main()
