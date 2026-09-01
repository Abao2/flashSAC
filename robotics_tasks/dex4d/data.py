"""Dex4D object assets without importing Isaac Lab.

Catalog merging, filtering, and asset paths are adapted from Dex4D-Simulation
commit 0dfccd82797e929cda6ab665003bc27a6b8cd31a (Apache-2.0), with changes for
deterministic caching and Isaac Sim-safe robot import.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import trimesh
import yaml

_SCALE_STRINGS = {0.06: "006", 0.08: "008", 0.10: "010", 0.12: "012", 0.15: "015"}
# Dex4D builds this mapping by iterating a Python set.  This is the order
# produced by the official CPython environment for the five supported scales.
_LEGACY_SCALE_IDS = {scale: index for index, scale in enumerate((0.06, 0.12, 0.10, 0.08, 0.15))}


class _MergeLoader(yaml.SafeLoader):
    """Match Dex4D's MergeLoader: repeated mapping keys extend list values."""


def _construct_mapping(loader: _MergeLoader, node: yaml.MappingNode) -> dict:
    loader.flatten_mapping(node)
    result: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=False)
        value = loader.construct_object(value_node, deep=True)
        if key in result:
            previous = result[key] if isinstance(result[key], list) else [result[key]]
            result[key] = previous + (value if isinstance(value, list) else [value])
        else:
            result[key] = value
    return result


_MergeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


@dataclass(frozen=True)
class ObjectSpec:
    code: str
    scale: float
    scale_str: str
    urdf_path: Path
    mesh_path: Path
    feature_path: Path
    code_index: int = 0
    scale_id: int = 0

    @property
    def keypoint_seed(self) -> int:
        """Seed passed to Dex4D's farthest-point sampler."""
        return self.code_index * 10 + self.scale_id


def resolve_dex4d_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    value = explicit or os.environ.get("DEX4D_ROOT")
    if not value:
        raise RuntimeError("Set DEX4D_ROOT to the Dex4D-Simulation repository")
    return Path(value).expanduser().resolve()


def _cache_root(cache_root: str | os.PathLike[str] | None) -> Path:
    value = cache_root or os.environ.get("DEX4D_CACHE_ROOT")
    return Path(value).expanduser().resolve() if value else Path.home() / ".cache/flashsac/dex4d"


def _file_key(path: Path, *parameters: object) -> str:
    stat = path.stat()
    payload = "\0".join((str(path.resolve()), str(stat.st_mtime_ns), str(stat.st_size), *(map(str, parameters))))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _scale_str(scale: float) -> str:
    rounded = round(float(scale), 2)
    try:
        return _SCALE_STRINGS[rounded]
    except KeyError as exc:
        raise ValueError(f"Unsupported Dex4D scale: {scale}") from exc


def _object_class(code: str) -> str:
    try:
        return code.split("/", 1)[1].split("-", 1)[0].lower()
    except IndexError as exc:
        raise ValueError(f"Invalid Dex4D object code: {code}") from exc


def load_object_specs(
    root: str | os.PathLike[str] | None = None,
    manifest: str | os.PathLike[str] | None = None,
    *,
    include_classes: Sequence[str] | None = None,
    exclude_classes: Sequence[str] | None = None,
    object_count: int | None = None,
) -> list[ObjectSpec]:
    """Load unique code-scale pairs in official catalog insertion order."""
    dex4d_root = resolve_dex4d_root(root)
    catalog_path = Path(manifest or "dex4d_policy/dex4d/cfg/train_set.yaml").expanduser()
    if not catalog_path.is_absolute():
        catalog_path = dex4d_root / catalog_path
    with catalog_path.open(encoding="utf-8") as stream:
        document = yaml.load(stream, Loader=_MergeLoader)
    scale_map = document.get("object_code_dict") if isinstance(document, Mapping) else None
    if not isinstance(scale_map, Mapping):
        raise ValueError(f"Missing object_code_dict in {catalog_path}")

    include_set = None if include_classes is None else {name.lower() for name in include_classes}
    exclude_set = {name.lower() for name in (exclude_classes or ())}
    entries = [
        (str(code), scales)
        for code, scales in scale_map.items()
        if (include_set is None or _object_class(str(code)) in include_set)
        and _object_class(str(code)) not in exclude_set
    ]
    if object_count is not None:
        if isinstance(object_count, bool) or not isinstance(object_count, Integral) or object_count < 0:
            raise ValueError("object_count must be a non-negative integer")
        entries = [(code, scales[:1]) for code, scales in entries[:object_count]]

    assets = dex4d_root / "dex4d_policy/assets"
    specs: list[ObjectSpec] = []
    seen: set[tuple[str, str]] = set()
    for code_index, (code, scales) in enumerate(entries):
        if not isinstance(scales, Sequence) or isinstance(scales, (str, bytes)):
            scales = [scales]
        for raw_scale in scales:
            if not isinstance(raw_scale, Real):
                raise ValueError(f"Invalid scale for {code}: {raw_scale!r}")
            scale = float(raw_scale)
            scale_str = _scale_str(scale)
            if (code, scale_str) in seen:
                continue
            seen.add((code, scale_str))
            object_dir = assets / "meshdatav3_scaled" / code / "coacd"
            specs.append(
                ObjectSpec(
                    code=code,
                    scale=scale,
                    scale_str=scale_str,
                    urdf_path=object_dir / f"coacd_{scale_str}.urdf",
                    mesh_path=object_dir / f"decomposed_{scale_str}.obj",
                    feature_path=assets / "meshdatav3_pc_feat" / code / f"pc_feat_{scale_str}.npy",
                    code_index=code_index,
                    scale_id=_LEGACY_SCALE_IDS[round(scale, 2)],
                )
            )
    return specs


def assign_object_specs(num_envs: int, specs: Sequence[ObjectSpec]) -> tuple[np.ndarray, list[ObjectSpec]]:
    """Reproduce Dex4D's code-stratified per-environment assignment.

    The official task spreads object *codes* over the environment index, then
    selects a scale with ``env_id % number_of_scales_for_that_code``.
    """
    if num_envs < 0 or not specs:
        raise ValueError("num_envs must be non-negative and specs must not be empty")
    indices_by_code: dict[str, list[int]] = {}
    for index, spec in enumerate(specs):
        indices_by_code.setdefault(spec.code, []).append(index)
    groups = list(indices_by_code.values())
    indices = np.empty(num_envs, dtype=np.int64)
    for env_id in range(num_envs):
        code_id = int(env_id / num_envs * len(groups))
        scales = groups[code_id]
        indices[env_id] = scales[env_id % len(scales)]
    return indices, [specs[index] for index in indices]


def validate_object_specs(specs: Sequence[ObjectSpec]) -> None:
    """Fail before simulator startup when assigned external assets are incomplete."""
    missing = sorted(
        {
            path
            for spec in specs
            for path in (spec.urdf_path, spec.mesh_path, spec.feature_path)
            if not path.is_file()
        }
    )
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        raise FileNotFoundError(f"Dex4D assigned assets are incomplete ({len(missing)} missing):\n{preview}{suffix}")


def load_pc_feature(spec: ObjectSpec) -> np.ndarray:
    feature = np.asarray(np.load(spec.feature_path, allow_pickle=False), dtype=np.float32).reshape(-1)
    if feature.shape != (64,):
        raise ValueError(f"Expected a 64D pc_feat, got {feature.shape}: {spec.feature_path}")
    return feature


def _farthest_points(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    selected = np.empty(count, dtype=np.int64)
    selected[0] = np.random.default_rng(seed).integers(len(points))
    distance = np.full(len(points), np.inf)
    for index in range(1, count):
        delta = points - points[selected[index - 1]]
        distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        selected[index] = int(np.argmax(distance))
    return points[selected]


def load_keypoints(
    spec: ObjectSpec,
    num_points: int = 128,
    seed: int = 42,
    cache_root: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Deterministically surface-sample an OBJ, apply FPS, and cache it."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    key = _file_key(spec.mesh_path, "surface-fps-v1", num_points, seed)
    cache_dir = _cache_root(cache_root) / "keypoints"
    cache_path = cache_dir / f"{spec.mesh_path.stem}-{key}.npy"
    if cache_path.exists():
        cached = np.asarray(np.load(cache_path, allow_pickle=False), dtype=np.float32)
        if cached.shape == (num_points, 3):
            return cached

    mesh = trimesh.load(spec.mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"OBJ has no triangle surface: {spec.mesh_path}")
    surface, _ = trimesh.sample.sample_surface(mesh, max(10_000, num_points), seed=seed)
    keypoints = _farthest_points(np.asarray(surface), num_points, seed).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, keypoints)
    os.replace(temporary, cache_path)
    return keypoints


def load_object_observation_data(
    specs: Sequence[ObjectSpec],
    assignment_indices: Sequence[int] | np.ndarray,
    num_keypoints: int = 128,
    seed: int = 42,
    cache_root: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load 64D features and keypoints in per-environment assignment order."""
    if not specs:
        raise ValueError("specs must not be empty")
    indices = np.asarray(assignment_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(specs)):
        raise ValueError("assignment_indices must be a 1D array indexing specs")
    # Only touch assets that are actually assigned.  This keeps a 64-env smoke
    # from sampling all 3,200 catalog entries while retaining exact env order.
    unique_indices = tuple(dict.fromkeys(int(index) for index in indices))
    feature_cache = {index: load_pc_feature(specs[index]) for index in unique_indices}
    keypoint_cache = {
        index: load_keypoints(
            specs[index],
            num_keypoints,
            seed + specs[index].keypoint_seed,
            cache_root,
        )
        for index in unique_indices
    }
    return (
        np.stack([feature_cache[int(index)] for index in indices]),
        np.stack([keypoint_cache[int(index)] for index in indices]),
    )


def prepare_sanitized_robot_urdf(
    root: str | os.PathLike[str] | None = None,
    robot_relpath: str | os.PathLike[str] = (
        "dex4d_policy/assets/urdf/xarm6_leap_description/xarm6_leap_right_2023.urdf"
    ),
    cache_root: str | os.PathLike[str] | None = None,
    *,
    rename_numeric_leap_joints: bool = True,
) -> Path:
    """Cache an importer-safe robot URDF without touching the official file."""
    source = Path(robot_relpath).expanduser()
    if not source.is_absolute():
        source = resolve_dex4d_root(root) / source
    mode = "numeric-joints" if rename_numeric_leap_joints else "named-joints"
    key = _file_key(source, f"{mode}-and-absolute-meshes-v2")
    output_dir = _cache_root(cache_root) / "urdf"
    output = output_dir / f"{source.stem}-{key}.urdf"
    if output.exists():
        return output

    tree = ET.parse(source)
    joints = tree.getroot().findall(".//joint")
    if rename_numeric_leap_joints:
        numeric = [joint for joint in joints if (joint.get("name") or "").isdigit()]
        names = [joint.get("name") for joint in numeric]
        if len(names) != 16 or set(names) != {str(index) for index in range(16)}:
            raise ValueError(f"Expected numeric LEAP joints 0..15 in {source}, got {names}")
        for joint in numeric:
            joint.set("name", f"leap_joint_{joint.get('name')}")
    for mesh in tree.getroot().findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and "://" not in filename and not Path(filename).is_absolute():
            mesh.set("filename", str((source.parent / filename).resolve()))

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    os.replace(temporary, output)
    return output
