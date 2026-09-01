"""Isaac Lab scene construction for the Dex4D XArm6 + LEAP task."""

from __future__ import annotations

import fcntl
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, NamedTuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

LEGACY_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "leap_joint_1",
    "leap_joint_0",
    "leap_joint_2",
    "leap_joint_3",
    "leap_joint_12",
    "leap_joint_13",
    "leap_joint_14",
    "leap_joint_15",
    "leap_joint_5",
    "leap_joint_4",
    "leap_joint_6",
    "leap_joint_7",
    "leap_joint_9",
    "leap_joint_8",
    "leap_joint_10",
    "leap_joint_11",
)
FINGERTIP_BODY_NAMES = ("fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip")

_ARM_STIFFNESS = dict(zip((f"joint{i}" for i in range(1, 7)), (100.0, 100.0, 64.0, 64.0, 64.0, 40.0)))
_ARM_EFFORT = dict(zip((f"joint{i}" for i in range(1, 7)), (50.0, 50.0, 32.0, 32.0, 32.0, 20.0)))
_M6_ARM_JOINTS = tuple(f"Joint{i}_L" for i in range(1, 8))
_WUJI_HAND_JOINTS = tuple(
    f"left_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)
_WUJI_TIPS = tuple(f"left_finger{i}_tip_link" for i in range(1, 6))
_WUJI_HAND_STIFFNESS = dict(zip(
    _WUJI_HAND_JOINTS,
    (
        0.40844710645048043, 0.6858601063643346, 0.2391112196099482, 0.20736128319711747,
        0.37352218155860073, 0.45592448909027794, 0.24368366649522863, 0.18026971340925335,
        0.3687093483646485, 0.4164253443634641, 0.22218607502059182, 0.19427606072023446,
        0.35718151495111794, 0.42977315313086895, 0.24930151196247122, 0.2285032688178066,
        0.3655325975433942, 0.41393113081120425, 0.22729367621965954, 0.1964723550341816,
    ),
))
_WUJI_HAND_DAMPING = dict(zip(
    _WUJI_HAND_JOINTS,
    (
        0.020882010257063675, 0.030610939996373314, 0.010181475050560962, 0.00909698844675045,
        0.01882274330029718, 0.019798167597016643, 0.010477031953162727, 0.008240212147903584,
        0.01848622487024593, 0.018032947229953678, 0.009592200014076666, 0.009152994605972402,
        0.018376606800780005, 0.01867700966212433, 0.01059512121009555, 0.009917602877441107,
        0.018616960278988272, 0.018732177029667153, 0.00951005441616486, 0.009017295241756363,
    ),
))
_WUJI_HAND_ARMATURE = {
    name: (0.0005 if name.endswith("joint1") or name == "left_finger1_joint2" else 0.0002)
    for name in _WUJI_HAND_JOINTS
}
_M6_WUJI_FILTER_PAIRS = tuple(
    ("left_palm_link", f"left_finger{finger}_link{link}")
    for finger in range(1, 6)
    for link in (1, 2)
) + (("Link5_L", "Link7_L"),)
_MATERIAL_PATH = "/World/Materials/Dex4D"


class RobotProfile(NamedTuple):
    name: str
    joint_names: tuple[str, ...]
    arm_joint_names: tuple[str, ...]
    hand_joint_names: tuple[str, ...]
    arm_joint_expr: str
    hand_joint_expr: str
    palm_body_name: str
    fingertip_body_names: tuple[str, ...]
    palm_body_offset_xyz: tuple[float, float, float]
    palm_body_offset_rpy: tuple[float, float, float]
    palm_task_offset: tuple[float, float, float]
    fingertip_offsets: tuple[tuple[float, float, float], ...]
    root_pos: tuple[float, float, float]
    root_rot: tuple[float, float, float, float]
    arm_stiffness: dict[str, float]
    arm_damping: dict[str, float] | float
    hand_stiffness: dict[str, float] | float
    hand_damping: dict[str, float] | float
    hand_armature: dict[str, float] | float
    arm_effort: dict[str, float] | float | None
    arm_velocity: dict[str, float] | float | None
    hand_effort: dict[str, float] | float | None
    hand_velocity: dict[str, float] | float | None
    merge_fixed_joints: bool
    import_self_collision: bool
    enabled_self_collisions: bool
    self_collision_filter_pairs: tuple[tuple[str, str], ...]
    rename_numeric_leap_joints: bool
    table_size: tuple[float, float, float]
    table_pos: tuple[float, float, float]

    @property
    def num_arm_dofs(self) -> int:
        return len(self.arm_joint_names)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


ROBOT_PROFILES = {
    "xarm6_leap": RobotProfile(
        name="xarm6_leap",
        joint_names=LEGACY_JOINT_NAMES,
        arm_joint_names=LEGACY_JOINT_NAMES[:6],
        hand_joint_names=LEGACY_JOINT_NAMES[6:],
        arm_joint_expr="joint[1-6]",
        hand_joint_expr="leap_joint_.*",
        palm_body_name="link6",
        fingertip_body_names=FINGERTIP_BODY_NAMES,
        palm_body_offset_xyz=(0.0, 0.0, -0.045),
        palm_body_offset_rpy=(3.14, 0.0, 0.0),
        palm_task_offset=(0.04, 0.0, -0.01),
        fingertip_offsets=(
            (-0.01, -0.035, 0.015),
            (-0.01, -0.035, 0.015),
            (-0.01, -0.035, 0.015),
            (-0.01, -0.050, -0.015),
        ),
        root_pos=(-0.5, 0.0, 0.6),
        root_rot=(1.0, 0.0, 0.0, 0.0),
        arm_stiffness=_ARM_STIFFNESS,
        arm_damping=1.0,
        hand_stiffness=3.0,
        hand_damping=0.5,
        hand_armature=0.0,
        arm_effort=_ARM_EFFORT,
        arm_velocity=3.14,
        hand_effort=0.95,
        hand_velocity=8.48,
        merge_fixed_joints=True,
        import_self_collision=False,
        enabled_self_collisions=False,
        self_collision_filter_pairs=(),
        rename_numeric_leap_joints=True,
        table_size=(1.2, 1.2, 0.6),
        table_pos=(0.0, 0.0, 0.3),
    ),
    "m6_wuji_left": RobotProfile(
        name="m6_wuji_left",
        joint_names=_M6_ARM_JOINTS + _WUJI_HAND_JOINTS,
        arm_joint_names=_M6_ARM_JOINTS,
        hand_joint_names=_WUJI_HAND_JOINTS,
        arm_joint_expr="Joint[1-7]_L",
        hand_joint_expr="left_finger[1-5]_joint[1-4]",
        palm_body_name="left_palm_link",
        fingertip_body_names=_WUJI_TIPS,
        palm_body_offset_xyz=(0.0, 0.0, 0.0),
        palm_body_offset_rpy=(0.0, 0.0, 0.0),
        palm_task_offset=(0.0, 0.0, 0.0),
        fingertip_offsets=((0.0, 0.0, 0.0),) * 5,
        root_pos=(-0.4792, -0.1431, 0.0),
        root_rot=(1.0, 0.0, 0.0, 0.0),
        arm_stiffness={name: 2000.0 for name in _M6_ARM_JOINTS},
        arm_damping={name: 100.0 for name in _M6_ARM_JOINTS},
        hand_stiffness=_WUJI_HAND_STIFFNESS,
        hand_damping=_WUJI_HAND_DAMPING,
        hand_armature=_WUJI_HAND_ARMATURE,
        # Preserve effort/velocity limits authored in the validated URDF.
        arm_effort=None,
        arm_velocity=None,
        hand_effort=None,
        hand_velocity=None,
        merge_fixed_joints=False,
        import_self_collision=True,
        enabled_self_collisions=True,
        self_collision_filter_pairs=_M6_WUJI_FILTER_PAIRS,
        rename_numeric_leap_joints=False,
        # Same narrow tabletop footprint validated with the stand_v3 root;
        # top remains z=0.6 so Dex4D object/reset/reward geometry is unchanged.
        table_size=(0.475, 0.4, 0.3),
        table_pos=(0.0, 0.0, 0.45),
    ),
}


def get_robot_profile(name: str) -> RobotProfile:
    try:
        return ROBOT_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown Dex4D robot profile {name!r}; expected {tuple(ROBOT_PROFILES)}") from error


class RobotContract(NamedTuple):
    """Indices needed to preserve Dex4D's legacy tensor ordering."""

    joint_ids: tuple[int, ...]
    palm_body_id: int
    fingertip_body_ids: tuple[int, ...]


def _validate_robot_urdf(path: Path, profile: RobotProfile) -> None:
    names = {joint.get("name", "") for joint in ET.parse(path).getroot().findall(".//joint")}
    missing = set(profile.joint_names) - names
    if missing or (profile.rename_numeric_leap_joints and any(name.isdigit() for name in names)):
        raise ValueError(
            f"Robot URDF does not satisfy profile {profile.name!r}: missing={sorted(missing)} path={path}"
        )


def _rigid_props(
    disable_gravity: bool, *, kinematic: bool = False, damping: float = 0.0
) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=kinematic,
        disable_gravity=disable_gravity,
        enable_gyroscopic_forces=True,
        linear_damping=damping,
        angular_damping=damping,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
        max_depenetration_velocity=1000.0,
    )


def _collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.001, rest_offset=0.0)


def _spawn_object_specs(env: Any) -> tuple[Any, ...]:
    assigned = tuple(env._assigned_object_specs)
    if len(assigned) != env.cfg.scene.num_envs:
        raise ValueError("Dex4D object assignment does not match scene.num_envs")
    # The smoke path deliberately uses one object.  A single prototype is
    # sufficient because MultiAssetSpawner repeats its list periodically.
    if assigned and all(spec == assigned[0] for spec in assigned):
        return assigned[:1]
    return assigned


def _author_collision_properties(usd_path: str) -> None:
    """Bake Dex4D contact offsets into converter physics layers.

    Isaac Sim makes imported visual/collision geometry instanceable.  Applying
    collision overrides after spawning therefore skips instance proxies.  The
    generated ``*_physics.usd`` layer is the shared, editable source of those
    proxies, so authoring the values there preserves instancing and makes the
    requested offsets effective in every environment.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    root_stage = Usd.Stage.Open(usd_path)
    if root_stage is None:
        raise RuntimeError(f"Could not open converted USD: {usd_path}")
    physics_layers = {
        layer.realPath or layer.identifier
        for layer in root_stage.GetUsedLayers()
        if (layer.realPath or layer.identifier).endswith("_physics.usd")
    }
    collider_count = 0
    for layer_path in physics_layers:
        stage = Usd.Stage.Open(layer_path)
        if stage is None:
            raise RuntimeError(f"Could not open converted physics layer: {layer_path}")
        changed = False
        for prim in stage.TraverseAll():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collider_count += 1
            collision_api = UsdPhysics.CollisionAPI(prim)
            physx_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not physx_api:
                physx_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                changed = True
            values = (
                (collision_api.CreateCollisionEnabledAttr(), True),
                (physx_api.CreateContactOffsetAttr(), 0.001),
                (physx_api.CreateRestOffsetAttr(), 0.0),
            )
            for attribute, value in values:
                current = attribute.Get()
                equal = current == value
                if isinstance(value, float) and current is not None:
                    equal = abs(float(current) - value) <= 1.0e-8
                if not equal:
                    attribute.Set(value)
                    changed = True
        if changed:
            layer = stage.GetRootLayer()
            layer_path = Path(layer.realPath or layer.identifier)
            temporary = layer_path.with_name(f".{layer_path.name}.{os.getpid()}.tmp")
            if not layer.Export(str(temporary)):
                raise RuntimeError(f"Could not export collision layer: {layer_path}")
            os.replace(temporary, layer_path)
    if collider_count == 0:
        raise RuntimeError(f"No collision schemas found in converted USD physics layers: {usd_path}")


def _convert_urdf_with_collision_properties(cfg: sim_utils.UrdfConverterCfg) -> str:
    """Serialize conversion/cache edits across Dex4D processes."""
    output = Path(cfg.usd_dir) / cfg.usd_file_name
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(f"{output.suffix}.dex4d.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        usd_path = sim_utils.UrdfConverter(cfg).usd_path
        _author_collision_properties(usd_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return usd_path


def _apply_self_collision_filters(
    usd_path: str, filter_pairs: tuple[tuple[str, str], ...]
) -> None:
    """Author the validated per-link self-collision exclusions into the cached USD."""
    if not filter_pairs:
        return
    from pxr import Usd, UsdPhysics

    root = Path(usd_path)
    physics = root.parent / "configuration" / f"{root.stem}_physics.usd"
    edit_path = physics if physics.exists() else root
    stage = Usd.Stage.Open(str(edit_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open robot USD for self-collision filters: {edit_path}")
    stage.Load()
    body_by_name = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    missing: set[str] = set()
    for left, right in filter_pairs:
        first, second = body_by_name.get(left), body_by_name.get(right)
        if first is None or second is None:
            missing.update(name for name, body in ((left, first), (right, second)) if body is None)
            continue
        relation = UsdPhysics.FilteredPairsAPI.Apply(first).CreateFilteredPairsRel()
        if second.GetPath() not in set(relation.GetTargets()):
            relation.AddTarget(second.GetPath())
    if missing:
        raise RuntimeError(f"Self-collision filter bodies missing from {edit_path}: {sorted(missing)}")
    stage.GetRootLayer().Save()


def _object_usd_paths(env: Any) -> tuple[str, ...]:
    cache_root = Path(env._sanitized_robot_urdf).parent.parent / "usd/objects"
    usd_paths: list[str] = []
    prepared: set[str] = set()
    for spec in _spawn_object_specs(env):
        converter_cfg = sim_utils.UrdfConverterCfg(
            asset_path=str(spec.urdf_path),
            usd_dir=str(cache_root / spec.code / spec.scale_str),
            usd_file_name=f"coacd_{spec.scale_str}.usd",
            force_usd_conversion=False,
            make_instanceable=True,
            fix_base=False,
            link_density=500.0,
            merge_fixed_joints=True,
            joint_drive=None,
            collider_type="convex_decomposition",
        )
        expected_path = str(Path(converter_cfg.usd_dir) / converter_cfg.usd_file_name)
        if expected_path not in prepared:
            usd_path = _convert_urdf_with_collision_properties(converter_cfg)
            prepared.add(expected_path)
        else:
            usd_path = expected_path
        usd_paths.append(usd_path)
    return tuple(usd_paths)


def _object_assets(usd_paths: tuple[str, ...], *, visible: bool) -> list[sim_utils.UsdFileCfg]:
    return [
        sim_utils.UsdFileCfg(
            usd_path=usd_path,
            visible=visible,
        )
        for usd_path in usd_paths
    ]


def setup_scene(env: Any) -> None:
    """Spawn and register the legacy-compatible Dex4D scene on ``env``."""
    if env.cfg.scene.replicate_physics:
        raise ValueError("Dex4D heterogeneous objects require scene.replicate_physics=False")
    profile = env._robot_profile
    robot_urdf = Path(env._sanitized_robot_urdf).expanduser().resolve()
    _validate_robot_urdf(robot_urdf, profile)

    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        friction_combine_mode="average",
        restitution_combine_mode="average",
    )
    material.func(_MATERIAL_PATH, material)

    robot_cache = robot_urdf.parent.parent / "usd/robot"
    object_usd_paths = _object_usd_paths(env)
    joint_defaults = dict(zip(profile.joint_names, env.cfg.robot_default_dof_pos, strict=True))
    robot_usd_path = _convert_urdf_with_collision_properties(
        sim_utils.UrdfConverterCfg(
            asset_path=str(robot_urdf),
            usd_dir=str(robot_cache),
            usd_file_name=f"{robot_urdf.stem}.usd",
            force_usd_conversion=False,
            make_instanceable=True,
            fix_base=True,
            link_density=1000.0,
            merge_fixed_joints=profile.merge_fixed_joints,
            self_collision=profile.import_self_collision,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            collider_type="convex_hull",
        )
    )
    _apply_self_collision_filters(robot_usd_path, profile.self_collision_filter_pairs)

    def actuator(
        joint_expr: str,
        stiffness,
        damping,
        *,
        armature=0.0,
        effort=None,
        velocity=None,
    ) -> ImplicitActuatorCfg:
        kwargs = {
            "joint_names_expr": [joint_expr],
            "stiffness": stiffness,
            "damping": damping,
            "armature": armature,
        }
        if effort is not None:
            kwargs["effort_limit_sim"] = effort
        if velocity is not None:
            kwargs["velocity_limit_sim"] = velocity
        return ImplicitActuatorCfg(**kwargs)

    robot_cfg = ArticulationCfg(
        prim_path=f"{env.scene.env_regex_ns}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path,
            activate_contact_sensors=True,
            rigid_props=_rigid_props(disable_gravity=True, damping=0.01),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=profile.enabled_self_collisions,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=profile.root_pos,
            rot=profile.root_rot,
            joint_pos=joint_defaults,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": actuator(
                profile.arm_joint_expr,
                profile.arm_stiffness,
                profile.arm_damping,
                effort=profile.arm_effort,
                velocity=profile.arm_velocity,
            ),
            "hand": actuator(
                profile.hand_joint_expr,
                profile.hand_stiffness,
                profile.hand_damping,
                armature=profile.hand_armature,
                effort=profile.hand_effort,
                velocity=profile.hand_velocity,
            ),
        },
    )
    common_object_cfg = {
        "random_choice": False,
        "mass_props": sim_utils.MassPropertiesCfg(density=500.0),
    }
    object_cfg = RigidObjectCfg(
        prim_path=f"{env.scene.env_regex_ns}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=_object_assets(object_usd_paths, visible=True),
            rigid_props=_rigid_props(disable_gravity=False),
            **common_object_cfg,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.7), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    table_cfg = RigidObjectCfg(
        prim_path=f"{env.scene.env_regex_ns}/Table",
        spawn=sim_utils.CuboidCfg(
            size=profile.table_size,
            rigid_props=_rigid_props(disable_gravity=True, kinematic=True),
            collision_props=_collision_props(),
            physics_material=material,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=profile.table_pos, rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    env.robot = Articulation(robot_cfg)
    env.object = RigidObject(object_cfg)
    env.table = RigidObject(table_cfg)
    ground = sim_utils.GroundPlaneCfg(physics_material=material)
    ground.func("/World/ground", ground)
    sim_utils.bind_physics_material("/World/envs", _MATERIAL_PATH, stronger_than_descendants=True)
    env.scene.filter_collisions(global_prim_paths=["/World/ground"])
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["table"] = env.table
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)


def resolve_robot_contract(env: Any) -> RobotContract:
    """Resolve and validate the post-import robot indices used by legacy Dex4D tensors."""
    profile = env._robot_profile
    robot = env.robot
    if len(robot.joint_names) != profile.num_joints:
        raise RuntimeError(
            f"Expected {profile.num_joints} robot DOFs for {profile.name}, "
            f"got {len(robot.joint_names)}: {robot.joint_names}"
        )
    joint_by_name = {name: index for index, name in enumerate(robot.joint_names)}
    body_by_name = {name: index for index, name in enumerate(robot.body_names)}
    missing_joints = [name for name in profile.joint_names if name not in joint_by_name]
    required_bodies = (profile.palm_body_name, *profile.fingertip_body_names)
    missing_bodies = [name for name in required_bodies if name not in body_by_name]
    if missing_joints or missing_bodies:
        raise RuntimeError(f"Dex4D robot contract mismatch: missing joints={missing_joints}, bodies={missing_bodies}")
    return RobotContract(
        joint_ids=tuple(joint_by_name[name] for name in profile.joint_names),
        palm_body_id=body_by_name[profile.palm_body_name],
        fingertip_body_ids=tuple(body_by_name[name] for name in profile.fingertip_body_names),
    )


__all__ = [
    "FINGERTIP_BODY_NAMES",
    "LEGACY_JOINT_NAMES",
    "ROBOT_PROFILES",
    "RobotContract",
    "RobotProfile",
    "get_robot_profile",
    "resolve_robot_contract",
    "setup_scene",
]
