import os

import numpy as np
from scipy.interpolate import CubicSpline

from ur_ikfast import ur_kinematics
from pydrake.all import (
    RigidTransform,
    Quaternion,
    RotationMatrix,
    RollPitchYaw,
    PiecewisePolynomial,
    PiecewiseQuaternionSlerp,
)

def cosine_weighted_cone(theta_max):
    """
    Generate a cosine-weighted direction
    inside cone around +Z.
    """
    u1 = np.random.rand()
    u2 = np.random.rand()

    sin_theta = np.sqrt(u1) * np.sin(theta_max)
    cos_theta = np.sqrt(1.0 - sin_theta**2)

    phi = 2.0 * np.pi * u2

    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta

    return np.array([x,y,z])

def dir_to_quat(direction):
    z_axis = np.array([0,0,1])

    dot = np.dot(z_axis, direction)
    # Aligned
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    # Aligned
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])

    axis = np.cross(z_axis, direction)
    axis /= np.linalg.norm(axis)

    angle = np.arccos(dot)

    half = 0.5 * angle

    w = np.cos(half)
    xyz = axis * np.sin(half)

    return np.array([w, xyz[0], xyz[1], xyz[2]])

def quat_apply(q: np.ndarray, vec: np.ndarray):
    xyz = q[1:]
    t = np.cross(xyz, vec) * 2
    return (vec + q[:1] * t + np.cross(xyz, t))


def get_wrist_pose(ee_pos: np.ndarray, ee_direction: np.ndarray):
    wrist_quat : Quaternion = RotationMatrix.MakeFromOneVector(ee_direction, axis_index=2).ToQuaternion()

    EE_radius = 0.02
    wrist_pos = ee_pos - ee_direction * EE_radius

    return np.concatenate([
        wrist_quat.wxyz(),
        wrist_pos,
    ])

def get_wrist_poses(positions: np.ndarray, directions: np.ndarray):
    poses = np.vstack([
        # get_initial_pose(positions, directions),
        np.array([get_wrist_pose(pos, direction) for pos, direction in zip(positions, directions)])
    ])
    return poses

def get_ee_pose(joints, robot_pose, eef_offset=0.02):
    ur5e_arm = ur_kinematics.URKinematics("ur5e")

    # [x, y, z, qx, qy, qz, qw]
    wrist_pose_local = ur5e_arm.forward(joints)
    wrist_quaternion = Quaternion(x=wrist_pose_local[3], y=wrist_pose_local[4], z=wrist_pose_local[5], w=wrist_pose_local[6])

    # shift ee position along wrist z by eef_offset (0 if no sphere EEF)
    ee_pos_local = wrist_pose_local[:3] + wrist_quaternion.rotation() @ np.array([0.0, 0.0, eef_offset])
    ee_pose_local_quat = wrist_quaternion

    pose_to_robot = RigidTransform(quaternion=ee_pose_local_quat, p=ee_pos_local)

    robot_pose_quat = Quaternion(robot_pose[:4])
    robot_to_world = RigidTransform(quaternion=robot_pose_quat, p=robot_pose[4:])

    RT_ee_pose_abs : RigidTransform = robot_to_world @ pose_to_robot

    ee_pos_abs = RT_ee_pose_abs.translation()
    ee_quat_abs = RT_ee_pose_abs.rotation().ToQuaternion().wxyz()

    return np.concatenate([ee_quat_abs, ee_pos_abs])

def ee_pose_to_wrist_pose(pose, eef_offset=0.02):
    pos = pose[4:]
    quat = Quaternion(pose[:4])
    dpos = quat.rotation() @ np.array([0.0, 0.0, -eef_offset])

    pose_wrist = np.concatenate([quat.wxyz(), pos + dpos])

    return pose_wrist

def get_joints(pose, robot_pose, last=np.zeros(6), get_all_solutions=False, eef_offset=0.02):
    ur5e_arm = ur_kinematics.URKinematics("ur5e")

    wrist_pose = ee_pose_to_wrist_pose(pose, eef_offset=eef_offset)

    wrist_quat = Quaternion(wrist_pose[:4])
    wrist_pose_to_world = RigidTransform(quaternion=wrist_quat, p=wrist_pose[4:])

    robot_pose_quat = Quaternion(robot_pose[:4])

    world_to_robot = RigidTransform(quaternion=robot_pose_quat, p=robot_pose[4:]).inverse()

    RT_pose_rel : RigidTransform = world_to_robot @ wrist_pose_to_world

    pose_pos_rel = RT_pose_rel.translation()
    pose_quat_rel = RT_pose_rel.rotation().ToQuaternion()

    # ur5e_arm requires [x, y, z, qx, qy, qz, w]
    pose_rel = np.array([
        pose_pos_rel[0], pose_pos_rel[1], pose_pos_rel[2],
        pose_quat_rel.x(), pose_quat_rel.y(), pose_quat_rel.z(), pose_quat_rel.w()
    ])

    return ur5e_arm.inverse(pose_rel, get_all_solutions, last)

# [qw, qx, qy, qz, x, y, z] -> [x, y, z, qw, qx, qy, qz]
# Supports batched poses
def convert_pose_to_isaaclab(poses):
    pos = poses[..., 4:]
    quat = poses[..., :4]

    return np.concatenate([pos, quat], axis=-1)


def generate_box_sdf(
    dims: np.ndarray,
    mass: float = 1.0,
    friction: float = 0.8,
    corner_friction: float = 0.1,
    model_name: str = "box",
    bounce: bool = True,
) -> str:
    """Generate an SDF string for a box with corner collision spheres."""
    x, y, z = dims
    hx, hy, hz = dims / 2

    # Inertia for uniform density box
    ixx = mass / 12 * (y**2 + z**2)
    iyy = mass / 12 * (x**2 + z**2)
    izz = mass / 12 * (x**2 + y**2)

    # Collision geometry slightly smaller to avoid degenerate contacts
    cx, cy, cz = x - 0.001, y - 0.001, z - 0.001

    # 8 corner positions
    corners = []
    for sx, name_x in [(-1, "minus_x"), (1, "plus_x")]:
        for sy, name_y in [(-1, "lower"), (1, "upper")]:
            for sz, name_z in [(-1, "left"), (1, "right")]:
                corners.append((
                    f"{name_x}_{name_y}_{name_z}_corner",
                    sx * hx, sy * hy, sz * hz,
                ))

    bounce_xml = ""
    if bounce:
        bounce_xml = """
          <bounce>
            <restitution_coefficient>0.0</restitution_coefficient>
            <threshold>1e+06</threshold>
          </bounce>"""

    corner_xml = ""
    for name, px, py, pz in corners:
        corner_xml += f"""
      <collision name="{name}">
        <pose> {px} {py} {pz} 0 0 0</pose>
        <geometry>
          <sphere>
            <radius>1e-7</radius>
          </sphere>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>{corner_friction}</mu>
              <mu2>{corner_friction}</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
"""

    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{model_name}">
    <link name="box">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>{iyy}</iyy>
          <iyz>0</iyz>
          <izz>{izz}</izz>
        </inertia>
      </inertial>

      <visual name="box_visual">
        <geometry>
          <box>
            <size>{x} {y} {z}</size>
          </box>
        </geometry>
        <material>
          <diffuse>0.9 0.9 0.9 0.9</diffuse>
        </material>
      </visual>

      <collision name="collision">
        <geometry>
          <box>
            <size>{cx} {cy} {cz}</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>{friction}</mu>
              <mu2>{friction}</mu2>
            </ode>
          </friction>{bounce_xml}
        </surface>
      </collision>
{corner_xml}
    </link>
  </model>
</sdf>
"""


def _pose_to_yml_transform(pose: np.ndarray) -> str:
    """Convert [qw,qx,qy,qz,x,y,z] pose to YML X_PF block."""
    quat = Quaternion(pose[:4])
    rpy = RollPitchYaw(quat).vector()  # radians
    rpy_deg = np.degrees(rpy)
    trans = pose[4:]

    lines = f"        base_frame: world\n"
    lines += f"        translation: [{trans[0]}, {trans[1]}, {trans[2]}]\n"

    # Only add rotation if non-identity
    if not np.allclose(rpy_deg, 0, atol=0.01):
        # Avoid -0.0 in output
        r, p, y = [0.0 if abs(v) < 0.01 else v for v in rpy_deg]
        lines += f"        rotation: !Rpy {{deg: [{r:.1f}, {p:.1f}, {y:.1f}]}}\n"

    return lines


def _generate_ground_sdf(friction: float, model_name: str = "ground") -> str:
    """Generate a ground plane SDF with configurable friction."""
    return f"""<sdf version='1.6'>
  <model name='{model_name}'>

  <link name="ground">
  <pose> 0 0 0 0 0 0</pose>
  <visual name='visual'>
    <pose frame=''> 0 0 0 0 0 0 </pose>
    <geometry>
      <box>
       <size>10 10 1</size>
      </box>
    </geometry>
    <material>
      <diffuse>0.39 0.4 0.42 1.0</diffuse>
    </material>
  </visual>

  <collision name='collision'>
    <pose> 0 0 0 0 0 0 </pose>
    <geometry>
      <box>
       <size>10 10 1</size>
      </box>
    </geometry>

    <surface>
      <friction>
      <ode>
        <mu>{friction}</mu>
        <mu2>{friction}</mu2>
      </ode>
      </friction>
    </surface>
  </collision>
  </link>

  </model>
</sdf>
"""


def _generate_ground_yml(ground_sdf_name: str, z_offset: float = -0.5) -> str:
    """Generate a ground model directive YML."""
    return f"""directives:
  - add_model:
      name: ground
      file: package://quasistatic_simulator/{ground_sdf_name}

  - add_frame:
      name: ground_world_offset
      X_PF:
        base_frame: world
        translation: [ 0, 0, {z_offset}]

  - add_weld:
      parent: ground_world_offset
      child: ground::ground
"""


def generate_ur5e_box_models(
    models_dir: str,
    object_dims: np.ndarray,
    arms: dict,
    mass: float = 1.0,
    friction: float = 0.8,
    ground_friction: float = None,
    ground_offset: float = 0.0,
    prefix: str = "generated",
    robot_sdf: str = "ur5e.sdf",
    gradients: bool = False,
    joint_kp: list = None,
) -> str:
    """Generate SDF and YML model files for N UR5e arms + box task.

    Args:
        models_dir: Path to quasistatic_simulator/models/
        object_dims: Box [x, y, z] dimensions
        arms: Dict mapping robot name to [qw,qx,qy,qz,x,y,z] base pose.
              e.g. {"ur5e": pose} or {"ur5e_l": pose_l, "ur5e_r": pose_r}
        mass: Box mass in kg
        friction: Box surface friction coefficient
        ground_friction: Ground surface friction. If None, uses shared ground.yml.
        ground_offset: Vertical offset of ground surface in meters (e.g. 0.018
            for a surface 1.8cm above the robot base plane).
        prefix: Filename prefix for generated files
        robot_sdf: SDF file for the robot (relative to models_dir package)
        gradients: If True, also generate a gradients variant
                   (ground shifted down 2mm, no bounce on box collision)

    Returns:
        q_model_path: Absolute path to the generated q_sys YML file.
            If gradients=True, returns (q_model_path, q_model_path_gradients).
    """
    q_sys_dir = os.path.join(models_dir, "q_sys")

    # 1. Generate box SDF
    box_sdf = generate_box_sdf(object_dims, mass=mass, friction=friction)
    box_sdf_name = f"{prefix}_box.sdf"
    with open(os.path.join(models_dir, box_sdf_name), "w") as f:
        f.write(box_sdf)

    if gradients:
        box_sdf_grad = generate_box_sdf(
            object_dims, mass=mass, friction=friction, bounce=False,
        )
        box_sdf_grad_name = f"{prefix}_box_gradients.sdf"
        with open(os.path.join(models_dir, box_sdf_grad_name), "w") as f:
            f.write(box_sdf_grad)

    # 1b. Generate ground SDF + YML if custom friction or offset specified
    custom_ground = ground_friction is not None or ground_offset != 0.0
    if custom_ground:
        gf = ground_friction if ground_friction is not None else 0.5
        z_base = -0.5 + ground_offset

        ground_sdf_name = f"{prefix}_ground.sdf"
        with open(os.path.join(models_dir, ground_sdf_name), "w") as f:
            f.write(_generate_ground_sdf(gf))

        ground_yml_name = f"{prefix}_ground.yml"
        with open(os.path.join(models_dir, ground_yml_name), "w") as f:
            f.write(_generate_ground_yml(ground_sdf_name, z_offset=z_base))

        if gradients:
            ground_yml_grad_name = f"{prefix}_ground_gradients.yml"
            with open(os.path.join(models_dir, ground_yml_grad_name), "w") as f:
                f.write(_generate_ground_yml(ground_sdf_name, z_offset=z_base - 0.002))
    else:
        ground_yml_name = "ground.yml"
        ground_yml_grad_name = "ground_gradients.yml"

    # 2. Generate model directive YML(s)
    def _write_directive(ground_file, out_name):
        arm_blocks = ""
        for name, pose in arms.items():
            transform_block = _pose_to_yml_transform(pose)
            arm_blocks += f"""
- add_model:
    name: {name}
    file: package://quasistatic_simulator/{robot_sdf}

- add_frame:
    name: world_{name}_offset
    X_PF:
{transform_block}
- add_weld:
    parent: world_{name}_offset
    child: {name}::base_link
"""
        directive = f"directives:\n- add_directives:\n    file: package://quasistatic_simulator/{ground_file}\n{arm_blocks}"
        with open(os.path.join(models_dir, out_name), "w") as f:
            f.write(directive)

    directive_name = f"{prefix}_ur5e.yml"
    _write_directive(ground_yml_name, directive_name)

    if gradients:
        directive_grad_name = f"{prefix}_ur5e_gradients.yml"
        _write_directive(ground_yml_grad_name, directive_grad_name)

    # 3. Generate q_sys YML(s)
    if joint_kp is None:
        joint_kp = [400, 400, 400, 300, 200, 100]
    kp_str = "[" + ", ".join(str(k) for k in joint_kp) + "]"
    robot_entries = ""
    for name in arms:
        robot_entries += f"  -\n    name: {name}\n    Kp: {kp_str}\n"

    def _write_q_sys(directive_ref, box_sdf_ref, out_path):
        q_sys_yml = f"""model_directive: package://quasistatic_simulator/{directive_ref}
robots:
{robot_entries}
objects:
  -
    name: box
    file: package://quasistatic_simulator/{box_sdf_ref}

quasistatic_sim_params:
  gravity: [0, 0, -9.8]
  nd_per_contact: 4
  contact_detection_tolerance: 0.2
  is_quasi_dynamic: True
  unactuated_mass_scale: .NAN
"""
        with open(out_path, "w") as f:
            f.write(q_sys_yml)

    q_sys_path = os.path.join(q_sys_dir, f"{prefix}_ur5e_box.yml")
    _write_q_sys(directive_name, box_sdf_name, q_sys_path)

    if gradients:
        q_sys_grad_path = os.path.join(q_sys_dir, f"{prefix}_ur5e_box_gradients.yml")
        _write_q_sys(directive_grad_name, box_sdf_grad_name, q_sys_grad_path)
        return q_sys_path, q_sys_grad_path

    return q_sys_path


def upsample_trj(
    trj_coarse,
    n_steps_per_h,
    quat_col_indices=None,
    method="foh",
):
    """Time-upsample a state trajectory to a finer knot spacing.

    Given `trj_coarse` of shape (N, dim_q) with N = T0 + 1 coarse knots,
    returns a trajectory of shape (T0*n_steps_per_h + 1, dim_q).

    Non-quaternion columns are interpolated by `method`:
      "zoh"   - zero-order hold (staircase).
      "foh"   - first-order hold / piecewise linear.
      "cubic" - natural cubic spline (scipy.CubicSpline, bc_type='natural').

    Quaternion columns, if `quat_col_indices` (expected wxyz, length 4) is
    given, are always interpolated with Drake's `PiecewiseQuaternionSlerp`
    regardless of `method`.
    """
    trj_coarse = np.asarray(trj_coarse, dtype=float)
    N, dim_q = trj_coarse.shape
    T0 = N - 1
    T_plus_1 = T0 * n_steps_per_h + 1

    t_knots = np.arange(N, dtype=float)
    t_fine = np.arange(T_plus_1) / n_steps_per_h

    if quat_col_indices is not None and len(quat_col_indices) == 4:
        quat_cols = np.asarray(quat_col_indices, dtype=int)
        non_quat_cols = np.setdiff1d(np.arange(dim_q), quat_cols, assume_unique=False)
    else:
        quat_cols = None
        non_quat_cols = np.arange(dim_q)

    result = np.zeros((T_plus_1, dim_q))

    if len(non_quat_cols) > 0:
        data = trj_coarse[:, non_quat_cols]
        if method == "zoh":
            # Each fine sample takes the value at the last coarse knot on
            # or before it. At t_fine == N-1 the index lands exactly on the
            # final knot, so the final sample is q[N-1].
            idx = np.minimum(np.floor(t_fine).astype(int), N - 1)
            result[:, non_quat_cols] = data[idx]
        elif method == "foh":
            poly = PiecewisePolynomial.FirstOrderHold(t_knots, data.T)
            for k, t in enumerate(t_fine):
                result[k, non_quat_cols] = poly.value(float(t)).squeeze()
        elif method == "cubic":
            cs = CubicSpline(t_knots, data, axis=0, bc_type="natural")
            result[:, non_quat_cols] = cs(t_fine)
        else:
            raise ValueError(
                f"Unknown method {method!r}; expected one of "
                "'zoh', 'foh', 'cubic'."
            )

    if quat_cols is not None:
        quats = []
        for i in range(N):
            wxyz = trj_coarse[i, quat_cols]
            n = np.linalg.norm(wxyz)
            quats.append(Quaternion(wxyz / (n if n > 1e-10 else 1.0)))
        slerp = PiecewiseQuaternionSlerp(t_knots, quats)
        for k, t in enumerate(t_fine):
            result[k, quat_cols] = slerp.value(float(t)).squeeze()

    return result
