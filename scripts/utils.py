import numpy as np

from ur_ikfast import ur_kinematics
from pydrake.all import RigidTransform, Quaternion, RotationMatrix

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

def get_ee_pose(joints, robot_pose, robot="ur5"):
    ur5_arm = ur_kinematics.URKinematics(robot)

    # [x, y, z, qx, qy, qz, qw]
    wrist_pose_local = ur5_arm.forward(joints)
    wrist_quaternion = Quaternion(x=wrist_pose_local[3], y=wrist_pose_local[4], z=wrist_pose_local[5], w=wrist_pose_local[6])

    # shift ee position by 0.02 in z direction of wrist frame
    ee_pos_local = wrist_pose_local[:3] + wrist_quaternion.rotation() @ np.array([0.0, 0.0, 0.02])
    # Use same orientation as wrist
    ee_pose_local_quat = wrist_quaternion
    
    pose_to_robot = RigidTransform(quaternion=ee_pose_local_quat, p=ee_pos_local)

    robot_pose_quat = Quaternion(robot_pose[:4])
    robot_to_world = RigidTransform(quaternion=robot_pose_quat, p=robot_pose[4:])

    RT_ee_pose_abs : RigidTransform = robot_to_world @ pose_to_robot

    ee_pos_abs = RT_ee_pose_abs.translation()
    ee_quat_abs = RT_ee_pose_abs.rotation().ToQuaternion().wxyz()

    return np.concatenate([ee_quat_abs, ee_pos_abs])

def get_joints(pose, robot_pose, last = np.zeros(6), get_all_solutions=False, robot='ur5'):
    ur5_arm = ur_kinematics.URKinematics(robot)

    pose_quat = Quaternion(pose[:4])
    pose_to_world = RigidTransform(quaternion=pose_quat, p=pose[4:])

    robot_pose_quat = Quaternion(robot_pose[:4])
    
    world_to_robot = RigidTransform(quaternion=robot_pose_quat, p=robot_pose[4:]).inverse()

    RT_pose_rel : RigidTransform = world_to_robot @ pose_to_world

    pose_pos_rel = RT_pose_rel.translation()
    pose_quat_rel = RT_pose_rel.rotation().ToQuaternion()

    # ur5_arm requires [x, y, z, qx, qy, qz, w]
    pose_rel = np.array([
        pose_pos_rel[0], pose_pos_rel[1], pose_pos_rel[2],
        pose_quat_rel.x(), pose_quat_rel.y(), pose_quat_rel.z(), pose_quat_rel.w()
    ])

    return ur5_arm.inverse(pose_rel, get_all_solutions, last)

# [qw, qx, qy, qz, x, y, z] -> [x, y, z, qw, qx, qy, qz]
# Supports batched poses
def convert_pose_to_isaaclab(poses):
    pos = poses[..., 4:]
    quat = poses[..., :4]

    return np.concatenate([pos, quat], axis=-1)
