import os

import numpy as np
import argparse
from scipy.interpolate import interp1d

from scripts import utils

from box_lift_setup import *

from pydrake.all import Quaternion

np.set_printoptions(precision=8, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("traj_file_path", type=str)
parser.add_argument("--hz", type=int, default=50)
args = parser.parse_args()


data = np.load(args.traj_file_path)

q_trj               = data["q_trj"]
u_trj               = data["u_trj"]
indices_q_u_into_q  = data["q_u_indices_into_x"]
indices_q_a_into_q  = data["q_a_indices_into_x"]
h                   = data["h"]


planner_hz = 1 / h
planner_hz_int = int(round(planner_hz))

# Only support perfect upsampling
assert np.isclose(planner_hz_int, planner_hz)
assert args.hz % planner_hz_int == 0

dt = 1.0 / args.hz

# linearly upsample q_trj, u_trj

def upsample_linear(x, k):
    N = x.shape[0]

    t_old = np.arange(N)
    t_new = np.linspace(0, N - 1, k * (N - 1) + 1)

    f = interp1d(t_old, x, axis=0, kind="linear")
    return f(t_new)

def upsample_slerp(wxyzs, k):
    out = []
    quats = [Quaternion(wxyz) for wxyz in wxyzs]
    
    for i in range(len(quats) - 1):
        for dt in np.linspace(0.0, 1.0, k, endpoint=False):
            quat_interp = quats[i].slerp(dt, quats[i + 1])
            out.append(quat_interp.wxyz())

    out.append(wxyzs[-1])

    return np.array(out)



upsampling_factor = int(round(args.hz / planner_hz_int))

u_trj = upsample_linear(u_trj, upsampling_factor)

obj_poses = q_trj[:, indices_q_u_into_q] # [qw, qx, qy, qz, x, y, z]

# We cannot linearly upsample quaternions so upsample seperately
obj_quats = upsample_slerp(obj_poses[:, :4], upsampling_factor)
obj_pos = upsample_linear(obj_poses[:, 4:], upsampling_factor)
obj_poses = np.hstack((obj_quats, obj_pos)) # [qw, qx, qy, qz, x, y, z]

q_a_trj = upsample_linear(q_trj[:, indices_q_a_into_q], upsampling_factor)

joints_l = q_a_trj[:, idx_q_a_l]
joints_r = q_a_trj[:, idx_q_a_r]
joints_target_l = u_trj[:, idx_q_a_l]
joints_target_r = u_trj[:, idx_q_a_r]

N = len(q_a_trj)
EE_poses_l = np.zeros((N, 7))
EE_poses_r = np.zeros((N, 7))
EE_poses_target_l = np.zeros((N, 7))
EE_poses_target_r = np.zeros((N, 7))

for i in range(N):
    EE_poses_l[i] = utils.get_ee_pose(joints_l[i], arm_l_pose)
    EE_poses_r[i] = utils.get_ee_pose(joints_r[i], arm_r_pose)
    EE_poses_target_l[i] = utils.get_ee_pose(joints_target_l[i], arm_l_pose)
    EE_poses_target_r[i] = utils.get_ee_pose(joints_target_r[i], arm_r_pose)


joint_vel_l = np.zeros_like(joints_l)
joint_vel_r = np.zeros_like(joints_r)
joint_target_vel_l = np.zeros_like(joints_target_l)
joint_target_vel_r = np.zeros_like(joints_target_r)

obj_vel = np.zeros((N, 6))

def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

for i in range(1, N-1):
    joint_vel_l[i] = (joints_l[i+1] - joints_l[i-1]) / (2 * dt)
    joint_vel_r[i] = (joints_r[i+1] - joints_r[i-1]) / (2 * dt)
    joint_target_vel_l[i] = (joints_l[i+1] - joints_l[i-1]) / (2 * dt)
    joint_target_vel_r[i] = (joints_r[i+1] - joints_r[i-1]) / (2 * dt)

    obj_vel[i, :3] = (obj_poses[i+1][4:] - obj_poses[i-1][4:]) / (2 * dt)
    obj_vel[i, 3:] = angular_velocities(obj_poses[i-1][:4], obj_poses[i+1][:4], 2*dt)


filename = os.path.basename(args.traj_file_path)
date_str = filename[-19:-4]


# Make sure all poses are [x, y, z, qw, qx, qy, qz]
np.savez_compressed(
    f"IK_data/box_lift_ur5/IK_{date_str}.npz",
    obj_poses           = utils.convert_pose_to_isaaclab(obj_poses),
    obj_vel             = obj_vel,
    EE_poses_l          = utils.convert_pose_to_isaaclab(EE_poses_l),
    EE_poses_r          = utils.convert_pose_to_isaaclab(EE_poses_r),
    EE_poses_target_l   = utils.convert_pose_to_isaaclab(EE_poses_target_l),
    EE_poses_target_r   = utils.convert_pose_to_isaaclab(EE_poses_target_r),
    joints_l            = joints_l,
    joints_target_l     = joints_target_l,
    joints_r            = joints_r,
    joints_target_r     = joints_target_r,
    joint_vel_l         = joint_vel_l,
    joint_target_vel_l  = joint_target_vel_l,
    joint_vel_r         = joint_vel_r,
    joint_target_vel_r  = joint_target_vel_r,
    arm_l_pose          = utils.convert_pose_to_isaaclab(arm_l_pose),
    arm_r_pose          = utils.convert_pose_to_isaaclab(arm_r_pose),
    dt                  = dt
)

