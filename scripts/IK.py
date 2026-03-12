import os

import numpy as np
import argparse
from scipy.interpolate import interp1d

import utils

from examples.box_lift.box_lift_setup import *

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

EE_trj = upsample_linear(q_trj[:, indices_q_a_into_q], upsampling_factor)

N = len(EE_trj)

EE_l = EE_trj[:, idx_q_a_l]
EE_r = EE_trj[:, idx_q_a_r]
EE_target_l = u_trj[:, idx_q_a_l]
EE_target_r = u_trj[:, idx_q_a_r]

EE_l_dirs = (obj_pos - EE_l)
EE_l_dirs /= np.linalg.norm(EE_l_dirs, axis=1, keepdims=True)
EE_r_dirs = (obj_pos - EE_r)
EE_r_dirs /= np.linalg.norm(EE_r_dirs, axis=1, keepdims=True)

# [x, y, z, qw, qx, qy, qz]
wrist_poses_l = utils.get_wrist_poses(EE_l, EE_l_dirs)
wrist_poses_target_l = utils.get_wrist_poses(EE_target_l, EE_l_dirs)
wrist_poses_r = utils.get_wrist_poses(EE_r, EE_r_dirs)
wrist_poses_target_r = utils.get_wrist_poses(EE_target_r, EE_r_dirs)

init_joints_l = np.array([
    0.0,
    -0.66 * np.pi,
    -0.5 * np.pi,
    - np.pi,
    0.0,
    0.0
])
init_joints_r = np.array([
    0.0,
    -0.33 * np.pi,
    0.5 * np.pi,
    0.0,
    0.0,
    0.0
])

joints_l = np.zeros((N, 6))
joints_r = np.zeros((N, 6))
joints_target_l = np.zeros((N, 6))
joints_target_r = np.zeros((N, 6))

EE_poses_l = np.zeros((N, 7))
EE_poses_r = np.zeros((N, 7))
EE_poses_target_l = np.zeros((N, 7))
EE_poses_target_r = np.zeros((N, 7))

for i in range(N):
    last_joints_l = init_joints_l if i == 0 else joints_l[i-1]
    last_joints_r = init_joints_r if i == 0 else joints_r[i-1]

    last_joints_r = init_joints_r

    joints_l[i] = utils.get_joints(wrist_poses_l[i], arm_l_pose, last_joints_l)
    joints_r[i] = utils.get_joints(wrist_poses_r[i], arm_r_pose, last_joints_r)
    joints_target_l[i] = utils.get_joints(wrist_poses_target_l[i], arm_l_pose, joints_l[i])
    joints_target_r[i] = utils.get_joints(wrist_poses_target_r[i], arm_r_pose, joints_r[i])

    if np.isnan(joints_r[i][0]):
        print(i)

    EE_poses_l[i] = utils.get_ee_pose(joints_l[i], arm_l_pose)
    EE_poses_r[i] = utils.get_ee_pose(joints_r[i], arm_r_pose)
    EE_poses_target_l[i] = utils.get_ee_pose(joints_target_l[i], arm_l_pose)
    EE_poses_target_r[i] = utils.get_ee_pose(joints_target_r[i], arm_r_pose)


filename = os.path.basename(args.traj_file_path)
date_str = filename[-19:-4]

# Make sure all poses are [x, y, z, qw, qx, qy, qz]
np.savez_compressed(
    f"IK_data/box_lift/IK_{date_str}.npz",
    obj_poses           = utils.convert_pose_to_isaaclab(obj_poses),
    EE_poses_l          = utils.convert_pose_to_isaaclab(EE_poses_l),
    EE_poses_r          = utils.convert_pose_to_isaaclab(EE_poses_r),
    EE_poses_target_l   = utils.convert_pose_to_isaaclab(EE_poses_target_l),
    EE_poses_target_r   = utils.convert_pose_to_isaaclab(EE_poses_target_r),
    joints_l            = joints_l,
    joints_target_l     = joints_target_l,
    joints_r            = joints_r,
    joints_target_r     = joints_target_r,
    arm_l_pose          = utils.convert_pose_to_isaaclab(arm_l_pose),
    arm_r_pose          = utils.convert_pose_to_isaaclab(arm_r_pose),
    dt                  = 1.0 / args.hz
)