import os

import numpy as np
import argparse
from scipy.interpolate import interp1d

from scripts import utils

from box_push_setup import *

from pydrake.all import Quaternion

np.set_printoptions(precision=8, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("traj_file_path", type=str)
parser.add_argument("--hz", type=int, default=50)
parser.add_argument("--slowdown", type=int, default=1)
args = parser.parse_args()


data = np.load(args.traj_file_path)

q_trj               = data["q_trj"]
u_trj               = data["u_trj"]
indices_q_u_into_q  = data["q_u_indices_into_x"]
indices_q_a_into_q  = data["q_a_indices_into_x"]
h                   = data["h"]

# The planner appends a "hold" action so q_trj and u_trj have matching
# lengths. The trailing slot ends up with joint_vel = 0 and obj_vel = 0
# (set by the zero-init below; the central-difference loop only fills
# interior indices), giving the RL policy an explicit hold target.
assert len(q_trj) == len(u_trj), (
    f"q_trj and u_trj length mismatch: {len(q_trj)} vs {len(u_trj)}. "
    "Re-run the planner so it appends the hold action."
)

planner_hz = 1 / h
planner_hz_int = int(round(planner_hz))

assert np.isclose(planner_hz_int, planner_hz)
assert args.hz % planner_hz_int == 0

dt = 1.0 / args.hz


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


upsampling_factor = int(round(args.hz / planner_hz_int)) * args.slowdown

u_trj = upsample_linear(u_trj, upsampling_factor)

obj_poses = q_trj[:, indices_q_u_into_q]  # [qw, qx, qy, qz, x, y, z]

obj_quats = upsample_slerp(obj_poses[:, :4], upsampling_factor)
obj_pos = upsample_linear(obj_poses[:, 4:], upsampling_factor)
obj_poses = np.hstack((obj_quats, obj_pos))

q_a_trj = upsample_linear(q_trj[:, indices_q_a_into_q], upsampling_factor)

joints = q_a_trj
joints_target = u_trj

N = len(q_a_trj)
EE_poses = np.zeros((N, 7))
EE_poses_target = np.zeros((N, 7))

for i in range(N):
    EE_poses[i] = utils.get_ee_pose(joints[i], arm_pose, "ur5e", eef_offset=eef_offset)
    EE_poses_target[i] = utils.get_ee_pose(joints_target[i], arm_pose, "ur5e", eef_offset=eef_offset)


joint_vel = np.zeros_like(joints)
joint_target_vel = np.zeros_like(joints_target)

obj_vel = np.zeros((N, 6))

def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

for i in range(1, N-1):
    joint_vel[i] = (joints[i+1] - joints[i-1]) / (2 * dt)
    joint_target_vel[i] = (joints_target[i+1] - joints_target[i-1]) / (2 * dt)

    obj_vel[i, :3] = (obj_poses[i+1][4:] - obj_poses[i-1][4:]) / (2 * dt)
    obj_vel[i, 3:] = angular_velocities(obj_poses[i-1][:4], obj_poses[i+1][:4], 2*dt)


filename = os.path.basename(args.traj_file_path)
date_str = filename[-19:-4]

output_dir = "../BoxLift/reference_trajectories/box_push_ur5e"
os.makedirs(output_dir, exist_ok=True)

# All poses converted to [x, y, z, qw, qx, qy, qz]
np.savez_compressed(
    os.path.join(output_dir, f"IK_{date_str}.npz"),
    obj_poses           = utils.convert_pose_to_isaaclab(obj_poses),
    obj_vel             = obj_vel,
    EE_poses            = utils.convert_pose_to_isaaclab(EE_poses),
    EE_poses_target     = utils.convert_pose_to_isaaclab(EE_poses_target),
    joints              = joints,
    joints_target       = joints_target,
    joint_vel           = joint_vel,
    joint_target_vel    = joint_target_vel,
    arm_pose            = utils.convert_pose_to_isaaclab(arm_pose),
    dt                  = dt,
    object_dims         = object_dims,
    object_mass         = box_mass,
)

print(f"Saved IsaacLab trajectory to {output_dir}/IK_{date_str}.npz")
