"""
Shared IsaacLab export for single-arm and bimanual box tasks.

Loads a planner/refiner output, upsamples to the RL rate, computes per-joint
and object velocities, and saves an IsaacLab-format npz with all poses in
[x, y, z, qw, qx, qy, qz] convention.

Reads task metadata (arm names, base poses, eef_offset, eef_robot, object
dims, mass) directly from the trajectory npz. The trajectory must have been
saved by run_planner / refine_rrt / collision_free_rrt with metadata
embedded.

Output naming convention:
  1 arm  (box_push, box_rotate):
    EE_poses, EE_poses_target, joints, joints_target,
    joint_vel, joint_target_vel, arm_pose

  2 arms (box_lift):
    EE_poses_l / EE_poses_r, EE_poses_target_l / EE_poses_target_r,
    joints_l / joints_r, joints_target_l / joints_target_r,
    joint_vel_l / joint_vel_r, joint_target_vel_l / joint_target_vel_r,
    arm_l_pose / arm_r_pose

Usage:
    python -m scripts.prepare_for_isaaclab ptc_data/box_push_ur5e/traj_refined_*.npz
"""
import argparse
import os

import numpy as np

from scripts import utils


def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[0] * q2[1] - q1[1] * q2[0] - q1[2] * q2[3] + q1[3] * q2[2],
        q1[0] * q2[2] + q1[1] * q2[3] - q1[2] * q2[0] - q1[3] * q2[1],
        q1[0] * q2[3] - q1[1] * q2[2] + q1[2] * q2[1] - q1[3] * q2[0],
    ])


def main():
    """Standalone entry point. All metadata comes from the trajectory npz."""
    np.set_printoptions(precision=8, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file_path", type=str)
    parser.add_argument("--hz", type=int, default=50)
    parser.add_argument("--slowdown", type=int, default=1)
    parser.add_argument(
        "--interpolation", type=str, default="cubic",
        choices=["zoh", "foh", "cubic"],
        help="Upsampling method for non-quaternion columns. Quaternions "
             "are always interpolated with SLERP.",
    )
    args = parser.parse_args()

    data = np.load(args.traj_file_path)
    task_name = str(data["task_name"])
    arm_names = [str(n) for n in data["arm_names"]]
    arm_poses = np.asarray(data["arm_poses"])
    eef_offset = float(data["eef_offset"])

    q_trj              = data["q_trj"]
    u_trj              = data["u_trj"]
    indices_q_u_into_q = data["q_u_indices_into_x"]
    indices_q_a_into_q = data["q_a_indices_into_x"]
    h                  = float(data["h"])

    # Natural format: len(q) == len(u) + 1. Backward-compat for old files.
    if len(q_trj) == len(u_trj) + 1:
        u_trj = np.concatenate([u_trj, u_trj[-1:]], axis=0)
    elif len(q_trj) == len(u_trj):
        pass
    else:
        raise ValueError(
            f"Unexpected q/u length relation: {len(q_trj)} vs {len(u_trj)}"
        )

    planner_hz = 1 / h
    planner_hz_int = int(round(planner_hz))
    assert np.isclose(planner_hz_int, planner_hz)
    assert args.hz % planner_hz_int == 0

    dt = 1.0 / args.hz
    upsampling_factor = int(round(args.hz / planner_hz_int)) * args.slowdown

    # --- Upsample (unified via utils.upsample_trj; quats always SLERP) ---
    u_trj_up = utils.upsample_trj(
        u_trj, upsampling_factor, method=args.interpolation
    )

    obj_poses_coarse = q_trj[:, indices_q_u_into_q]  # [qw, qx, qy, qz, x, y, z]
    obj_poses = utils.upsample_trj(
        obj_poses_coarse,
        upsampling_factor,
        quat_col_indices=np.array([0, 1, 2, 3]),
        method=args.interpolation,
    )

    q_a_trj = utils.upsample_trj(
        q_trj[:, indices_q_a_into_q],
        upsampling_factor,
        method=args.interpolation,
    )

    N = len(q_a_trj)

    # Object velocities (central difference)
    obj_vel = np.zeros((N, 6))
    for i in range(1, N - 1):
        obj_vel[i, :3] = (obj_poses[i + 1][4:] - obj_poses[i - 1][4:]) / (2 * dt)
        obj_vel[i, 3:] = angular_velocities(
            obj_poses[i - 1][:4], obj_poses[i + 1][:4], 2 * dt
        )

    # --- Per-arm computations ---
    n_joints_in_traj = q_a_trj.shape[1]
    n_arms = len(arm_names)
    assert n_joints_in_traj % n_arms == 0, (
        f"Actuated joint count in traj ({n_joints_in_traj}) not "
        f"divisible by n_arms ({n_arms})"
    )
    per_arm = n_joints_in_traj // n_arms

    per_arm_data = []
    for i, (arm_name, arm_pose) in enumerate(zip(arm_names, arm_poses)):
        local_idx = np.arange(i * per_arm, (i + 1) * per_arm)
        joints = q_a_trj[:, local_idx]
        joints_target = u_trj_up[:, local_idx]

        ee_poses = np.zeros((N, 7))
        ee_poses_target = np.zeros((N, 7))
        for i in range(N):
            ee_poses[i] = utils.get_ee_pose(
                joints[i], arm_pose, eef_offset=eef_offset
            )
            ee_poses_target[i] = utils.get_ee_pose(
                joints_target[i], arm_pose, eef_offset=eef_offset
            )

        joint_vel = np.zeros_like(joints)
        joint_target_vel = np.zeros_like(joints_target)
        for i in range(1, N - 1):
            joint_vel[i] = (joints[i + 1] - joints[i - 1]) / (2 * dt)
            joint_target_vel[i] = (
                joints_target[i + 1] - joints_target[i - 1]
            ) / (2 * dt)

        per_arm_data.append({
            "name": arm_name,
            "arm_pose": arm_pose,
            "joints": joints,
            "joints_target": joints_target,
            "joint_vel": joint_vel,
            "joint_target_vel": joint_target_vel,
            "EE_poses": ee_poses,
            "EE_poses_target": ee_poses_target,
        })

    # --- Build save dict with naming convention by arm count ---
    save_dict = {
        "obj_poses":   utils.convert_pose_to_isaaclab(obj_poses),
        "obj_vel":     obj_vel,
        "dt":          dt,
        "object_dims": np.asarray(data["object_dims"]),
        "object_mass": float(data["object_mass"]),
    }

    if len(per_arm_data) == 1:
        arm = per_arm_data[0]
        save_dict.update({
            "EE_poses":         utils.convert_pose_to_isaaclab(arm["EE_poses"]),
            "EE_poses_target":  utils.convert_pose_to_isaaclab(arm["EE_poses_target"]),
            "joints":           arm["joints"],
            "joints_target":    arm["joints_target"],
            "joint_vel":        arm["joint_vel"],
            "joint_target_vel": arm["joint_target_vel"],
            "arm_pose":         utils.convert_pose_to_isaaclab(arm["arm_pose"]),
        })
    elif len(per_arm_data) == 2:
        # Bimanual: save with _l / _r suffix in the order of arm_poses.
        for suffix, arm in zip(("l", "r"), per_arm_data):
            save_dict.update({
                f"EE_poses_{suffix}":         utils.convert_pose_to_isaaclab(arm["EE_poses"]),
                f"EE_poses_target_{suffix}":  utils.convert_pose_to_isaaclab(arm["EE_poses_target"]),
                f"joints_{suffix}":           arm["joints"],
                f"joints_target_{suffix}":    arm["joints_target"],
                f"joint_vel_{suffix}":        arm["joint_vel"],
                f"joint_target_vel_{suffix}": arm["joint_target_vel"],
                f"arm_{suffix}_pose":         utils.convert_pose_to_isaaclab(arm["arm_pose"]),
            })
    else:
        raise NotImplementedError(
            f"Only 1 or 2 arms supported, got {len(per_arm_data)}"
        )

    # --- Save ---
    output_dir = f"../BoxLift/reference_trajectories/{task_name}"
    os.makedirs(output_dir, exist_ok=True)

    input_basename = os.path.splitext(os.path.basename(args.traj_file_path))[0]
    output_filename = f"{input_basename}_{args.interpolation}.npz"
    output_path = os.path.join(output_dir, output_filename)

    np.savez_compressed(output_path, **save_dict)
    print(f"Saved IsaacLab trajectory to {output_path}")


if __name__ == "__main__":
    main()
