"""
Shared trajectory evaluation (smoothness + accuracy) for any task.

The rollout and smoothness metrics are completely task-agnostic. The only
task-specific inputs are:
  - q_sim (the quasistatic simulator, to roll out u_trj)
  - idx_q_u / idx_q_a (state indices)
  - pose_sampling_function(1.0) to know what "goal" means

Pass these via the task setup module. Example usage:

    import box_push_setup as setup
    from scripts.evaluate_trajectory import evaluate_file
    metrics = evaluate_file(setup, "ptc_data/.../traj_refined_*.npz")

Or from a per-example CLI wrapper that also supports --json mode.
"""
import argparse
import copy
from scripts.task_setup import deduce_setup
import json

import numpy as np


def rollout_trajectory(u_trj, q0, q_sim, h):
    """Forward-simulate u_trj from q0 through the quasistatic sim.

    Returns q_trj of length T+1 (natural MPC format). NaN u rows
    (regrasps) are passed through as a hold.
    """
    sim_params = copy.deepcopy(q_sim.get_sim_params())
    sim_params.h = h

    T = len(u_trj)
    q_trj = np.zeros((T + 1, len(q0)))
    q_trj[0] = q0
    q_curr = q0.copy()
    for t in range(T):
        if np.any(np.isnan(u_trj[t])):
            q_trj[t + 1] = q_curr
            continue
        q_curr = q_sim.calc_dynamics(q_curr, u_trj[t], sim_params)
        q_trj[t + 1] = q_curr
    return q_trj


def windowed_du_variance(du, windows=[5, 10, 20]):
    """Per-scale variance of du over sliding windows.

    Returns (mean_var, max_var, per_scale_dict). Steady motion gives
    low variance; direction reversals within a window give high variance.
    """
    if len(du) < 3:
        return 0.0, 0.0, {}

    per_scale = {}
    all_vars = []

    for w in windows:
        if len(du) < w:
            continue
        n_windows = len(du) - w + 1
        window_vars = np.zeros((n_windows, du.shape[1]))
        for t in range(n_windows):
            window_vars[t] = np.var(du[t:t + w], axis=0)

        scale_mean = float(np.mean(window_vars))
        scale_max = float(np.max(window_vars))
        per_scale[w] = (scale_mean, scale_max)
        all_vars.append(window_vars)

    if not all_vars:
        return 0.0, 0.0, {}

    combined = np.concatenate(all_vars, axis=0)
    return float(np.mean(combined)), float(np.max(combined)), per_scale


def compute_metrics(u_trj, q_trj_rerolled, h, idx_q_u, idx_q_a, goal_pose):
    """Compute smoothness and accuracy metrics for a trajectory."""
    # Filter out NaN rows (regrasps)
    valid = ~np.any(np.isnan(u_trj), axis=1)
    u_valid = u_trj[valid]

    # --- Smoothness metrics ---
    du = np.diff(u_valid, axis=0)
    d2u = np.diff(du, axis=0)

    # Convert to physical units: velocity (rad/s) and acceleration (rad/s^2)
    velocity_local = du / h if h > 0 else du
    acceleration = d2u / (h * h) if h > 0 else d2u

    rms_velocity = float(np.sqrt(np.mean(velocity_local ** 2)))
    max_velocity = float(np.max(np.abs(velocity_local)))
    rms_accel = float(np.sqrt(np.mean(acceleration ** 2)))
    max_accel = float(np.max(np.abs(acceleration)))

    max_accel_per_joint = np.max(np.abs(acceleration), axis=0).tolist()

    # Windowed velocity variance at multiple time scales (sample-rate independent).
    velocity = du / h if h > 0 else du
    hz = int(round(1.0 / h)) if h > 0 else 50
    windows = [max(2, hz // 10), max(3, hz // 5), max(5, hz * 2 // 5)]
    _, _, vel_var_per_scale = windowed_du_variance(velocity, windows=windows)
    du_var_per_scale = {
        f"{w / hz:.2f}s": list(v) for w, v in vel_var_per_scale.items()
    }

    # --- Accuracy metrics ---
    # Final object pose vs desired goal
    final_pose = q_trj_rerolled[-1, idx_q_u]
    pos_error = float(np.linalg.norm(goal_pose[4:] - final_pose[4:]))
    quat_error = float(np.linalg.norm(goal_pose[:4] - final_pose[:4]))

    duration = len(u_trj) * h

    # Total path length of the object
    obj_positions = q_trj_rerolled[:, idx_q_u[4:]]
    obj_path_length = float(
        np.sum(np.linalg.norm(np.diff(obj_positions, axis=0), axis=1))
    )

    return {
        "rms_velocity": rms_velocity,
        "max_velocity": max_velocity,
        "rms_accel": rms_accel,
        "max_accel": max_accel,
        "max_accel_per_joint": max_accel_per_joint,
        "du_var_per_scale": {
            str(k): list(v) for k, v in du_var_per_scale.items()
        },
        "goal_pos_error": pos_error,
        "goal_quat_error": quat_error,
        "duration": duration,
        "n_steps": len(u_trj),
        "obj_path_length": obj_path_length,
        "is_unchanged": False,
    }


def compute_score(metrics):
    """Single scalar score (lower is better).

    Combines physical smoothness (rms/max acceleration, sub-scale variance)
    and goal-tracking accuracy. Tune weights in your task's wrapper if
    you want different emphasis.
    """
    du_var = metrics["du_var_per_scale"]
    smoothness_cost = (
        metrics["max_accel"] * 0.5
        + metrics["rms_accel"] * 2.0
        + sum(v[0] for v in du_var.values()) * 0.05
    )

    pos_err = metrics["goal_pos_error"]
    accuracy_cost = pos_err * 100.0
    if pos_err > 0.05:
        accuracy_cost += (pos_err - 0.05) * 500.0

    return smoothness_cost + accuracy_cost


def evaluate_file(traj_path: str):
    """Load a trajectory file and compute all metrics against the setup."""
    data = np.load(traj_path)

    setup = deduce_setup(data["task_name"])

    u_trj = data["u_trj"]
    q_trj_stored = data["q_trj"]
    h = float(data["h"])

    q_trj_rerolled = rollout_trajectory(
        u_trj, q_trj_stored[0], setup.q_sim, h
    )

    goal_pose = setup.pose_sampling_function(1.0)
    metrics = compute_metrics(
        u_trj, q_trj_rerolled, h, setup.idx_q_u, setup.idx_q_a, goal_pose
    )
    metrics["score"] = compute_score(metrics)
    return metrics


def print_metrics(metrics):
    print(f"\n{'Metric':<25} {'Value':>12}")
    print("-" * 40)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k:<25} {v}")
        elif isinstance(v, list):
            print(f"{k:<25} {[f'{x:.5f}' for x in v]}")
        elif isinstance(v, bool):
            print(f"{k:<25} {v}")
        elif isinstance(v, int):
            print(f"{k:<25} {v:>12d}")
        else:
            print(f"{k:<25} {v:12.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file_path", type=str)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    metrics = evaluate_file(args.traj_file_path)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_metrics(metrics)


if __name__ == "__main__":
    main()
