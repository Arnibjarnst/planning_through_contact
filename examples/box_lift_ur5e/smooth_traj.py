"""
Savitzky-Golay filter-based smoother for box_lift_ur5e trajectories.

Contact-agnostic sliding-window polynomial smoothing. The smoothed joint
targets may not reproduce the exact object motion from the RRT, but we
leave the reference q_trj unchanged and trust the downstream RL policy
to handle any tracking error.

References:
- Savitzky, A. & Golay, M.J.E. (1964). "Smoothing and Differentiation
  of Data by Simplified Least Squares Procedures." Analytical Chemistry,
  36(8), 1627-1639.
- Schafer, R.W. (2011). "What is a Savitzky-Golay Filter?" IEEE Signal
  Processing Magazine, 28(4), 111-117.

Regrasp segments are handled: each segment is filtered independently so
the NaN boundary rows are preserved.

Usage:
    python smooth_traj.py ptc_data/box_lift_ur5e/traj_<ts>.npz \\
        [--window 11] [--polyorder 3]
"""
import argparse
import os
import numpy as np
from scipy.signal import savgol_filter

from irs_rrt.irs_rrt import IrsRrt

from box_lift_setup import *


parser = argparse.ArgumentParser()
parser.add_argument("traj_file_path", type=str)
parser.add_argument("--window", type=int, default=11)
parser.add_argument("--polyorder", type=int, default=3)
args = parser.parse_args()


data = np.load(args.traj_file_path)
q_trj = data["q_trj"]
u_trj = data["u_trj"]
h = float(data["h"])

print(f"Loaded {len(q_trj)} knots at {1.0/h:.1f} Hz")

segments = IrsRrt.get_regrasp_segments(u_trj)
print(f"Found {len(segments)} contact segments")


def smooth_segment(u_seg, window, polyorder):
    """Savitzky-Golay filter a u segment along the time axis. Pins endpoints."""
    n = len(u_seg)
    if n <= polyorder + 1:
        return u_seg.copy()

    w = min(window, n if n % 2 == 1 else n - 1)
    if w < polyorder + 2:
        w = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
        w = min(w, n if n % 2 == 1 else n - 1)
    if w < polyorder + 2:
        return u_seg.copy()

    u_smooth = savgol_filter(
        u_seg, window_length=w, polyorder=polyorder, axis=0, mode="nearest",
    )
    u_smooth[0] = u_seg[0]
    u_smooth[-1] = u_seg[-1]
    return u_smooth


u_smoothed = u_trj.copy()
for t_start, t_end in segments:
    u_smoothed[t_start:t_end] = smooth_segment(
        u_trj[t_start:t_end], args.window, args.polyorder,
    )


# Keep q_trj unchanged — it's the reference object trajectory the planner
# achieved, and we don't want to change it and risk missing the lift goal.
# The downstream RL policy handles any tracking error from the smoothing.


def du_stats(u):
    valid_mask = ~np.any(np.isnan(u), axis=1)
    u_valid = u[valid_mask]
    if len(u_valid) < 2:
        return 0.0, 0.0
    du = np.diff(u_valid, axis=0)
    return float(np.sqrt((du ** 2).mean())), float(np.abs(du).max())


rms_before, max_before = du_stats(u_trj)
rms_after, max_after = du_stats(u_smoothed)
print(f"RMS |du|:  {rms_before:.5f} → {rms_after:.5f}")
print(f"Max |du|:  {max_before:.5f} → {max_after:.5f}")

print("Publishing original trajectory (reference object motion)...")
q_vis.publish_trajectory(q_trj, h=h)


src_name = os.path.basename(args.traj_file_path)
assert src_name.startswith("traj_") and src_name.endswith(".npz")
ts = src_name[len("traj_"):-len(".npz")]
out_path = os.path.join(data_folder, f"traj_smoothed_{ts}.npz")

np.savez_compressed(
    out_path,
    q_trj=q_trj,          # unchanged reference object trajectory
    u_trj=u_smoothed,     # smoothed joint targets (what RL tracks)
    h=h,
    q_u_indices_into_x=idx_q_u,
    q_a_indices_into_x=idx_q_a,
)
print(f"Saved smoothed trajectory to {out_path}")
