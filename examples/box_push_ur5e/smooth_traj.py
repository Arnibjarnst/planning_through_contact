"""
Non-optimization-based smoothing of box_push_ur5e RRT trajectories.

Applies several smoothing filters to the raw RRT joint-target trajectory
so they can be compared side-by-side. For each method, we also re-roll
out the quasistatic dynamics with the smoothed actions so we can see
how well the object trajectory tracks the planner's reference — i.e.
smoothness of targets vs. performance degradation.

Methods (7 total, all non-optimization):
  1. savgol           - Savitzky-Golay (sliding polynomial LSQ)
  2. butter           - Butterworth zero-phase IIR (filtfilt)
  3. gaussian         - Gaussian kernel FIR (scipy.ndimage.gaussian_filter1d)
  4. cubic_smooth     - Cubic smoothing spline (UnivariateSpline with s>0)
  5. cubic_interp     - Cubic interpolating spline (CubicSpline, passes through knots)
  6. moving_avg       - Uniform moving-average (dumb FIR baseline)
  7. bessel           - Bessel zero-phase IIR (filtfilt, nearly linear phase)

Outputs:
  - One per-method npz: `traj_smoothed_<method>_<ts>.npz`  (pipeline-compatible)
  - One comparison npz:  `traj_comparison_<ts>.npz`         (for the notebook)

The per-method files have the same schema as the planner output:
    q_trj, u_trj, h, q_u_indices_into_x, q_a_indices_into_x
so they can be fed straight into collision_free_rrt.py.

References for the filters:
  - Savitzky, A. & Golay, M.J.E. (1964). Analytical Chemistry 36(8), 1627-1639.
  - Schafer, R.W. (2011). "What is a Savitzky-Golay filter?" IEEE SPM 28(4).
  - Crenna, F., Rossi, G.B., Berardengo, M. (2021). "Filtering biomechanical
    signals in movement analysis." Sensors 21(13), 4580.

Usage:
    python smooth_traj.py ptc_data/box_push_ur5e/traj_<ts>.npz \\
        [--window 11] [--polyorder 3] [--cutoff_hz 5.0] [--gaussian_sigma 2.0]
"""
import argparse
import copy
import os
import numpy as np

from scipy.signal import savgol_filter, butter, bessel, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline, CubicSpline

from irs_rrt.irs_rrt import IrsRrt

from box_push_setup import *


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("traj_file_path", type=str)
parser.add_argument(
    "--window", type=int, default=11,
    help="Savitzky-Golay window length (odd) and moving-average window.",
)
parser.add_argument("--polyorder", type=int, default=3, help="Savitzky-Golay polynomial order.")
parser.add_argument(
    "--cutoff_hz", type=float, default=5.0,
    help="Cutoff frequency (Hz) for Butterworth and Bessel low-pass.",
)
parser.add_argument(
    "--iir_order", type=int, default=4,
    help="Order of Butterworth and Bessel filters.",
)
parser.add_argument(
    "--gaussian_sigma", type=float, default=2.0,
    help="Gaussian kernel sigma in samples.",
)
parser.add_argument(
    "--spline_smooth_factor", type=float, default=None,
    help="Smoothing factor `s` for UnivariateSpline. "
         "None = scipy default (~length of data).",
)
args = parser.parse_args()


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
data = np.load(args.traj_file_path)
q_trj_original = data["q_trj"]
u_trj_original = data["u_trj"]
h = float(data["h"])
fs = 1.0 / h  # sample rate

# Backward-compat: convert old format len(q) == len(u) to natural format
# len(q) == len(u) + 1 by appending the post-state of the last u.
if len(q_trj_original) == len(u_trj_original):
    print("Detected old trajectory format; appending post-state.")
    sim_params_legacy = copy.deepcopy(q_sim.get_sim_params())
    sim_params_legacy.h = h
    q_post_last = q_sim.calc_dynamics(
        q_trj_original[-1], u_trj_original[-1], sim_params_legacy
    )
    q_trj_original = np.concatenate(
        [q_trj_original, q_post_last[None, :]], axis=0
    )
assert len(q_trj_original) == len(u_trj_original) + 1, (
    f"Expected len(q) == len(u) + 1, got {len(q_trj_original)} vs {len(u_trj_original)}"
)

print(f"Loaded {len(u_trj_original)} actions @ {fs:.1f} Hz")

segments = IrsRrt.get_regrasp_segments(u_trj_original)
print(f"Found {len(segments)} contact segments: {segments}")


# --------------------------------------------------------------------------
# Per-method segment smoothers
#
# Each smoother takes a NaN-free u segment of shape (T, dim_u) and returns
# a smoothed segment of the same shape. Endpoints are pinned to preserve
# the contact start/end configuration.
# --------------------------------------------------------------------------

def _pin_endpoints(u_smooth, u_orig):
    u_smooth[0] = u_orig[0]
    u_smooth[-1] = u_orig[-1]
    return u_smooth


def smooth_savgol(u, window, polyorder):
    n = len(u)
    if n <= polyorder + 1:
        return u.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < polyorder + 2:
        w = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
        w = min(w, n if n % 2 == 1 else n - 1)
    if w < polyorder + 2:
        return u.copy()
    u_s = savgol_filter(u, window_length=w, polyorder=polyorder, axis=0, mode="nearest")
    return _pin_endpoints(u_s, u)


def smooth_iir(u, design_fn, order, cutoff_hz, fs):
    """Generic zero-phase IIR low-pass via filtfilt."""
    n = len(u)
    if n < 3 * order + 3:
        # filtfilt has a minimum-length requirement; skip if too short.
        return u.copy()
    # Normalize cutoff to Nyquist (fs/2)
    wn = min(cutoff_hz / (0.5 * fs), 0.99)
    b, a = design_fn(order, wn, btype="low")
    # filtfilt along time axis (axis=0)
    u_s = filtfilt(b, a, u, axis=0)
    return _pin_endpoints(u_s, u)


def smooth_butter(u, order, cutoff_hz, fs):
    return smooth_iir(u, butter, order, cutoff_hz, fs)


def smooth_bessel(u, order, cutoff_hz, fs):
    # scipy's bessel takes norm='phase' as default which gives a normalized phase
    def bessel_design(order, wn, btype):
        return bessel(order, wn, btype=btype, norm="phase")
    return smooth_iir(u, bessel_design, order, cutoff_hz, fs)


def smooth_gaussian(u, sigma):
    if len(u) < 3:
        return u.copy()
    u_s = gaussian_filter1d(u, sigma=sigma, axis=0, mode="nearest")
    return _pin_endpoints(u_s, u)


def smooth_cubic_smoothing_spline(u, s_factor):
    """Fit UnivariateSpline (cubic, k=3) with smoothing factor s per joint."""
    n, d = u.shape
    if n < 4:
        return u.copy()
    t = np.arange(n, dtype=float)
    u_s = np.zeros_like(u)
    for j in range(d):
        try:
            s = s_factor if s_factor is not None else n * np.var(u[:, j]) * 0.01
            spl = UnivariateSpline(t, u[:, j], k=3, s=s)
            u_s[:, j] = spl(t)
        except Exception:
            u_s[:, j] = u[:, j]
    return _pin_endpoints(u_s, u)


def smooth_cubic_interp_spline(u):
    """
    Cubic spline that passes through every original knot — does NOT
    actually smooth the raw u_trj (it reproduces it exactly at the
    knots). Included as a reference; only useful if you're evaluating
    at finer time points. For fair comparison at the original knots,
    this returns the original unchanged (it's a no-op at the knots).
    """
    n = len(u)
    if n < 4:
        return u.copy()
    t = np.arange(n, dtype=float)
    cs = CubicSpline(t, u, axis=0)
    return cs(t)  # at the knots, equals u exactly


def smooth_moving_avg(u, window):
    """Causal-friendly centered moving average (implemented as uniform convolution)."""
    n = len(u)
    if n < 3:
        return u.copy()
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return u.copy()
    # Uniform kernel, convolve along time axis for each joint.
    kernel = np.ones(w) / w
    u_s = np.zeros_like(u)
    pad = w // 2
    for j in range(u.shape[1]):
        padded = np.pad(u[:, j], (pad, pad), mode="edge")
        u_s[:, j] = np.convolve(padded, kernel, mode="valid")
    return _pin_endpoints(u_s, u)


# --------------------------------------------------------------------------
# Apply all methods segment-wise
# --------------------------------------------------------------------------
def apply_segmentwise(u_full, smoother):
    """Apply smoother to each contact segment independently, preserving NaN rows."""
    out = u_full.copy()
    for t_start, t_end in segments:
        if t_end > t_start:
            out[t_start:t_end] = smoother(u_full[t_start:t_end])
    return out


methods = {
    "savgol":       lambda u: smooth_savgol(u, args.window, args.polyorder),
    "butter":       lambda u: smooth_butter(u, args.iir_order, args.cutoff_hz, fs),
    "gaussian":     lambda u: smooth_gaussian(u, args.gaussian_sigma),
    "cubic_smooth": lambda u: smooth_cubic_smoothing_spline(u, args.spline_smooth_factor),
    "cubic_interp": lambda u: smooth_cubic_interp_spline(u),
    "moving_avg":   lambda u: smooth_moving_avg(u, args.window),
    "bessel":       lambda u: smooth_bessel(u, args.iir_order, args.cutoff_hz, fs),
}

u_smoothed_by_method = {}
for name, fn in methods.items():
    u_smoothed_by_method[name] = apply_segmentwise(u_trj_original, fn)


# --------------------------------------------------------------------------
# Re-roll out each smoothed u_trj through the quasistatic simulator so we
# can compare the resulting object trajectory to the original (reference)
# object trajectory.
# --------------------------------------------------------------------------
sim_params = copy.deepcopy(q_sim.get_sim_params())
sim_params.h = h


def rollout(u_trj, q0):
    """
    Roll out the quasistatic dynamics from q0 following u_trj.
    NaN action rows are passed through (q jumps to the post-regrasp state
    from the original q_trj). Returns q of length T+1 (natural MPC format).
    """
    T = len(u_trj)
    q = np.zeros((T + 1, q0.shape[0]))
    q[0] = q0
    q_curr = q0.copy()
    for t in range(T):
        if np.any(np.isnan(u_trj[t])):
            # Regrasp: jump to post-regrasp state from the reference
            q_curr = q_trj_original[t + 1].copy()
            q[t + 1] = q_curr
            continue
        q_curr = q_sim.calc_dynamics(q_curr, u_trj[t], sim_params)
        q[t + 1] = q_curr
    return q


q0 = q_trj_original[0]

# Also re-roll out the ORIGINAL u_trj so we can check that rolling out is
# self-consistent (this should roughly match q_trj_original; any drift is
# the integration error of the quasistatic simulator itself).
q_trj_original_rerolled = rollout(u_trj_original, q0)


q_rerolled_by_method = {}
for name, u_s in u_smoothed_by_method.items():
    q_rerolled_by_method[name] = rollout(u_s, q0)


# --------------------------------------------------------------------------
# Smoothness stats
# --------------------------------------------------------------------------
def du_stats(u):
    valid = ~np.any(np.isnan(u), axis=1)
    u_v = u[valid]
    if len(u_v) < 2:
        return 0.0, 0.0
    du = np.diff(u_v, axis=0)
    return float(np.sqrt((du ** 2).mean())), float(np.abs(du).max())


def goal_error(q_rerolled):
    """Distance from the re-rolled final object pose to the planned final object pose."""
    q_u_ref  = q_trj_original[-1, idx_q_u]
    q_u_sim  = q_rerolled[-1, idx_q_u]
    pos_err  = float(np.linalg.norm(q_u_ref[4:] - q_u_sim[4:]))
    quat_err = float(np.linalg.norm(q_u_ref[:4] - q_u_sim[:4]))
    return pos_err, quat_err


print(f"\n{'method':<14} {'RMS |du|':>12} {'max |du|':>12} {'goal pos err':>14} {'goal quat err':>14}")
print("-" * 70)
rms_orig, max_orig = du_stats(u_trj_original)
pos_orig, quat_orig = goal_error(q_trj_original_rerolled)
print(f"{'original':<14} {rms_orig:12.5f} {max_orig:12.5f} {pos_orig:14.5f} {quat_orig:14.5f}")
for name in methods:
    rms, mx = du_stats(u_smoothed_by_method[name])
    pos, quat = goal_error(q_rerolled_by_method[name])
    print(f"{name:<14} {rms:12.5f} {mx:12.5f} {pos:14.5f} {quat:14.5f}")


# --------------------------------------------------------------------------
# Save outputs
# --------------------------------------------------------------------------
src_name = os.path.basename(args.traj_file_path)
assert src_name.startswith("traj_") and src_name.endswith(".npz")
ts = src_name[len("traj_"):-len(".npz")]

# 1. Per-method pipeline-compatible files.
#    Each one saves the ORIGINAL q_trj (planner's goal-achieving reference)
#    alongside the smoothed u_trj, same as our previous design.
for name, u_s in u_smoothed_by_method.items():
    out_path = os.path.join(data_folder, f"traj_smoothed_{name}_{ts}.npz")
    np.savez_compressed(
        out_path,
        q_trj=q_trj_original,   # reference trajectory (unchanged)
        u_trj=u_s,              # smoothed joint targets
        h=h,
        q_u_indices_into_x=idx_q_u,
        q_a_indices_into_x=idx_q_a,
    )
print(f"\nWrote {len(methods)} per-method files to {data_folder}")

# 2. Single comparison npz (for the notebook)
comparison_path = os.path.join(data_folder, f"traj_comparison_{ts}.npz")
comparison_data = dict(
    h=h,
    q_u_indices_into_x=idx_q_u,
    q_a_indices_into_x=idx_q_a,
    u_trj_original=u_trj_original,
    q_trj_original=q_trj_original,
    q_trj_original_rerolled=q_trj_original_rerolled,
    method_names=np.array(list(methods.keys())),
)
for name in methods:
    comparison_data[f"u_trj_{name}"] = u_smoothed_by_method[name]
    comparison_data[f"q_trj_{name}_rerolled"] = q_rerolled_by_method[name]

# Save hyperparameters used
comparison_data["params_window"] = args.window
comparison_data["params_polyorder"] = args.polyorder
comparison_data["params_cutoff_hz"] = args.cutoff_hz
comparison_data["params_iir_order"] = args.iir_order
comparison_data["params_gaussian_sigma"] = args.gaussian_sigma
comparison_data["params_spline_smooth_factor"] = (
    args.spline_smooth_factor if args.spline_smooth_factor is not None else -1.0
)

np.savez_compressed(comparison_path, **comparison_data)
print(f"Wrote comparison file to {comparison_path}")


# --------------------------------------------------------------------------
# Interactive viz: pick which trajectory to publish in meshcat.
# --------------------------------------------------------------------------
viz_options = [("original", q_trj_original)]
viz_options += [(name, q_rerolled_by_method[name]) for name in methods]

print("\nAvailable trajectories to visualize:")
for i, (name, _) in enumerate(viz_options):
    print(f"  [{i}] {name}")
print("  [q] quit")

while True:
    try:
        choice = input("\nWhich trajectory to publish? [0-{0}/q]: ".format(len(viz_options) - 1)).strip()
    except EOFError:
        break
    if choice.lower() in ("q", "quit", "exit", ""):
        break
    # Accept either index or name.
    traj = None
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(viz_options):
            traj = viz_options[idx]
    else:
        for name, q in viz_options:
            if name == choice:
                traj = (name, q)
                break
    if traj is None:
        print(f"  unknown choice '{choice}'")
        continue
    name, q = traj
    print(f"  publishing '{name}'...")
    q_vis.publish_trajectory(q, h=h)
