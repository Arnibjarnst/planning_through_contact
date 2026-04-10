"""
iMPC refinement of an RRT trajectory for the box_push_ur5e task.

Takes a raw planner output `traj_<ts>.npz` and produces a smoother
`traj_refined_<ts>.npz` of the same format by running
IrsMpcQuasistatic.run_traj_opt_on_rrt_segment on each contact segment.

The R cost in IrsMpcQuasistatic is applied to du = u_t - u_{t-1}
(see irs_mpc.py:209-213), so a high R value directly penalizes
jumpy joint targets and yields smooth trajectories.

The inner optimization runs at h_small = 0.01 s, then the result is
downsampled back to the planner's h so prepare_for_isaaclab.py works
unchanged.

Usage:
    python refine_rrt.py ptc_data/box_push_ur5e/traj_<ts>.npz
"""
import argparse
import os
import copy
import numpy as np

import pydrake.all

from qsim_cpp import ForwardDynamicsMode

from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters

from box_push_setup import *


# --------------------------------------------------------------------------
# Parse CLI
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("traj_file_path", type=str)
parser.add_argument(
    "--h_small", type=float, default=0.01,
    help="Inner-optimization time step (s). Typically finer than the planner h.",
)
parser.add_argument(
    "--output_hz", type=int, default=50,
    help="Rate (Hz) at which to save the refined trajectory. "
         "Should match the downstream RL policy frequency. Must divide 1/h_small.",
)
parser.add_argument(
    "--max_iterations", type=int, default=20,
    help="Number of iMPC outer iterations.",
)
args = parser.parse_args()


# --------------------------------------------------------------------------
# Load trajectory
# --------------------------------------------------------------------------
data = np.load(args.traj_file_path)
q_knots_trimmed = data["q_trj"]
u_knots_trimmed = data["u_trj"]
h_planner = float(data["h"])

planner_hz = int(round(1.0 / h_planner))
inner_hz = int(round(1.0 / args.h_small))
output_hz = args.output_hz
h_output = 1.0 / output_hz

assert inner_hz % output_hz == 0, (
    f"Inner rate ({inner_hz} Hz) must be an integer multiple of "
    f"output rate ({output_hz} Hz)"
)
inner_to_output_ds = inner_hz // output_hz

print(f"Loaded trajectory with {len(q_knots_trimmed)} knots at {planner_hz} Hz")
print(f"iMPC at {inner_hz} Hz → save at {output_hz} Hz "
      f"(downsample factor {inner_to_output_ds})")


# --------------------------------------------------------------------------
# Configure iMPC
# --------------------------------------------------------------------------
params = IrsMpcQuasistaticParameters()
params.h = args.h_small

# Q: running state cost. Heavy weight on object x,y position (the push goal),
# very light weight on arm joints (let the optimizer move them freely).
params.Q_dict = {
    idx_u: np.array([1, 1, 1, 1, 50, 50, 5]),
    idx_a: np.ones(6) * 1e-3,
}

# Qd: terminal state cost. Boost object weight by 200x.
params.Qd_dict = {}
for model in q_sim.get_actuated_models():
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_sim.get_unactuated_models():
    params.Qd_dict[model] = params.Q_dict[model] * 200

# R: control cost on du = u_t - u_{t-1}. HIGH to penalize jumpy actions.
params.R_dict = {idx_a: 100 * np.ones(6)}

# Trust region on u_t around x_trj[t, joint_indices] per MPC subproblem.
# Interpreted by irs_mpc.py:536-543 as:
#     x_trj[t, joints] - u_size*h  <=  u_t  <=  x_trj[t, joints] + u_size*h
# This is effectively a joint-velocity cap: ±u_size rad/s. 10 rad/s gives the
# optimizer plenty of room per iteration without violating UR5e limits.
dim_u = q_sim.num_actuated_dofs()
u_size = 10.0
params.u_bounds_abs = np.array([
    -np.ones(dim_u) * u_size * params.h,
    np.ones(dim_u) * u_size * params.h,
])

params.smoothing_mode = SmoothingMode.k1AnalyticIcecream

# Sampling-based bundling (not used with analytic smoothing, but required field)
params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
params.std_u_initial = np.ones(dim_u) * 0.3
params.num_samples = 100

# Analytic bundling schedule
params.log_barrier_weight_initial = 100
log_barrier_weight_final = 6000
max_iterations = args.max_iterations
base = np.exp(
    np.log(log_barrier_weight_final / params.log_barrier_weight_initial)
    / max_iterations
)
params.calc_log_barrier_weight = lambda kappa0, i: kappa0 * (base ** i)

params.use_A = False
params.rollout_forward_dynamics_mode = ForwardDynamicsMode.kSocpMp


prob_mpc = IrsMpcQuasistatic(q_sim=q_sim, parser=q_parser, params=params)


# --------------------------------------------------------------------------
# Refine each regrasp segment
# --------------------------------------------------------------------------
segments = IrsRrt.get_regrasp_segments(u_knots_trimmed)
print(f"Found {len(segments)} contact segments: {segments}")

# Projection sim params for re-seeding q0 after each segment (non-penetration)
sim_params_projection = copy.deepcopy(q_sim.get_sim_params())
sim_params_projection.h = h_planner
sim_params_projection.unactuated_mass_scale = 1e-4


def project_to_non_penetration(q):
    return q_sim.calc_dynamics(q, q[idx_q_a], sim_params_projection)


def linear_upsample(u_trj_coarse, n_steps_per_h):
    """
    Linearly interpolate u_trj from its coarse rate to a finer rate
    n_steps_per_h * (original length). This gives a SMOOTH initial guess
    for the iMPC instead of the zero-order-hold staircase that
    IrsMpcQuasistatic.calc_u_trj_small produces. A smoother initial
    guess means the R cost on du = u_t - u_{t-1} is already small,
    so the optimizer refines a gentle slope instead of trying to
    smooth out sharp jumps within a tight trust region.
    """
    T0, dim_u = u_trj_coarse.shape
    T_fine = T0 * n_steps_per_h
    # Sample positions: k-th fine sample falls at k/n_steps_per_h on the
    # coarse time axis (in RRT-knot units). We linearly interpolate
    # between adjacent coarse knots, clamping at the ends.
    t_fine = np.arange(T_fine) / n_steps_per_h
    i_lo = np.clip(np.floor(t_fine).astype(int), 0, T0 - 1)
    i_hi = np.clip(i_lo + 1, 0, T0 - 1)
    alpha = (t_fine - i_lo)[:, None]
    return (1 - alpha) * u_trj_coarse[i_lo] + alpha * u_trj_coarse[i_hi]


q_trj_refined_list = []
u_trj_refined_list = []

for i_s, (t_start, t_end) in enumerate(segments):
    u_trj_seg = u_knots_trimmed[t_start:t_end]
    q_trj_seg = q_knots_trimmed[t_start:t_end + 1]

    if len(u_trj_seg) == 0:
        continue

    # n_steps_per_h: upsample each RRT knot by this factor so that each
    # inner step has duration h_small. Also enforce >=10 knots per segment
    # so the optimizer has room to work (matches box_lift heuristic).
    n_steps_per_h_for_rate = int(round(h_planner / args.h_small))
    n_steps_per_h_for_length = max(1, int(np.ceil(10 / max(len(u_trj_seg), 1))))
    n_steps_per_h = max(n_steps_per_h_for_rate, n_steps_per_h_for_length)

    q0 = np.array(q_trj_seg[0])
    if len(q_trj_refined_list) > 0:
        q0[idx_q_u] = q_trj_refined_list[-1][-1, idx_q_u]
        q0 = project_to_non_penetration(q0)

    q_final = np.array(q_trj_seg[-1])
    if i_s == len(segments) - 1:
        q_final[idx_q_u] = rrt_params.goal[idx_q_u]

    # Build a linearly-interpolated initial guess at the inner rate.
    u_trj_small = linear_upsample(u_trj_seg, n_steps_per_h)

    # Desired state trajectory: constant at q_final (same convention as
    # IrsMpcQuasistatic.run_traj_opt_on_rrt_segment, but we control
    # the initial guess ourselves).
    q_d = np.copy(q0)
    q_d[idx_q_u] = q_final[idx_q_u]
    T_inner = len(u_trj_small)
    q_trj_d = np.tile(q_d, (T_inner + 1, 1))

    print(
        f"Segment {i_s}: {len(u_trj_seg)} RRT knots, "
        f"n_steps_per_h={n_steps_per_h}, inner knots={T_inner} "
        f"(linear-interp init)"
    )

    prob_mpc.initialize_problem(x0=q0, x_trj_d=q_trj_d, u_trj_0=u_trj_small)
    prob_mpc.iterate(max_iterations=max_iterations, cost_Qu_f_threshold=0)

    q_trj_opt = np.array(prob_mpc.x_trj_best)
    u_trj_opt = np.array(prob_mpc.u_trj_best)
    idx_best = prob_mpc.idx_best

    # Downsample from inner rate (1/h_small) to output rate (output_hz).
    q_trj_down = q_trj_opt[::inner_to_output_ds]
    u_trj_down = u_trj_opt[::inner_to_output_ds]

    print(
        f"  optimized: {len(u_trj_opt)} knots @ {inner_hz} Hz, "
        f"downsampled to {len(u_trj_down)} knots @ {output_hz} Hz "
        f"(best iter {idx_best})"
    )

    q_trj_refined_list.append(q_trj_down)
    u_trj_refined_list.append(u_trj_down)


# --------------------------------------------------------------------------
# Concatenate and save
# --------------------------------------------------------------------------
# Build q_trj_save and u_trj_save with the same invariant the planner writes:
#   len(q_trj_save) == len(u_trj_save)
#   u_trj_save has a row of NaNs at each regrasp boundary between segments
#   q_trj_save[t] is the state immediately before u_trj_save[t]
if len(q_trj_refined_list) == 0:
    raise RuntimeError("No segments were refined.")

q_trj_parts = []
u_trj_parts = []
for i, (q_seg, u_seg) in enumerate(zip(q_trj_refined_list, u_trj_refined_list)):
    # Each segment gives us q of length L+1 and u of length L.
    # We save q_seg[:-1] (length L) paired with u_seg (length L).
    q_trj_parts.append(q_seg[:-1])
    u_trj_parts.append(u_seg)

    # If not the last segment, insert a regrasp marker row:
    #   - q row = final state of this segment
    #   - u row = all NaN
    if i < len(q_trj_refined_list) - 1:
        q_trj_parts.append(q_seg[-1:])
        u_trj_parts.append(np.full((1, len(idx_q_a)), np.nan))

q_trj_save = np.concatenate(q_trj_parts, axis=0)
u_trj_save = np.concatenate(u_trj_parts, axis=0)
assert len(q_trj_save) == len(u_trj_save), (
    f"q_trj and u_trj length mismatch: {len(q_trj_save)} vs {len(u_trj_save)}"
)

# For visualization, we need a q trajectory with the final state included.
q_trj_for_viz = np.concatenate([q_trj_save, q_trj_refined_list[-1][-1:]], axis=0)

print(f"Final refined trajectory: {len(q_trj_save)} states, {len(u_trj_save)} actions")

# Visualize refined trajectory at output rate
q_vis.publish_trajectory(q_trj_for_viz, h=h_output)

# Build output filename by inserting "refined_" after "traj_"
src_name = os.path.basename(args.traj_file_path)
assert src_name.startswith("traj_") and src_name.endswith(".npz"), (
    f"Expected traj_<ts>.npz, got {src_name}"
)
ts = src_name[len("traj_"):-len(".npz")]
out_path = os.path.join(data_folder, f"traj_refined_{ts}.npz")

np.savez_compressed(
    out_path,
    q_trj=q_trj_save,
    u_trj=u_trj_save,
    h=h_output,  # saved at output_hz, not planner rate
    q_u_indices_into_x=idx_q_u,
    q_a_indices_into_x=idx_q_a,
)
print(f"Saved refined trajectory @ {output_hz} Hz to {out_path}")
