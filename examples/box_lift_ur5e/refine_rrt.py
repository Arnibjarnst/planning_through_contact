"""
iMPC refinement of an RRT trajectory for the box_lift_ur5e task (bimanual).

Takes a raw planner output `traj_<ts>.npz` and produces a smoother
`traj_refined_<ts>.npz` of the same format by running
IrsMpcQuasistatic.run_traj_opt_on_rrt_segment on each contact segment.

The R cost penalizes du = u_t - u_{t-1} directly (see irs_mpc.py:209-213),
so a high R yields smooth joint targets.

The inner optimization runs at h_small = 0.01 s, then the result is
downsampled back to the planner's h (0.1 s = 10 Hz) before saving, so
collision_free_rrt.py and prepare_for_isaaclab.py work unchanged.

Usage:
    python refine_rrt.py ptc_data/box_lift_ur5e/traj_<ts>.npz
"""
import argparse
import copy
import os
import numpy as np

from qsim_cpp import ForwardDynamicsMode

from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters

from box_lift_setup import *


# --------------------------------------------------------------------------
# Parse CLI
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("traj_file_path", type=str)
parser.add_argument("--h_small", type=float, default=0.01)
parser.add_argument(
    "--output_hz", type=int, default=50,
    help="Rate (Hz) at which to save the refined trajectory. "
         "Should match the downstream RL policy frequency.",
)
parser.add_argument("--max_iterations", type=int, default=20)
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
# Configure iMPC (bimanual, smoothness-leaning)
# --------------------------------------------------------------------------
params = IrsMpcQuasistaticParameters()
params.h = args.h_small

params.Q_dict = {
    idx_u: np.array([10, 10, 10, 10, 1, 1, 1]),  # lift task: rotation matters
    idx_a_l: np.ones(6) * 1e-3,
    idx_a_r: np.ones(6) * 1e-3,
}
params.Qd_dict = {}
for model in q_sim.get_actuated_models():
    params.Qd_dict[model] = params.Q_dict[model]
for model in q_sim.get_unactuated_models():
    params.Qd_dict[model] = params.Q_dict[model] * 200

# HIGH R → smoothness
params.R_dict = {
    idx_a_l: 100 * np.ones(6),
    idx_a_r: 100 * np.ones(6),
}

dim_u = q_sim.num_actuated_dofs()
u_size = 2.0
params.u_bounds_abs = np.array([
    -np.ones(dim_u) * u_size * params.h,
    np.ones(dim_u) * u_size * params.h,
])

params.smoothing_mode = SmoothingMode.k1AnalyticIcecream

params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
params.std_u_initial = np.ones(dim_u) * 0.3
params.num_samples = 100

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
# Refine each segment
# --------------------------------------------------------------------------
segments = IrsRrt.get_regrasp_segments(u_knots_trimmed)
print(f"Found {len(segments)} contact segments: {segments}")

sim_params_projection = copy.deepcopy(q_sim.get_sim_params())
sim_params_projection.h = h_planner
sim_params_projection.unactuated_mass_scale = 1e-4


def project_to_non_penetration(q):
    return q_sim.calc_dynamics(q, q[idx_q_a], sim_params_projection)


q_trj_refined_list = []
u_trj_refined_list = []

for i_s, (t_start, t_end) in enumerate(segments):
    u_trj_seg = u_knots_trimmed[t_start:t_end]
    q_trj_seg = q_knots_trimmed[t_start:t_end + 1]

    if len(u_trj_seg) == 0:
        continue

    # n_steps_per_h: upsample each planner knot into this many inner knots.
    # Aim for the inner rate AND at least 10 knots per segment.
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

    print(
        f"Segment {i_s}: {len(u_trj_seg)} knots, "
        f"n_steps_per_h={n_steps_per_h}, inner knots={len(u_trj_seg) * n_steps_per_h}"
    )

    q_trj_opt, u_trj_opt, idx_best = prob_mpc.run_traj_opt_on_rrt_segment(
        n_steps_per_h=n_steps_per_h,
        h_small=args.h_small,
        q0=q0,
        q_final=q_final,
        u_trj=u_trj_seg,
        max_iterations=max_iterations,
    )

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
# Concatenate and save (matching planner format)
# --------------------------------------------------------------------------
if len(q_trj_refined_list) == 0:
    raise RuntimeError("No segments were refined.")

q_trj_parts = []
u_trj_parts = []
for i, (q_seg, u_seg) in enumerate(zip(q_trj_refined_list, u_trj_refined_list)):
    q_trj_parts.append(q_seg[:-1])
    u_trj_parts.append(u_seg)
    if i < len(q_trj_refined_list) - 1:
        q_trj_parts.append(q_seg[-1:])
        u_trj_parts.append(np.full((1, len(idx_q_a)), np.nan))

q_trj_save = np.concatenate(q_trj_parts, axis=0)
u_trj_save = np.concatenate(u_trj_parts, axis=0)
assert len(q_trj_save) == len(u_trj_save)

q_trj_for_viz = np.concatenate([q_trj_save, q_trj_refined_list[-1][-1:]], axis=0)

print(f"Final refined trajectory: {len(q_trj_save)} states @ {output_hz} Hz")
q_vis.publish_trajectory(q_trj_for_viz, h=h_output)

src_name = os.path.basename(args.traj_file_path)
assert src_name.startswith("traj_") and src_name.endswith(".npz")
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
