"""
Shared iMPC trajectory-refinement driver.

Loads a planner output, splits it into contact segments, runs iMPC on
each one via `IrsMpcQuasistatic.run_traj_opt_on_rrt_segment`, downsamples
to the output rate, stitches NaN regrasp boundaries back in, and saves a
refined trajectory alongside the original.

Defaults match box_push_ur5e (single-arm, follow_trajectory / foh, etc.).
The Q/R cost weights and trust-region size are hardcoded below; tasks
that need different values (e.g. box_lift_ur5e's bimanual structure or
rotation-heavy Q) should add a thin wrapper in their example folder that
calls `main(setup, ..., **overrides)`.

Usage:
    python -m scripts.refine_rrt ptc_data/box_push_ur5e/traj_<ts>.npz
    python -m scripts.refine_rrt <path> --h_small 0.02 --output_hz 50
"""
import argparse
import copy
import os

import numpy as np

# To fix .so import paths for qsim_cpp
import pydrake.common

from qsim_cpp import ForwardDynamicsMode

from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import (
    SmoothingMode,
    IrsMpcQuasistaticParameters,
)

from scripts.task_setup import deduce_setup


def main(
    setup,
    traj_file_path: str,
    h_small: float = 0.05,
    output_hz: int = 10,
    max_iterations: int = 15,
    target_mode: str = "follow_trajectory",
    interp_method: str = "foh",
    use_rate_floor: bool = False,
    length_floor: int = 2,
):
    """Refine a trajectory with iMPC segment-by-segment.

    setup:       task setup module (from scripts.task_setup.deduce_setup).
                 Must expose q_sim, q_parser, q_vis, rrt_params, data_folder,
                 and single-arm attributes idx_u, idx_a, idx_q_u, idx_q_a.
    h_small:     inner-optimization time step (s).
    output_hz:   rate at which to save the refined trajectory. Must divide
                 1/h_small.
    max_iterations: iMPC outer iterations.
    target_mode: passed through to run_traj_opt_on_rrt_segment.
    interp_method: passed through to run_traj_opt_on_rrt_segment (also
                 controls u_trj upsampling mode via calc_u_trj_small).
    use_rate_floor: if True, n_steps_per_h is floored at round(h_planner/h_small).
                    If False, only the length-based floor applies.
    length_floor: minimum n_steps_per_h even when ceil(10/len(u_trj_seg)) < it.
    """
    # ----- Load -----
    data = np.load(traj_file_path)
    q_knots_trimmed = data["q_trj"]
    u_knots_trimmed = data["u_trj"]
    h_planner = float(data["h"])

    assert len(q_knots_trimmed) == len(u_knots_trimmed) + 1, (
        f"Expected len(q) == len(u) + 1, got "
        f"{len(q_knots_trimmed)} vs {len(u_knots_trimmed)}"
    )

    planner_hz = int(round(1.0 / h_planner))
    inner_hz = int(round(1.0 / h_small))
    h_output = 1.0 / output_hz
    assert inner_hz % output_hz == 0, (
        f"Inner rate ({inner_hz} Hz) must be an integer multiple of "
        f"output rate ({output_hz} Hz)"
    )
    inner_to_output_ds = inner_hz // output_hz

    print(f"Loaded trajectory with {len(q_knots_trimmed)} knots at {planner_hz} Hz")
    print(f"iMPC at {inner_hz} Hz → save at {output_hz} Hz "
          f"(downsample factor {inner_to_output_ds})")

    # ----- Build iMPC params (box_push-style single-arm defaults) -----
    q_sim = setup.q_sim
    idx_u = setup.idx_u
    idx_a = setup.idx_a
    idx_q_u = setup.idx_q_u
    idx_q_a = setup.idx_q_a

    params = IrsMpcQuasistaticParameters()
    params.h = h_small

    params.Q_dict = {
        idx_u: np.array([1, 1, 1, 1, 10, 10, 10]),
        idx_a: np.ones(6) * 1e-3,
    }
    params.Qd_dict = {}
    for model in q_sim.get_actuated_models():
        params.Qd_dict[model] = params.Q_dict[model]
    for model in q_sim.get_unactuated_models():
        params.Qd_dict[model] = params.Q_dict[model] * 1000

    params.R_dict = {idx_a: 100 * np.ones(6)}
    params.R_accel_dict = {idx_a: 100 * np.ones(6)}

    dim_u = q_sim.num_actuated_dofs()
    u_size = 10.0
    params.u_bounds_abs = np.array([
        -np.ones(dim_u) * u_size * params.h,
        np.ones(dim_u) * u_size * params.h,
    ])

    params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
    params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
    params.std_u_initial = np.ones(dim_u) * 0.3
    params.num_samples = 100

    params.log_barrier_weight_initial = 100
    log_barrier_final = 1000
    base = np.exp(
        np.log(log_barrier_final / params.log_barrier_weight_initial)
        / max_iterations
    )
    params.calc_log_barrier_weight = lambda kappa0, i: kappa0 * (base ** i)

    params.use_A = False
    params.rollout_forward_dynamics_mode = ForwardDynamicsMode.kSocpMp

    prob_mpc = IrsMpcQuasistatic(
        q_sim=q_sim, parser=setup.q_parser, params=params
    )

    # ----- Refine each contact segment -----
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

        n_for_rate = (
            int(round(h_planner / h_small)) if use_rate_floor else 0
        )
        n_for_length = max(
            length_floor, int(np.ceil(10 / max(len(u_trj_seg), 1)))
        )
        n_steps_per_h = max(n_for_rate, n_for_length)

        # Mutable copy so we can overwrite q_trj_refine[0] (non-penetration
        # projection for carry-over) and q_trj_refine[-1] (final-segment goal).
        q_trj_refine = np.array(q_trj_seg)
        if len(q_trj_refined_list) > 0:
            q_trj_refine[0, idx_q_u] = q_trj_refined_list[-1][-1, idx_q_u]
            q_trj_refine[0] = project_to_non_penetration(q_trj_refine[0])
        if i_s == len(segments) - 1:
            q_trj_refine[-1, idx_q_u] = setup.rrt_params.goal[idx_q_u]

        print(
            f"Segment {i_s}: {len(u_trj_seg)} RRT knots, "
            f"n_steps_per_h={n_steps_per_h}, "
            f"inner knots={len(u_trj_seg) * n_steps_per_h}"
        )

        q_trj_opt, u_trj_opt, idx_best = prob_mpc.run_traj_opt_on_rrt_segment(
            n_steps_per_h=n_steps_per_h,
            q_trj=q_trj_refine,
            u_trj=u_trj_seg,
            max_iterations=max_iterations,
            target_mode=target_mode,
            interp_method=interp_method,
        )

        # Downsample from inner rate to output rate, preserving the final state.
        q_trj_down = q_trj_opt[::inner_to_output_ds]
        if not np.array_equal(q_trj_down[-1], q_trj_opt[-1]):
            q_trj_down = np.concatenate([q_trj_down, q_trj_opt[-1:]], axis=0)
        u_trj_down = u_trj_opt[::inner_to_output_ds]
        assert len(q_trj_down) == len(u_trj_down) + 1, (
            f"Segment {i_s}: len(q_down)={len(q_trj_down)} "
            f"!= len(u_down)+1={len(u_trj_down)+1}"
        )

        print(
            f"  optimized: {len(u_trj_opt)} knots @ {inner_hz} Hz, "
            f"downsampled to {len(u_trj_down)} knots @ {output_hz} Hz "
            f"(best iter {idx_best})"
        )

        q_trj_refined_list.append(q_trj_down)
        u_trj_refined_list.append(u_trj_down)

    if len(q_trj_refined_list) == 0:
        raise RuntimeError("No segments were refined.")

    # ----- Stitch NaN regrasp rows between segments, save -----
    q_trj_parts = []
    u_trj_parts = []
    for i, (q_seg, u_seg) in enumerate(
        zip(q_trj_refined_list, u_trj_refined_list)
    ):
        q_trj_parts.append(q_seg)
        u_trj_parts.append(u_seg)
        if i < len(q_trj_refined_list) - 1:
            u_trj_parts.append(np.full((1, len(idx_q_a)), np.nan))

    q_trj_save = np.concatenate(q_trj_parts, axis=0)
    u_trj_save = np.concatenate(u_trj_parts, axis=0)
    assert len(q_trj_save) == len(u_trj_save) + 1

    print(
        f"Final refined trajectory: {len(q_trj_save)} states, "
        f"{len(u_trj_save)} actions"
    )
    setup.q_vis.publish_trajectory(q_trj_save, h=h_output)

    src_name = os.path.basename(traj_file_path)
    assert src_name.startswith("traj_") and src_name.endswith(".npz")
    ts = src_name[len("traj_"):-len(".npz")]
    out_path = os.path.join(setup.data_folder, f"traj_refined_{ts}.npz")

    out = {k: data[k] for k in data.files}
    out["q_trj"] = q_trj_save
    out["u_trj"] = u_trj_save
    out["h"] = h_output
    out["q_u_indices_into_x"] = idx_q_u
    out["q_a_indices_into_x"] = idx_q_a

    np.savez_compressed(out_path, **out)
    print(f"Saved refined trajectory @ {output_hz} Hz to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file_path", type=str)
    parser.add_argument("--h_small", type=float, default=0.05)
    parser.add_argument("--output_hz", type=int, default=10)
    parser.add_argument("--max_iterations", type=int, default=15)
    parser.add_argument(
        "--target_mode", type=str, default="follow_trajectory",
        choices=["constant_final", "interpolate_endpoints", "follow_trajectory"],
    )
    parser.add_argument(
        "--interp_method", type=str, default="foh",
        choices=["zoh", "foh", "cubic"],
    )
    args = parser.parse_args()

    data = np.load(args.traj_file_path)
    task_name = str(data["task_name"])
    data.close()

    setup = deduce_setup(task_name)
    main(
        setup,
        args.traj_file_path,
        h_small=args.h_small,
        output_hz=args.output_hz,
        max_iterations=args.max_iterations,
        target_mode=args.target_mode,
        interp_method=args.interp_method,
    )
