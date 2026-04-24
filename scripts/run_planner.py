"""
Shared RRT planning driver for any box-push/lift/rotate example.

Handles:
  - Setting up the IrsRrtTrajectory with the task setup module's rrt_params
    and contact sampler
  - Profiling
  - Running iterate()
  - Stripping any leading regrasp NaN row
  - Asserting the new len(q) == len(u) + 1 trajectory format
  - Saving the trajectory + tree with a timestamp

Per-example wrapper (thin):

    import box_push_setup as setup
    from scripts.run_planner import main
    main(setup)

Or run directly against an example by task name:

    python -m scripts.run_planner box_push_ur5e
"""
import argparse
import os
import cProfile
import pstats
from datetime import datetime

import numpy as np

from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory
from scripts.task_setup import deduce_setup, metadata_from_setup


def main(setup, q_sim_smooth=None, post_setup=None):
    """
    Run the trajectory-RRT planner for the given task setup module.

    Args:
      setup: the task setup module (e.g. box_push_setup). Must expose the
             attributes built by any of the *_setup.py files:
             q0, q_sim, q_sim_py, q_vis, rrt_params, contact_sampler,
             pose_sampling_function, idx_u, data_folder.
      q_sim_smooth: optional separate simulator for gradient computation.
                    Defaults to None (same scene for gradients).
      post_setup: optional callable(setup) invoked right after constructing
                  the planner but before iterate(). Use this for tiny
                  task-specific tweaks (e.g. setting
                  contact_sampler.flip_axis_prob).
    """
    profiler = cProfile.Profile()
    profiler.enable()

    setup.q_sim_py.update_mbp_positions_from_vector(setup.q0)
    setup.q_sim_py.draw_current_configuration()

    prob_rrt = IrsRrtTrajectory(
        setup.rrt_params,
        setup.contact_sampler,
        setup.q_sim,
        setup.q_sim_py,
        setup.pose_sampling_function,
        q_sim_smooth=q_sim_smooth,
    )

    setup.q_vis.draw_object_triad(
        length=0.1, radius=0.001, opacity=1, path="sphere/sphere"
    )

    if post_setup is not None:
        post_setup(setup)

    time_to_dist_to_goal = prob_rrt.iterate()
    time_to_dist_to_goal = np.array(time_to_dist_to_goal)

    (
        q_knots_trimmed,
        u_knots_trimmed,
    ) = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
    setup.q_vis.publish_trajectory(q_knots_trimmed, h=setup.rrt_params.h)

    # Strip the leading NaN regrasp row if present. The RRT always starts
    # from q0 with an initial-contact-sample edge (NaN u); trim_regrasps
    # collapses consecutive NaNs so there's at most one leading NaN row.
    # It carries no joint-target information and is redundant because
    # collision_free_rrt.py unconditionally plans a q0->first-contact path.
    if len(u_knots_trimmed) > 0 and np.any(np.isnan(u_knots_trimmed[0])):
        u_knots_trimmed = u_knots_trimmed[1:]
        q_knots_trimmed = q_knots_trimmed[1:]
        print("Stripped leading regrasp (NaN u_0) from saved trajectory.")

    # Natural format: q_trj has one more element than u_trj.
    assert len(q_knots_trimmed) == len(u_knots_trimmed) + 1, (
        f"Expected len(q) == len(u) + 1, got "
        f"{len(q_knots_trimmed)} vs {len(u_knots_trimmed)}"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prob_rrt.save_tree(os.path.join(setup.data_folder, f"tree_{ts}.pkl"))

    # Bundle task metadata into the saved trajectory so downstream tools
    # (refine_rrt, collision_free_rrt, prepare_for_isaaclab) don't need
    # to import the setup module.
    md = metadata_from_setup(setup)

    np.savez_compressed(
        os.path.join(setup.data_folder, f"traj_{ts}.npz"),
        q_trj               = q_knots_trimmed,
        u_trj               = u_knots_trimmed,
        h                   = setup.rrt_params.h,
        q_u_indices_into_x  = prob_rrt.q_u_indices_into_x,
        q_a_indices_into_x  = prob_rrt.q_a_indices_into_x,
        **md,
    )

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(30)

    return prob_rrt, q_knots_trimmed, u_knots_trimmed, ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task_name",
        type=str,
        help="Task name, e.g. box_push_ur5e. Loaded via deduce_setup().",
    )
    args = parser.parse_args()

    setup = deduce_setup(args.task_name)
    main(setup)
