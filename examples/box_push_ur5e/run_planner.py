import os
import numpy as np
from datetime import datetime

import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory

from box_push_setup import *

q_sim_py.update_mbp_positions_from_vector(q0)
q_sim_py.draw_current_configuration()

rrt_params.connect_from_behind = False
rrt_params.connect_to_front = False
rrt_params.joint_limits[idx_u] = np.zeros((7,2))

prob_rrt = IrsRrtTrajectory(
    rrt_params,
    contact_sampler,
    q_sim,
    q_sim_py,
    pose_sampling_function,
    q_sim_smooth=None
)

q_vis.draw_object_triad(
    length=0.1, radius=0.001, opacity=1, path="sphere/sphere"
)


time_to_dist_to_goal = prob_rrt.iterate()


time_to_dist_to_goal = np.array(time_to_dist_to_goal)

(
    q_knots_trimmed,
    u_knots_trimmed,
) = prob_rrt.get_trimmed_q_and_u_knots_to_goal_with_hold()
q_vis.publish_trajectory(q_knots_trimmed, h=rrt_params.h)

# Strip the leading NaN regrasp row if present. The RRT always starts
# from q0 with an initial-contact-sample edge (NaN u); trim_regrasps
# collapses consecutive NaNs so there's at most one leading NaN row.
# It carries no joint-target information and is redundant because
# collision_free_rrt.py unconditionally plans a q0->first-contact path.
if len(u_knots_trimmed) > 0 and np.any(np.isnan(u_knots_trimmed[0])):
    u_knots_trimmed = u_knots_trimmed[1:]
    q_knots_trimmed = q_knots_trimmed[1:]
    print("Stripped leading regrasp (NaN u_0) from saved trajectory.")

assert len(q_knots_trimmed) == len(u_knots_trimmed), (
    f"q_trj and u_trj length mismatch: {len(q_knots_trimmed)} vs {len(u_knots_trimmed)}"
)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
prob_rrt.save_tree(
    os.path.join(
        data_folder, f"tree_{ts}.pkl"
    )
)

np.savez_compressed(
    os.path.join(data_folder, f"traj_{ts}.npz"),
    q_trj               = q_knots_trimmed,
    u_trj               = u_knots_trimmed,
    h                   = rrt_params.h,
    q_u_indices_into_x  = prob_rrt.q_u_indices_into_x,
    q_a_indices_into_x  = prob_rrt.q_a_indices_into_x,
)

profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumtime")

stats.print_stats(30)
