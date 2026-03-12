import numpy as np
from datetime import datetime

from irs_rrt.irs_rrt_projection import IrsRrtProjection

from box_lift_setup import *


q_sim_py.update_mbp_positions_from_vector(q0)
q_sim_py.draw_current_configuration()

prob_rrt = IrsRrtProjection(
    rrt_params,
    contact_sampler,
    q_sim,
    q_sim_py,
    pose_sampling_function,
    # q_sim_smooth=q_sim_smooth # Different scene for gradients
    q_sim_smooth=None # Same scene for gradients
)


q_vis.draw_object_triad(
    length=0.1, radius=0.001, opacity=1, path="sphere/sphere"
)

# while True:
#     theta, phi = np.array([float(x) for x in input().split()])
#     q = np.copy(q0)
#     q[idx_q_a_l] = np.array([0, 0, 1, theta, phi])
#     q_sim_py.update_mbp_positions_from_vector(q)
#     q_sim_py.draw_current_configuration()

time_to_dist_to_goal = prob_rrt.iterate_batched(4)

time_to_dist_to_goal = np.array(time_to_dist_to_goal)

(
    q_knots_trimmed,
    u_knots_trimmed,
) = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
q_vis.publish_trajectory(q_knots_trimmed, h=rrt_params.h)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
prob_rrt.save_tree(
    os.path.join(
        data_folder, f"tree_{ts}.pkl"
    )
)

# Connect regrasp segments


np.savez_compressed(
    os.path.join(data_folder, f"traj_{ts}.npz"),
    q_trj               = q_knots_trimmed[:-1],
    u_trj               = u_knots_trimmed,
    h                   = rrt_params.h,
    q_u_indices_into_x  = prob_rrt.q_u_indices_into_x,
    q_a_indices_into_x  = prob_rrt.q_a_indices_into_x,
)

