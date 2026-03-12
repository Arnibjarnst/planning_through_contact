import numpy as np
from datetime import datetime

from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory

from box_lift_setup import *

ts = np.linspace(0, 1, 11)


q0_dict = {idx_a_l: q_a0_l, idx_a_r: q_a0_r, idx_u: pose_sampling_function(ts[0])}
q0 = q_sim.get_q_vec_from_dict(q0_dict)

rrt_params.goal[idx_q_u] = pose_sampling_function(ts[1])

i = 1

rrt_params.initial_contact_samples = 128

q_trj = np.zeros((1, len(q0)))
q_trj[0] = np.array(q0)

while i < len(ts):
    t1 = ts[i]
    rrt_params.goal[idx_q_u] = pose_sampling_function(t1)
    rrt_params.root_node = IrsTrajectoryNode(q0)
    rrt_params.subgoal_ts = [ts[i-1], ts[i]]

    prob_rrt = IrsRrtTrajectory(
        rrt_params,
        contact_sampler,
        q_sim,
        q_sim_py,
        pose_sampling_function,
        # q_sim_smooth=q_sim_smooth # Different scene for gradients
        q_sim_smooth=None # Same scene for gradients
    )

    prob_rrt.iterate()

    (
        q_knots_trimmed,
        u_knots_trimmed,
    ) = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
    q_vis.publish_trajectory(q_knots_trimmed, h=rrt_params.h)

    input("next")

    q0 = q_knots_trimmed[-1]
    i += 1
    rrt_params.initial_contact_samples = 0

    q_trj = np.concatenate((q_trj, q_knots_trimmed[1:]))


q_vis.publish_trajectory(q_trj, h=rrt_params.h)
