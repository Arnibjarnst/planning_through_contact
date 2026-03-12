import argparse
import pickle
import numpy as np

from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory
from irs_mpc2.quasistatic_visualizer import (
    QuasistaticVisualizer,
    InternalVisualizationType,
)

parser = argparse.ArgumentParser()
parser.add_argument("tree_file_path", type=str)
args = parser.parse_args()

with open(args.tree_file_path, "rb") as f:
    tree = pickle.load(f)

prob_rrt = IrsRrtTrajectory.make_from_pickled_tree(
    tree, internal_vis=InternalVisualizationType.Cpp
)

q, u = prob_rrt.get_trimmed_q_and_u_knots_to_goal()

q_vis = QuasistaticVisualizer(q_sim=prob_rrt.q_sim, q_sim_py=prob_rrt.q_sim_py)

idx_q_u = prob_rrt.q_u_indices_into_x
idx_q_a = prob_rrt.q_a_indices_into_x

for qi in q:
    # move object in xy plane away from robot
    q_temp = np.copy(qi)
    q_temp[idx_q_u[4]] = 3


    # Simulate one step
    qnext = prob_rrt.q_sim.calc_dynamics(q_temp, q_temp[idx_q_a], prob_rrt.sim_params)
    obj_pose_next = qnext[idx_q_u]

    angleDiff, posDiff = prob_rrt.calc_q_u_diff(q_temp[idx_q_u], obj_pose_next)
    
    print(angleDiff, posDiff)
    q_vis.draw_configuration(q_temp)
    input("Before:")
    q_vis.draw_configuration(qnext)
    input("After:")
    
q_vis.publish_trajectory(q, 0.1)

input("EXIT")

