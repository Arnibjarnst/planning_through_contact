import os
import pickle
import copy
import numpy as np

from pydrake.all import (
    RigidTransform,
    Quaternion,
)
from pydrake.geometry import Cylinder, Meshcat, MeshcatVisualizer, Rgba, Sphere


import irs_rrt
from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.quasistatic_visualizer import (
    QuasistaticVisualizer,
    InternalVisualizationType,
)
from qsim_cpp import ForwardDynamicsMode, GradientMode

from examples.box_lift.box_lift_setup import *

# pickled_tree_path = os.path.join(
#     os.path.dirname(irs_rrt.__file__),
#     "..",
#     "ptc_data",
#     "box_lift",
#     "randomized",
#     "tree_10000_0.pkl",
# )

# # pickled_q_and_u_path = os.path.join(
# #     os.path.dirname(irs_rrt.__file__),
# #     "..",
# #     "bimanual_optimized_q_and_u_trj.pkl",
# # )


# with open(pickled_tree_path, "rb") as f:
#     tree = pickle.load(f)

# # with open(pickled_q_and_u_path, "rb") as f:
# #     q_and_u_dict = pickle.load(f)

# # q_trj_list = q_and_u_dict["q_trj_list"]
# # u_trj_list = q_and_u_dict["u_trj_list"]

# prob_rrt = IrsRrt.make_from_pickled_tree(
#     tree, internal_vis=InternalVisualizationType.Cpp
# )

# q_sim, q_sim_py = prob_rrt.q_sim, prob_rrt.q_sim_py
# q_vis = QuasistaticVisualizer(q_sim=q_sim, q_sim_py=q_sim_py)

# # get goal and some problem data from RRT parameters.
# q_u_goal = prob_rrt.rrt_params.goal[q_sim.get_q_u_indices_into_q()]
# Q_WB_d = Quaternion(q_u_goal[:4])
# p_WB_d = q_u_goal[4:]
# dim_q = prob_rrt.dim_q
# dim_u = q_sim.num_actuated_dofs()

# # Get trimmed trajectory
# q_knots_trimmed, u_knots_trimmed = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
# # split trajectory into segments according to re-grasps.
# segments = prob_rrt.get_regrasp_segments(u_knots_trimmed)

# # Calculate contact forces
# sim_params = copy.deepcopy(q_sim.get_sim_params())
# # sim_params.h = q_and_u_dict["h_small"]
# sim_params.gradient_mode = GradientMode.kNone
# # Have to use kSocpMP or kQpMp to be able to calculate contact forces
# sim_params.calc_contact_forces = True
# sim_params.forward_mode = ForwardDynamicsMode.kSocpMp


# indices_q_a_into_q = q_sim.get_q_a_indices_into_q()
# indices_q_u_into_q = q_sim.get_q_u_indices_into_q()

# contact_results_list = []
# q_knots_computed_all = np.copy(q_knots_trimmed)
# for i_s, (t_start, t_end) in enumerate(segments):
#     u_trj = u_knots_trimmed[t_start:t_end]
#     q_trj = q_knots_trimmed[t_start : t_end + 1]

#     q_knots_computed = np.zeros_like(q_trj)
#     q_knots_computed[0] = q_trj[0]

#     # calc contact results for the first state in q_trj.
#     q_sim.calc_dynamics(q_trj[0], q_trj[0, indices_q_a_into_q], sim_params)
#     contact_results_list.append(q_sim.get_contact_results_copy())

#     T = len(u_trj)
#     for t in range(T):
#         q_knots_computed[t + 1] = q_sim.calc_dynamics(
#             q_knots_computed[t], u_trj[t], sim_params
#         )
#         contact_results_list.append(q_sim.get_contact_results_copy())

#     print(
#         "q_knots_norm_diff trimmed vs computed",
#         np.linalg.norm(q_knots_computed - q_trj),
#     )

#     q_knots_computed_all[t_start: t_end + 1] = q_knots_computed
#     # assert np.allclose(q_knots_computed, q_trj, atol=1e-4)


# idx_a_l = q_vis.plant.GetModelInstanceByName(eef_l_name)
# idx_a_r = q_vis.plant.GetModelInstanceByName(eef_r_name)

# b_ids_l = q_vis.plant.GetBodyIndices(idx_a_l)
# b_ids_r = q_vis.plant.GetBodyIndices(idx_a_r)

# for contact_result in contact_results_list:
#     for i in range(contact_result.num_point_pair_contacts()):
#         ci = contact_result.point_pair_contact_info(i)

#         # f_W points from A into B.
#         f_origin = ci.contact_point()
#         f_v = ci.contact_force()
#         f_v_norm = np.linalg.norm(f_v)

#         if f_v_norm < 1e-3:
#             continue

#         b_id_A = ci.bodyA_index()
#         b_id_B = ci.bodyB_index()


# # visualize goal.
# q_vis.draw_object_triad(
#     length=0.1,
#     radius=0.001,
#     opacity=1,
#     path="sphere/sphere",
# )
# q_vis.draw_goal_triad(
#     length=0.1, radius=0.005, opacity=0.7, X_WG=RigidTransform(Q_WB_d, p_WB_d)
# )

# # q_vis.publish_trajectory(q_knots_trimmed, prob_rrt.rrt_params.h, contact_results_list)

# import time

# h_vis = 0.5
# while True:
#     start_time = time.time()
#     last_i = -1

#     while last_i < len(q_knots_computed_all) - 1:
#         curr_time = (time.time() - start_time)
#         curr_i = int(curr_time / h_vis)

#         if curr_i == last_i:
#             continue
        

#         q_sim_py.meshcat.Delete("visualizer/contact/")


#         contact_result = contact_results_list[curr_i]
#         for j in range(contact_result.num_point_pair_contacts()):
#             cj = contact_result.point_pair_contact_info(j)
#             f_origin = cj.contact_point()
#             f_v = cj.contact_force()
#             f_v_norm = np.linalg.norm(f_v)
#             f_v /= f_v_norm

#             if f_v_norm < 1e-3:
#                 continue

#             path = f"visualizer/contact/{curr_i}/{j}"

#             if curr_i == len(contact_results_list) - 1:
#                 print(f_v, f_v_norm)

#             orig_direction = np.array([0,0,1])
#             xyz = np.cross(f_v, orig_direction)
#             w = 1 + np.dot(f_v, orig_direction)
#             wxyz = np.array([w, xyz[0], xyz[1], xyz[2]])
#             wxyz /= np.linalg.norm(wxyz)
#             quat = Quaternion(wxyz)

#             q_sim_py.meshcat.SetTransform(path, RigidTransform(quat, f_origin))

#             q_sim_py.meshcat.SetObject(
#                 path + "/arrow", Cylinder(0.002, f_v_norm / 100.0), Rgba(1, 0, 0, 1.0)
#             )

#         q_vis.draw_configuration(q_knots_computed_all[curr_i])

#         last_i = curr_i


#     # while curr_time - start_time < T:
#     #     curr_time_i = (curr_time - start_time) / prob_rrt.rrt_params.h
        
#     #     curr_time_interp = curr_time_i - int(curr_time_i)
#     #     curr_time_i = int(curr_time_i)

#     #     q_curr = np.copy(q_knots_computed_all[curr_time_i])
#     #     q_next = np.copy(q_knots_computed_all[curr_time_i + 1])
        
#     #     # interpolate eef pos
#     #     q_curr[indices_q_a_into_q] += (q_next[indices_q_a_into_q] - q_curr[indices_q_a_into_q]) * curr_time_interp

#     #     # interpolate obj pos
#     #     obj_pos_idxs = indices_q_u_into_q[4:]
#     #     q_curr[obj_pos_idxs] += (q_next[obj_pos_idxs] - q_curr[obj_pos_idxs]) * curr_time_interp

#     #     # Slerp obj rotation
#     #     obj_quat_idxs = indices_q_u_into_q[:4]
#     #     curr_quat = Quaternion(q_curr[obj_quat_idxs])
#     #     next_quat = Quaternion(q_next[obj_quat_idxs])
#     #     q_curr[obj_quat_idxs] = curr_quat.slerp(curr_time_interp, next_quat).wxyz()

#     #     RigidTransform()

#     #     q_sim_py.meshcat.SetTransform(path, RigidTransform(Quaternion(1,0,0,0), f_origin))

#     #     q_sim_py.meshcat.SetObject(
#     #         path + "/arrow", Cylinder(0.01, f_W_norm / 100.0), Rgba(1, 0, 0, 1.0)
#     #     )

#     #     q_vis.draw_configuration(q_curr)

#     #     curr_time = time.time()

#     input("press Enter to reset")