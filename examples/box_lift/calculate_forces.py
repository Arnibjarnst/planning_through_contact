import os
import pickle
import copy
import numpy as np
import argparse

from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.quasistatic_visualizer import (
    InternalVisualizationType,
)
from qsim_cpp import ForwardDynamicsMode, GradientMode

from examples.box_lift.box_lift_setup import *

parser = argparse.ArgumentParser()
parser.add_argument("tree_file_path", type=str)
args = parser.parse_args()


with open(args.tree_file_path, "rb") as f:
    tree = pickle.load(f)

prob_rrt = IrsRrt.make_from_pickled_tree(
    tree, internal_vis=InternalVisualizationType.Cpp
)

q_sim, q_sim_py = prob_rrt.q_sim, prob_rrt.q_sim_py
plant = q_sim.get_plant()

indices_q_a_into_q = q_sim.get_q_a_indices_into_q()
indices_q_u_into_q = q_sim.get_q_u_indices_into_q()


# Get trimmed trajectory
q_knots_trimmed, u_knots_trimmed = prob_rrt.get_trimmed_q_and_u_knots_to_goal()
# split trajectory into segments according to re-grasps.
segments = prob_rrt.get_regrasp_segments(u_knots_trimmed)

# Calculate contact forces
sim_params = copy.deepcopy(q_sim.get_sim_params())
# sim_params.h = q_and_u_dict["h_small"]
sim_params.gradient_mode = GradientMode.kNone
# Have to use kSocpMp or kQpMp to be able to calculate contact forces
sim_params.calc_contact_forces = True

sim_params.forward_mode = ForwardDynamicsMode.kSocpMp

forces = np.zeros((len(q_knots_trimmed), 6))
idx_a_l = plant.GetModelInstanceByName(eef_l_name)
idx_a_r = plant.GetModelInstanceByName(eef_r_name)

b_ids_l = plant.GetBodyIndices(idx_a_l)
b_ids_r = plant.GetBodyIndices(idx_a_r)
finger_l_idxs = [0,1,2]
finger_r_idxs = [3,4,5]

q_knots_computed_all = np.copy(q_knots_trimmed)
for i_s, (t_start, t_end) in enumerate(segments):
    u_trj = u_knots_trimmed[t_start:t_end]
    q_trj = q_knots_trimmed[t_start : t_end + 1]

    q_knots_computed = np.zeros_like(q_trj)
    q_knots_computed[0] = q_trj[0]

    T = len(u_trj)
    for t in range(T):
        q_knots_computed[t + 1] = q_sim.calc_dynamics(
            q_knots_computed[t], u_trj[t], sim_params
        )
        # q_knots_computed[t+1] = q_sim.calc_dynamics(
        #     q_trj[t], u_trj[t], sim_params
        # )
        contact_result = q_sim.get_contact_results_copy()
        i_f = t_start + t
        for j in range(contact_result.num_point_pair_contacts()):
            cj = contact_result.point_pair_contact_info(j)

            # f_W points from A into B.
            f_v = cj.contact_force()

            b_id_A = cj.bodyA_index()
            b_id_B = cj.bodyB_index()

            if b_id_A in b_ids_l:
                forces[i_f, finger_l_idxs] += f_v
            elif b_id_A in b_ids_r:
                forces[i_f, finger_r_idxs] += f_v

            if b_id_B in b_ids_l:
                forces[i_f, finger_l_idxs] -= f_v
            elif b_id_B in b_ids_r:
                forces[i_f, finger_r_idxs] -= f_v    

    print(
        "q_knots_norm_diff trimmed vs computed",
        np.linalg.norm(q_knots_computed - q_trj),
    )

    q_knots_computed_all[t_start: t_end + 1] = q_knots_computed

    # assert np.allclose(q_knots_computed, q_trj, atol=1e-3)


filename = args.tree_file_path[:-4] + "_w_forces.pkl"

out = {
    "obj_traj": q_knots_computed_all[:, indices_q_u_into_q],
    "eef_traj": q_knots_computed_all[:, indices_q_a_into_q],
    "forces": forces
}

with open(filename, "wb") as f:
    pickle.dump(out, f)


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def plot_vectors(origins, vectors, colors=None):
#     """
#     Plot 3D vectors as arrows starting at given origins.

#     origins: (N, 3) array of starting points
#     vectors: (N, 3) array of vector directions
#     colors:  optional list/array of colors
#     """
#     origins = np.asarray(origins)
#     vectors = np.asarray(vectors)
#     N = origins.shape[0]

#     if colors is None:
#         colors = ['C0'] * N  # default color for all

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     for i in range(N):
#         x, y, z = origins[i]
#         u, v, w = vectors[i]
#         ax.quiver(x, y, z, u, v, w, color=colors[i], length=1, normalize=False)

#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     ax.set_zlim(0, 1)

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Vectors')

#     plt.show()


# for i, q in enumerate(q_knots_computed_all):
#     f = forces[i].reshape((2,3)) / 10.0
#     q_eef = q[indices_q_a_into_q].reshape((2,3))

#     plot_vectors(q_eef, f)



