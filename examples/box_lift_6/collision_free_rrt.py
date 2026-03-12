import numpy as np
import networkx as nx
from tqdm import tqdm

from pydrake.all import (
    JacobianWrtVariable,
)

from box_lift_setup import *

from irs_rrt.rrt_base import Rrt, Node, Edge
from irs_rrt.rrt_params import RrtParams


class CollisionFreeRRT(Rrt):
    def __init__(self, irs_rrt, cf_params, qu):
        self.irs_rrt = irs_rrt
        self.q_sim_py = irs_rrt.q_sim_py
        self.q_sim = irs_rrt.q_sim
        self.plant = self.q_sim.get_plant()
        self.sg = self.q_sim.get_scene_graph()

        self.q_lb = irs_rrt.q_lb
        self.q_ub = irs_rrt.q_ub

        # ee_r is fixed so move far away and consider unactuated
        self.qu = np.concatenate([[-100, -100, -100, 0 ,0], qu])
        self.ind_q_u = np.concatenate([idx_q_a_r, idx_q_u])

        self.ind_q_a = idx_q_a_l

        self.dim_x = self.plant.num_positions()
        self.dim_u = len(self.ind_q_a)

        self.ee_l_model_idx = idx_a_l

        self.params = cf_params
        super().__init__(cf_params)

    def is_collision(self, x):
        """
        Checks if given configuration vector x is in collision.
        """
        q_a = x[self.ind_q_a]
        if np.linalg.norm(q_a - self.root_node.q) < 1e-6:
            return False
        self.q_sim_py.update_mbp_positions_from_vector(x)
        # self.q_sim_py.draw_current_configuration()

        plant = self.q_sim_py.get_plant()
        sg = self.q_sim_py.get_scene_graph()
        query_object = sg.GetOutputPort("query").Eval(self.q_sim_py.context_sg)
        collision_pairs = (
            query_object.ComputeSignedDistancePairwiseClosestPoints(0.002)
        )
        inspector = query_object.inspector()

        # 1. Compute closest distance pairs and normals.
        for collision in collision_pairs:
            f_id = inspector.GetFrameId(collision.id_A)
            body_A = plant.GetBodyFromFrameId(f_id)
            f_id = inspector.GetFrameId(collision.id_B)
            body_B = plant.GetBodyFromFrameId(f_id)

            if body_A.model_instance() == self.ee_l_model_idx or body_B.model_instance() == self.ee_l_model_idx:
                return True

        return False

    def sample_subgoal(self):
        while True:
            qa = np.random.rand(self.dim_u)
            qa = (self.q_ub - self.q_lb) * qa + self.q_lb

            q_goal = np.zeros(self.dim_x)
            q_goal[self.ind_q_a] = qa
            q_goal[self.ind_q_u] = self.qu

            if not (self.is_collision(q_goal)):
                return q_goal[self.ind_q_a]

    def calc_distance_batch(self, q_query: np.array):
        error_batch = q_query - self.get_q_matrix_up_to(self.size)
        metric_mat = np.diag(np.ones(self.dim_u))

        intsum = np.einsum("Bi,ij->Bj", error_batch, metric_mat)
        metric_batch = np.einsum("Bi,Bi->B", intsum, error_batch)

        return metric_batch

    def map_qa_to_q(self, qa):
        q = np.zeros(self.dim_x)
        q[self.ind_q_a] = qa
        q[self.ind_q_u] = self.qu
        return q

    def extend_towards_q(self, parent_node: Node, q: np.array):
        q_start = parent_node.q

        # Linearly interpolate with step size.
        distance = np.linalg.norm(q - q_start)
        direction = (q - q_start) / distance

        if distance < self.params.stepsize:
            xnext = q
        else:
            xnext = q_start + self.params.stepsize * direction

        collision = True
        if self.segment_has_no_collision(q_start, xnext, 10):
            collision = False

        child_node = Node(xnext)
        child_node.subgoal = q

        edge = Edge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = 0.0

        q = np.zeros(self.dim_x)
        q[self.ind_q_u] = self.qu
        q[self.ind_q_a] = xnext
        # self.q_sim_py.update_mbp_positions_from_vector(q)
        # self.q_sim_py.draw_current_configuration()

        return child_node, edge, collision

    def iterate(self):
        """
        Main method for iteration.
        """
        pbar = tqdm(total=self.max_size)

        while self.size < self.params.max_size:
            pbar.update(1)

            collision = True
            while collision:

                # 1. Sample a subgoal.
                if self.cointoss_for_goal() or self.is_close_to_goal():
                    subgoal = self.params.goal
                else:
                    subgoal = self.sample_subgoal()

                # 2. Sample closest node to subgoal
                parent_node = self.select_closest_node(subgoal)

                # 3. Extend to subgoal.
                child_node, edge, collision = self.extend_towards_q(
                    parent_node, subgoal
                )

            # 4. Attempt to rewire a candidate child node.
            if self.params.rewire:
                parent_node, child_node, edge = self.rewire(
                    parent_node, child_node
                )

            # 5. Register the new node to the graph.
            self.add_node(child_node)
            child_node.value = parent_node.value + edge.cost
            self.add_edge(edge)

            if pbar.last_print_t % 10 == 0:
                self.q_sim_py.update_mbp_positions_from_vector(child_node.q)
                self.q_sim_py.draw_current_configuration(child_node.q)

            # 6. Check for termination.
            if self.is_close_to_goal():
                print("done")
                self.goal_node_idx = child_node
                break

        pbar.close()

    def get_final_path_qa(self):
        # Find closest to the goal.
        q_final = self.select_closest_node(self.params.goal)

        # Find path from root to goal.
        path = nx.shortest_path(
            self.graph, source=self.root_node.id, target=q_final.id
        )

        path_T = len(path)

        x_trj = np.zeros((path_T, self.dim_q))

        for i in range(path_T - 1):
            x_trj[i, :] = self.get_node_from_id(path[i]).q
        x_trj[path_T - 1, :] = self.get_node_from_id(path[path_T - 1]).q

        return x_trj

    def get_final_path_q(self):
        # Find closest to the goal.
        qa_final = self.select_closest_node(self.params.goal)
        # Find path from root to goal.
        path = nx.shortest_path(
            self.graph, source=self.root_node.id, target=qa_final.id
        )

        path_T = len(path)

        x_trj = np.zeros((path_T, self.dim_x))

        for i in range(path_T - 1):
            x_trj[
                i, self.ind_q_a
            ] = self.get_node_from_id(path[i]).q
            x_trj[i, self.ind_q_u] = self.qu
        x_trj[
            path_T - 1, self.ind_q_a
        ] = self.get_node_from_id(path[path_T - 1]).q
        x_trj[path_T - 1, self.ind_q_u] = self.qu

        return x_trj

    def interpolate_traj(self, q_start, q_end, T):
        return np.linspace(q_start, q_end, T)

    def segment_has_no_collision(self, q_start, q_end, T):
        q_trj = self.interpolate_traj(q_start, q_end, T)
        has_collision = False
        for t in range(q_trj.shape[0]):
            if self.is_collision(self.map_qa_to_q(q_trj[t])):
                has_collision = True
        return not has_collision

    def shortcut_path(self, x_trj, num_tries=100):
        x_trj_shortcut = np.copy(x_trj)
        T = x_trj_shortcut.shape[0]
        for _ in range(num_tries):
            # choose two random points on the path.
            ind_a, ind_b = np.sort(np.random.choice(T, 2, replace=False))

            x_a = x_trj_shortcut[
                ind_a, self.ind_q_a
            ]
            x_b = x_trj_shortcut[
                ind_b, self.ind_q_a
            ]

            if self.segment_has_no_collision(x_a, x_b, 100):
                # TODO: calculate min step count needed to shortcut a -> b using stepsize
                x_trj_shortcut[
                    ind_a:(ind_b+1), self.ind_q_a
                ] = self.interpolate_traj(x_a, x_b, ind_b - ind_a + 1)

        return x_trj_shortcut

def step_out(q_sim, q_sim_py, x, scale=0.06, num_iters=3):
    """
    Given a near-contact configuration, give a trajectory that steps out.
    """
    q_sim_py.update_mbp_positions(q_sim.get_q_dict_from_vec(x))

    plant = q_sim_py.get_plant()
    sg = q_sim_py.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
    collision_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(
        0.2
    )

    inspector = query_object.inspector()

    # 1. Compute closest distance pairs and normals.

    min_dist_left = np.inf
    min_dist_right = np.inf

    min_body_left = None
    min_body_right = None
    min_normal_left = None
    min_normal_right = None

    for collision in collision_pairs:
        f_id = inspector.GetFrameId(collision.id_A)
        body_A = plant.GetBodyFromFrameId(f_id)
        f_id = inspector.GetFrameId(collision.id_B)
        body_B = plant.GetBodyFromFrameId(f_id)

        # left ee collision
        if (body_A.model_instance() == idx_a_l) or (body_B.model_instance() == idx_a_l):
            if collision.distance < min_dist_left:
                min_dist_left = collision.distance
                min_body_left = body_A if body_A.model_instance() == idx_a_l else body_B
                normal_sign = 1 if body_A.model_instance() == idx_a_r else -1
                min_normal_left = normal_sign * collision.nhat_BA_W

        # right ee collision
        if (body_A.model_instance() == idx_a_r) or (body_B.model_instance() == idx_a_r):
            if collision.distance < min_dist_right:
                min_dist_right = collision.distance
                min_body_right = body_A if body_A.model_instance() == idx_a_r else body_B
                normal_sign = 1 if body_A.model_instance() == idx_a_r else -1
                min_normal_right = normal_sign * collision.nhat_BA_W

    # 2. Compute Jacobians and qdot.
    J_L = plant.CalcJacobianTranslationalVelocity(
        q_sim_py.context_plant,
        JacobianWrtVariable.kV,
        min_body_left.body_frame(),
        np.array([0, 0, 0]),
        plant.world_frame(),
        plant.world_frame(),
    )
    
    J_La = J_L[:2, :len(idx_q_a_l)]

    J_R = plant.CalcJacobianTranslationalVelocity(
        q_sim_py.context_plant,
        JacobianWrtVariable.kV,
        min_body_right.body_frame(),
        np.array([0, 0, 0]),
        plant.world_frame(),
        plant.world_frame(),
    )

    J_Ra = J_R[:2, len(idx_q_a_l):len(idx_q_a)]

    qdot_La = np.linalg.pinv(J_La).dot(scale * min_normal_left[:2])
    qdot_Ra = np.linalg.pinv(J_Ra).dot(scale * min_normal_right[:2])

    qdot = np.zeros_like(x)
    qdot[idx_q_a_l] = qdot_La
    qdot[idx_q_a_r] = qdot_Ra
    qnext = x + qdot

    return qnext


def find_collision_free_path(irs_rrt, qa_start, qa_end, qu):
    params = RrtParams()
    params.goal = qa_end
    params.root_node = qa_start

    cf_rrt = CollisionFreeRRT(irs_rrt, params, qu)
    cf_rrt.iterate()


def is_collision(q_sim, x):
    """
    Checks if given configuration vector x is in collision.
    """
    q_sim.update_mbp_positions(q_sim.get_q_dict_from_vec(x))
    # self.q_sim_py.draw_current_configuration()

    sg = q_sim.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim.context_sg)
    collision_pairs = query_object.ComputePointPairPenetration()
    return len(collision_pairs) > 0


def generate_random_configuration(q_sim, qu, q_lb, q_ub):
    """
    Generate random configuration of qa.
    """
    ind_q_u = q_sim.get_q_u_indices_into_q()
    ind_q_a = q_sim.get_q_a_indices_into_q()
    while True:
        x = np.random.rand(q_sim.dim_x)
        x = (q_ub - q_lb) * x + q_lb
        q = np.zeros(q_sim.dim_x)
        q[ind_q_u] = qu
        q[ind_q_a] = x[ind_q_a]

        if not is_collision(q_sim, q):
            return q[ind_q_a]

import argparse
import pickle

from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.quasistatic_visualizer import (
    InternalVisualizationType,
)

parser = argparse.ArgumentParser()
parser.add_argument("tree_file_path", type=str)
args = parser.parse_args()


with open(args.tree_file_path, "rb") as f:
    tree = pickle.load(f)

prob_rrt = IrsRrt.make_from_pickled_tree(
    tree, internal_vis=InternalVisualizationType.Cpp
)

(
    q_knots_trimmed,
    u_knots_trimmed,
) = prob_rrt.get_trimmed_q_and_u_knots_to_goal()

segments = prob_rrt.get_regrasp_segments(u_knots_trimmed)

q_knots_ref_list = []
u_knots_ref_list = []
for t_start, t_end in segments:
    q_knots_ref_list.append(q_knots_trimmed[t_start:(t_end+1)])
    u_knots_ref_list.append(u_knots_trimmed[t_start:t_end])

q_knots_patched_list = []
u_knots_patched_list = []

for n in range(len(segments) - 1):
    q_start = q_knots_ref_list[n][-1]
    q_end = q_knots_ref_list[n + 1][0]

    qu_start = q_start[idx_q_u]
    qu_end = q_end[idx_q_u]

    assert np.all(qu_start == qu_end)


    qin = step_out(q_sim, q_sim_py, q_start, scale=0.01)
    qout = step_out(q_sim, q_sim_py, q_end, scale=0.01)

    # print("qstart")
    # q_vis.draw_configuration(q_start)
    # input("next")
    # print("qin")
    # q_vis.draw_configuration(qin)
    # input("next")

    # print("q_end")
    # q_vis.draw_configuration(q_end)
    # input("next")
    # print("qout")
    # q_vis.draw_configuration(qout)
    # input("continue")

    cf_ee_trjs = []

    # Iterate one EE at a time (always just assume it's ee_l)
    for idx_ee in [idx_q_a_l, idx_q_a_r]:
        qa_start = q_start[idx_ee]
        qa_end = q_end[idx_ee]

        qa_in = qin[idx_ee]
        qa_out = qout[idx_ee]

        cf_params = RrtParams()
        cf_params.goal = qa_out
        cf_params.root_node = Node(qa_in)
        cf_params.termination_tolerance = 1e-3
        cf_params.goal_as_subgoal_prob = 0.1
        cf_params.stepsize = rrt_params.stepsize * 2
        cf_params.max_size = 100000
        cf_params.h = rrt_params.h

        cf_rrt = CollisionFreeRRT(prob_rrt, cf_params, qu_start)
        cf_rrt.q_lb = (
            np.minimum(qa_start, qa_end) - 0.3
        )
        cf_rrt.q_lb = np.maximum(
            cf_rrt.q_lb,
            joint_limits[idx_a_l][:,0],
        )

        cf_rrt.q_ub = (
            np.maximum(qa_start, qa_end) + 0.3
        )
        cf_rrt.q_ub = np.minimum(
            cf_rrt.q_ub,
            joint_limits[idx_a_l][:,1],
        )

        cf_rrt.iterate()

        cf_rrt_trj = cf_rrt.shortcut_path(cf_rrt.get_final_path_q())

        cf_ee_trjs.append(cf_rrt_trj[:, idx_q_a_l])

    # piece together the two ee trajectories:
    ee_l_trj_len = cf_ee_trjs[0].shape[0]
    ee_r_trj_len = cf_ee_trjs[1].shape[0]

    cf_trj = np.zeros((
        max(ee_l_trj_len, ee_r_trj_len),
        dim_x
    ))

    cf_trj[:, idx_q_u] = qu_start
    cf_trj[:ee_l_trj_len, idx_q_a_l] = cf_ee_trjs[0]
    cf_trj[:ee_r_trj_len, idx_q_a_r] = cf_ee_trjs[1]

    # Fill rest with last pos (qout)
    cf_trj[ee_l_trj_len:, idx_q_a_l] = cf_ee_trjs[0][-1]
    cf_trj[ee_r_trj_len:, idx_q_a_r] = cf_ee_trjs[1][-1]

    patch_trj = np.zeros((0, dim_x))

    # Move slowly out
    patch_trj = np.vstack(
        (patch_trj, np.linspace(q_start, qin, 10))
    )
    patch_trj = np.vstack((patch_trj, cf_trj))

    # Move slowly in
    patch_trj = np.vstack(
        (patch_trj, np.linspace(qout, q_end, 10))
    )

    # q_vis.publish_trajectory(patch_trj, 0.1)

    # input("next")

    q_knots_patched_list.append(q_knots_ref_list[n])
    q_knots_patched_list.append(patch_trj[1:-1]) # first and last already inclded in q_knots_ref_list[n]

    u_knots_patched_list.append(u_knots_ref_list[n])
    u_knots_patched_list.append(
        patch_trj[1:, idx_q_a]
    )

q_knots_patched_list.append(q_knots_ref_list[-1])
u_knots_patched_list.append(u_knots_ref_list[-1])

q_trj_final = np.concatenate(q_knots_patched_list)
u_trj_final = np.concatenate(u_knots_patched_list)

q_vis.publish_trajectory(q_trj_final, rrt_params.h)


original_filename = args.tree_file_path.split("/")[-1]
ts = original_filename[5:-4]
new_filename = f"traj_full_{ts}.npz"
output_path = os.path.join(data_folder, new_filename)

print(f"Saved full trajectory to {output_path}")

np.savez_compressed(
    output_path,
    q_trj               = q_trj_final[:-1],
    u_trj               = u_trj_final,
    h                   = rrt_params.h,
    q_u_indices_into_x  = prob_rrt.q_u_indices_into_x,
    q_a_indices_into_x  = prob_rrt.q_a_indices_into_x,
)