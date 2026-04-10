import numpy as np
import networkx as nx
from tqdm import tqdm

from pydrake.all import (
    JacobianWrtVariable,
)

from box_lift_setup import *

from irs_rrt.rrt_base import Rrt, Node, Edge
from irs_rrt.rrt_params import RrtParams


class CFNode(Node):
    """
    IrsNode. Each node is responsible for keeping a copy of the bundled dynamics
    and the Gaussian parametrized by the bundled dynamics.
    """

    def __init__(self, q: np.array):
        super().__init__(q)

        # Boolean to say if node can be sampled to go towards goal (should only happen once per node)
        self.to_goal = True

class CollisionFreeRRT(Rrt):
    def __init__(self, cf_params, qu, contact_buffer):
        self.q_lb = np.tile(joint_limits_ur5e[:, 0], 2)
        self.q_ub = np.tile(joint_limits_ur5e[:, 1], 2)

        self.qu = qu
        self.contact_buffer = contact_buffer

        self.dim_u = len(idx_q_a)

        self.params = cf_params
        super().__init__(cf_params)

    def is_collision(self, x):
        """
        Checks if given configuration vector x is in collision.
        """
        q_sim_py.update_mbp_positions_from_vector(x)
        sg = q_sim_py.get_scene_graph()
        query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
        collision_pairs = (
            query_object.ComputeSignedDistancePairwiseClosestPoints(self.contact_buffer)
        )
        inspector = query_object.inspector()

        # 1. Compute closest distance pairs and normals.
        for collision in collision_pairs:
            f_id = inspector.GetFrameId(collision.id_A)
            body_A = plant.GetBodyFromFrameId(f_id)
            f_id = inspector.GetFrameId(collision.id_B)
            body_B = plant.GetBodyFromFrameId(f_id)

            if body_A.model_instance() in [idx_a_l, idx_a_r] or body_B.model_instance() in [idx_a_l, idx_a_r]:
                return True

        return False

    def sample_subgoal(self):
        while True:
            qa = np.random.rand(self.dim_u)
            qa = (self.q_ub - self.q_lb) * qa + self.q_lb

            q_goal = self.map_qa_to_q(qa)
            if not (self.is_collision(q_goal)):
                return qa

    def calc_distance_batch(self, q_query: np.array):
        error_batch = q_query - self.get_q_matrix_up_to(self.size)
        metric_mat = np.diag(np.ones(self.dim_u))

        intsum = np.einsum("Bi,ij->Bj", error_batch, metric_mat)
        metric_batch = np.einsum("Bi,Bi->B", intsum, error_batch)

        return metric_batch

    def map_qa_to_q(self, qa):
        q = np.zeros(dim_x)
        q[idx_q_a] = qa
        q[idx_q_u] = self.qu
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

        child_node = CFNode(xnext)
        child_node.subgoal = q

        edge = Edge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = 0.0

        return child_node, edge, collision
    
    def select_closest_node(
        self, subgoal: np.array, to_goal: bool = False, print_distance: bool = False
    ):
        """
        Given a subgoal, and find the node that is closest from the subgoal.
        """
        if not to_goal:
            return super().select_closest_node(subgoal, print_distance)
        
        d_batch = self.calc_distance_batch(subgoal)
        i_min = np.argsort(d_batch)
        i = 0
        while i < len(i_min):
            node: CFNode = self.get_node_from_id(i_min[i])
            if node.to_goal:
                if print_distance:
                    print("closest distance to subgoal", d_batch[node.id])
                return node
            i += 1
        return None

    def iterate(self, debug=False):
        """
        Main method for iteration.
        """
        pbar = tqdm(total=self.max_size)

        while self.size < self.params.max_size:
            pbar.update(1)

            collision = True
            while collision:
                to_goal = False
                # 1. Sample a subgoal.
                if self.cointoss_for_goal():
                    subgoal = self.params.goal
                    to_goal = True
                    # print("goal")
                else:
                    subgoal = self.sample_subgoal()
                    # print("subgoal")

                # 2. Sample closest node to subgoal
                parent_node = self.select_closest_node(subgoal, to_goal)

                if parent_node is None:
                    continue

                # 3. Extend to subgoal.
                child_node, edge, collision = self.extend_towards_q(
                    parent_node, subgoal
                )


                if debug:
                    q = self.map_qa_to_q(subgoal)
                    q_sim_py.update_mbp_positions_from_vector(q)
                    q_sim_py.draw_current_configuration()
                    input("subgoal")
                    q = self.map_qa_to_q(parent_node.q)
                    q_sim_py.update_mbp_positions_from_vector(q)
                    q_sim_py.draw_current_configuration()
                    input("parent")
                    q = self.map_qa_to_q(child_node.q)
                    q_sim_py.update_mbp_positions_from_vector(q)
                    q_sim_py.draw_current_configuration()
                    input(f"child, collision={collision}")
                    debug = input("Continue debugging?") != "0"

                if to_goal:
                    parent_node.to_goal = False

            # 4. Register the new node to the graph.
            self.add_node(child_node)
            child_node.value = parent_node.value + edge.cost
            self.add_edge(edge)

            if self.size % 50 == 0:
                q = self.map_qa_to_q(child_node.q)
                q_sim_py.update_mbp_positions_from_vector(q)
                q_sim_py.draw_current_configuration()

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

        x_trj = np.zeros((path_T, self.dim_u))

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

        x_trj = np.zeros((path_T, dim_x))

        for i in range(path_T - 1):
            x_trj[
                i, idx_q_a
            ] = self.get_node_from_id(path[i]).q
            x_trj[i, idx_q_u] = self.qu
        x_trj[
            path_T - 1, idx_q_a
        ] = self.get_node_from_id(path[path_T - 1]).q
        x_trj[path_T - 1, idx_q_u] = self.qu

        return x_trj

    def interpolate_traj(self, q_start, q_end, T):
        return np.linspace(q_start, q_end, T)

    def segment_has_no_collision(self, q_start, q_end, T):
        q_trj = self.interpolate_traj(q_start, q_end, T)
        has_collision = False
        for t in range(q_trj.shape[0]):
            if self.is_collision(self.map_qa_to_q(q_trj[t])):
                has_collision = True
                break
        return not has_collision

    def shortcut_path(self, x_trj, num_tries=100):
        x_trj_shortcut = np.copy(x_trj)
        for _ in range(num_tries):
            # choose two random points on the path.
            ind_a, ind_b = np.sort(np.random.choice(x_trj_shortcut.shape[0], 2, replace=False))

            x_a = x_trj_shortcut[
                ind_a, idx_q_a
            ]
            x_b = x_trj_shortcut[
                ind_b, idx_q_a
            ]

            if self.segment_has_no_collision(x_a, x_b, 100):
                # TODO: calculate min step count needed to shortcut a -> b using stepsize
                shortcut_length = np.linalg.norm(x_b - x_a)
                shortcut_step_count = int(np.ceil(shortcut_length / cf_params.stepsize))
                shortcut_trj = self.interpolate_traj(x_a, x_b, shortcut_step_count + 1)

                shortcut_x_trj = np.zeros((len(shortcut_trj), dim_x))
                shortcut_x_trj[:, idx_q_a] = shortcut_trj
                shortcut_x_trj[:, idx_q_u] = self.qu

                x_trj_shortcut = np.concatenate((
                    x_trj_shortcut[:ind_a],
                    shortcut_x_trj,
                    x_trj_shortcut[(ind_b+1):]
                ))

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
        scale
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

    qnext = np.copy(x)
    if min_body_left:
        # 2. Compute Jacobians and qdot.
        J_L = plant.CalcJacobianTranslationalVelocity(
            q_sim_py.context_plant,
            JacobianWrtVariable.kV,
            min_body_left.body_frame(),
            np.array([0, 0, 0]),
            plant.world_frame(),
            plant.world_frame(),
        )

        J_La = J_L[:2, idx_q_a_l]

        qdot_La = np.linalg.pinv(J_La).dot((scale - min_dist_left) * min_normal_left[:2])

        qnext[idx_q_a_l] += qdot_La

    if min_body_right:
        # 2. Compute Jacobians and qdot.
        J_R = plant.CalcJacobianTranslationalVelocity(
            q_sim_py.context_plant,
            JacobianWrtVariable.kV,
            min_body_right.body_frame(),
            np.array([0, 0, 0]),
            plant.world_frame(),
            plant.world_frame(),
        )

        J_Ra = J_R[:2, idx_q_a_r]

        qdot_Ra = np.linalg.pinv(J_Ra).dot((scale - min_dist_right) * min_normal_right[:2])

        qnext[idx_q_a_r] += qdot_Ra

    return qnext


import argparse

from irs_rrt.irs_rrt import IrsRrt

parser = argparse.ArgumentParser()
parser.add_argument("traj_file_path", type=str)
args = parser.parse_args()

data = np.load(args.traj_file_path)

q_knots_trimmed = data["q_trj"]
u_knots_trimmed = data["u_trj"]
# Inherit the input file's rate so refined trajectories don't get clobbered.
h_input = float(data["h"]) if "h" in data.files else rrt_params.h

segments = IrsRrt.get_regrasp_segments(u_knots_trimmed)

q_knots_ref_list = [[q0]]
u_knots_ref_list = [[]]
for t_start, t_end in segments:
    q_knots_ref_list.append(q_knots_trimmed[t_start:(t_end+1)])
    u_knots_ref_list.append(u_knots_trimmed[t_start:t_end])

q_knots_patched_list = []
u_knots_patched_list = []


for n in range(len(q_knots_ref_list) - 1):
    q_start = q_knots_ref_list[n][-1]
    q_end = q_knots_ref_list[n + 1][0]

    qu_start = q_start[idx_q_u]
    qu_end = q_end[idx_q_u]

    assert np.all(qu_start == qu_end)

    step_out_distance = 0.02
    qin = step_out(q_sim, q_sim_py, q_start, scale=step_out_distance)
    qout = step_out(q_sim, q_sim_py, q_end, scale=step_out_distance)

    print("qin")
    q_sim_py.update_mbp_positions_from_vector(qin)
    q_sim_py.draw_current_configuration()
    input("next")

    print("qout")
    q_sim_py.update_mbp_positions_from_vector(qout)
    q_sim_py.draw_current_configuration()
    input("start")

    qa_in = qin[idx_q_a]
    qa_out = qout[idx_q_a]

    cf_params = RrtParams()
    cf_params.goal = qa_out
    cf_params.root_node = CFNode(qa_in)
    cf_params.termination_tolerance = 1e-3
    cf_params.goal_as_subgoal_prob = 0.1
    cf_params.stepsize = rrt_params.stepsize
    cf_params.max_size = 200000
    cf_params.h = h_input

    print(f"qa_in: {qa_in}")
    print(f"qa_out: {qa_out}")

    cf_rrt = CollisionFreeRRT(cf_params, qu_start, 0.0)
    # cf_rrt.q_lb = (
    #     np.minimum(qa_start, qa_end) - 0.3
    # )
    # cf_rrt.q_lb = np.maximum(
    #     cf_rrt.q_lb,
    #     joint_limits[idx_a_l][:,0],
    # )

    # cf_rrt.q_ub = (
    #     np.maximum(qa_start, qa_end) + 0.3
    # )
    # cf_rrt.q_ub = np.minimum(
    #     cf_rrt.q_ub,
    #     joint_limits[idx_a_l][:,1],
    # )

    cf_rrt.iterate(debug=True)

    cf_rrt_trj = cf_rrt.shortcut_path(cf_rrt.get_final_path_q(), num_tries=1000)

    patch_trj = np.zeros((0, dim_x))

    # Move slowly out
    step_in_size = np.linalg.norm((qin - q_start)[idx_q_a])
    step_in_frames = 1 + int(np.ceil(step_in_size / cf_params.stepsize))
    patch_trj = np.vstack(
        (patch_trj, np.linspace(q_start, qin, step_in_frames))
    )
    patch_trj = np.vstack((patch_trj, cf_rrt_trj))

    # # Move slowly in
    step_out_size = np.linalg.norm((qout - q_end)[idx_q_a])
    step_out_frames = 1 + int(np.ceil(step_out_size / cf_params.stepsize))
    patch_trj = np.vstack(
        (patch_trj, np.linspace(qout, q_end, step_out_frames))
    )

    q_vis.publish_trajectory(patch_trj, 0.1)

    input("next")

    q_knots_patched_list.append(q_knots_ref_list[n])
    q_knots_patched_list.append(patch_trj[1:-1]) # first and last already inclded in q_knots_ref_list[n]

    if len(u_knots_ref_list[n]) > 0:
        u_knots_patched_list.append(u_knots_ref_list[n])
    u_knots_patched_list.append(
        patch_trj[1:, idx_q_a]
    )
    

q_knots_patched_list.append(q_knots_ref_list[-1])
u_knots_patched_list.append(u_knots_ref_list[-1])

q_trj_final = np.concatenate(q_knots_patched_list)
u_trj_final = np.concatenate(u_knots_patched_list)


q_vis.publish_trajectory(q_trj_final, h_input)


original_filename = args.traj_file_path.split("/")[-1]
ts = original_filename[5:-4]
new_filename = f"traj_full_{ts}.npz"
output_path = os.path.join(data_folder, new_filename)

print(f"Saved full trajectory to {output_path}")

np.savez_compressed(
    output_path,
    q_trj               = q_trj_final,
    u_trj               = u_trj_final,
    h                   = h_input,
    q_u_indices_into_x  = idx_q_u,
    q_a_indices_into_x  = idx_q_a,
)