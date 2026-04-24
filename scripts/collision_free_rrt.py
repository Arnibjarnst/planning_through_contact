"""
Task-agnostic collision-free RRT post-processing.

Loads a planner trajectory, identifies regrasp segments where the object
is stationary but the robot changes contact, and plans collision-free
paths between them. All arms are planned simultaneously in joint space.

Usage:
    python -m scripts.collision_free_rrt ptc_data/box_push_ur5e/traj_<ts>.npz
    python -m scripts.collision_free_rrt <path> --step_out_distance 0.02
"""
import argparse
import os

import numpy as np
import networkx as nx
from tqdm import tqdm

import pydrake.common
from pydrake.all import JacobianWrtVariable

from irs_rrt.irs_rrt import IrsRrt
from irs_rrt.rrt_base import Rrt, Node, Edge
from irs_rrt.rrt_params import RrtParams

from scripts.task_setup import deduce_setup


class CFNode(Node):
    def __init__(self, q: np.array):
        super().__init__(q)
        self.to_goal = True


class CollisionFreeRRT(Rrt):
    def __init__(self, cf_params, qu, idx_q_a, idx_q_u, dim_x,
                 actuated_model_idxs, q_sim_py, plant,
                 joint_limits, contact_buffer=0.0):
        self.qu = qu
        self.idx_q_a = idx_q_a
        self.idx_q_u = idx_q_u
        self.dim_x = dim_x
        self.actuated_model_idxs = actuated_model_idxs
        self.q_sim_py = q_sim_py
        self.plant = plant
        self.contact_buffer = contact_buffer

        self.q_lb = joint_limits[:, 0]
        self.q_ub = joint_limits[:, 1]

        self.dim_u = len(idx_q_a)
        self.params = cf_params
        super().__init__(cf_params)
        
        assert not self.is_collision(self.map_qa_to_q(cf_params.root_node.q))
        assert not self.is_collision(self.map_qa_to_q(cf_params.goal))

    def map_qa_to_q(self, qa):
        q = np.zeros(self.dim_x)
        q[self.idx_q_a] = qa
        q[self.idx_q_u] = self.qu
        return q

    def is_collision(self, x):
        self.q_sim_py.update_mbp_positions_from_vector(x)
        sg = self.q_sim_py.get_scene_graph()
        query_object = sg.GetOutputPort("query").Eval(self.q_sim_py.context_sg)
        collision_pairs = (
            query_object.ComputeSignedDistancePairwiseClosestPoints(self.contact_buffer)
        )
        inspector = query_object.inspector()

        for collision in collision_pairs:
            f_id = inspector.GetFrameId(collision.id_A)
            body_A = self.plant.GetBodyFromFrameId(f_id)
            f_id = inspector.GetFrameId(collision.id_B)
            body_B = self.plant.GetBodyFromFrameId(f_id)

            if body_A.model_instance() in self.actuated_model_idxs or \
               body_B.model_instance() in self.actuated_model_idxs:
                return True

        return False

    def sample_subgoal(self):
        while True:
            qa = np.random.uniform(self.q_lb, self.q_ub)
            if not self.is_collision(self.map_qa_to_q(qa)):
                return qa

    def calc_distance_batch(self, q_query: np.array):
        error_batch = q_query - self.get_q_matrix_up_to(self.size)
        return np.sum(error_batch ** 2, axis=1)

    def extend_towards_q(self, parent_node: Node, q: np.array):
        q_start = parent_node.q
        distance = np.linalg.norm(q - q_start)
        direction = (q - q_start) / distance

        if distance < self.params.stepsize:
            xnext = q
        else:
            xnext = q_start + self.params.stepsize * direction

        collision = not self.segment_has_no_collision(q_start, xnext, 10)

        child_node = CFNode(xnext)
        child_node.subgoal = q

        edge = Edge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = 0.0

        return child_node, edge, collision

    def select_closest_node(self, subgoal, to_goal=False, print_distance=False):
        if not to_goal:
            return super().select_closest_node(subgoal, print_distance)

        d_batch = self.calc_distance_batch(subgoal)
        i_min = np.argsort(d_batch)
        for i in i_min:
            node = self.get_node_from_id(i)
            if node.to_goal:
                return node
        return None

    def iterate(self, max_collision_retries=5000):
        # Sanity checks
        q_root_full = self.map_qa_to_q(self.root_node.q)
        q_goal_full = self.map_qa_to_q(self.params.goal)
        root_in_collision = self.is_collision(q_root_full)
        goal_in_collision = self.is_collision(q_goal_full)
        dist_to_goal = np.linalg.norm(self.params.goal - self.root_node.q)

        print(f"  Root in collision: {root_in_collision}")
        print(f"  Goal in collision: {goal_in_collision}")
        print(f"  Distance root->goal: {dist_to_goal:.4f}")
        print(f"  Stepsize: {self.params.stepsize:.4f}")
        print(f"  Termination tolerance: {self.params.termination_tolerance:.6f}")

        if root_in_collision:
            print("  WARNING: root config is in collision!")
        if goal_in_collision:
            print("  WARNING: goal config is in collision! RRT will likely fail.")

        pbar = tqdm(total=self.params.max_size)
        collision_streak = 0
        total_collision_attempts = 0
        goal_attempts = 0
        goal_collisions = 0

        while self.size < self.params.max_size:
            pbar.update(1)

            collision = True
            attempts = 0
            while collision and attempts < max_collision_retries:
                attempts += 1
                total_collision_attempts += 1
                to_goal = False
                if self.cointoss_for_goal():
                    subgoal = self.params.goal
                    to_goal = True
                    goal_attempts += 1
                else:
                    subgoal = self.sample_subgoal()

                parent_node = self.select_closest_node(subgoal, to_goal)
                if parent_node is None:
                    continue

                child_node, edge, collision = self.extend_towards_q(
                    parent_node, subgoal
                )

                if to_goal:
                    parent_node.to_goal = False
                    if collision:
                        goal_collisions += 1

            if collision:
                collision_streak += 1
                if collision_streak % 100 == 0:
                    closest_dist = np.sqrt(np.min(self.calc_distance_batch(self.params.goal)))
                    print(f"\n  Stuck: {collision_streak} consecutive failed nodes, "
                          f"closest to goal: {closest_dist:.4f}, "
                          f"tree size: {self.size}, "
                          f"goal attempts: {goal_attempts} ({goal_collisions} collided)")
                continue

            collision_streak = 0
            self.add_node(child_node)
            child_node.value = parent_node.value + edge.cost
            self.add_edge(edge)

            if self.size % 500 == 0:
                closest_dist = np.sqrt(np.min(self.calc_distance_batch(self.params.goal)))
                pbar.set_postfix(closest=f"{closest_dist:.4f}", nodes=self.size)

            if self.is_close_to_goal():
                print(f"\ndone (tree size: {self.size}, "
                      f"total extension attempts: {total_collision_attempts})")
                break

        if self.size >= self.params.max_size:
            closest_dist = np.sqrt(np.min(self.calc_distance_batch(self.params.goal)))
            print(f"\n  FAILED: max tree size reached. "
                  f"Closest to goal: {closest_dist:.4f}")

        pbar.close()

    def get_final_path_qa(self):
        q_final = self.select_closest_node(self.params.goal)
        path = nx.shortest_path(
            self.graph, source=self.root_node.id, target=q_final.id
        )
        return np.array([self.get_node_from_id(p).q for p in path])

    def get_final_path_q(self):
        qa_path = self.get_final_path_qa()
        q_path = np.zeros((len(qa_path), self.dim_x))
        q_path[:, self.idx_q_a] = qa_path
        q_path[:, self.idx_q_u] = self.qu
        return q_path

    def segment_has_no_collision(self, q_start, q_end, T):
        q_trj = np.linspace(q_start, q_end, T)
        for t in range(len(q_trj)):
            if self.is_collision(self.map_qa_to_q(q_trj[t])):
                return False
        return True

    def shortcut_path(self, x_trj, num_tries=1000):
        x_trj_s = np.copy(x_trj)
        for _ in range(num_tries):
            if len(x_trj_s) < 3:
                break
            ind_a, ind_b = np.sort(np.random.choice(len(x_trj_s), 2, replace=False))
            x_a = x_trj_s[ind_a, self.idx_q_a]
            x_b = x_trj_s[ind_b, self.idx_q_a]

            if self.segment_has_no_collision(x_a, x_b, 100):
                shortcut_length = np.linalg.norm(x_b - x_a)
                n_steps = max(2, int(np.ceil(shortcut_length / self.params.stepsize)) + 1)
                shortcut_qa = np.linspace(x_a, x_b, n_steps)
                shortcut_q = np.zeros((n_steps, self.dim_x))
                shortcut_q[:, self.idx_q_a] = shortcut_qa
                shortcut_q[:, self.idx_q_u] = self.qu

                x_trj_s = np.concatenate([
                    x_trj_s[:ind_a],
                    shortcut_q,
                    x_trj_s[ind_b + 1:]
                ])

        return x_trj_s


def step_out(setup, x, scale=0.06):
    """Step all arms away from contact using Jacobian-based control."""
    q_sim_py = setup.q_sim_py
    plant = setup.plant
    q_sim_py.update_mbp_positions_from_vector(x)

    sg = q_sim_py.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
    collision_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(scale)
    inspector = query_object.inspector()

    actuated_models = list(setup.q_sim.get_actuated_models())
    idx_q_a = setup.idx_q_a
    n_arms = len(actuated_models)
    joints_per_arm = len(idx_q_a) // n_arms

    arm_infos = {}
    for i, model in enumerate(actuated_models):
        start = i * joints_per_arm
        end = (i + 1) * joints_per_arm
        arm_infos[model] = idx_q_a[start:end]

    qnext = np.copy(x)

    for model, arm_joints in arm_infos.items():
        min_dist = np.inf
        min_body = None
        min_normal = None

        for collision in collision_pairs:
            f_id = inspector.GetFrameId(collision.id_A)
            body_A = plant.GetBodyFromFrameId(f_id)
            f_id = inspector.GetFrameId(collision.id_B)
            body_B = plant.GetBodyFromFrameId(f_id)

            if body_A.model_instance() == model or body_B.model_instance() == model:
                if collision.distance < min_dist:
                    min_dist = collision.distance
                    min_body = body_A if body_A.model_instance() == model else body_B
                    normal_sign = 1 if body_A.model_instance() == model else -1
                    min_normal = normal_sign * collision.nhat_BA_W

        if min_body:
            J = plant.CalcJacobianTranslationalVelocity(
                q_sim_py.context_plant,
                JacobianWrtVariable.kV,
                min_body.body_frame(),
                np.array([0, 0, 0]),
                plant.world_frame(),
                plant.world_frame(),
            )
            J_a = J[:2, arm_joints]
            qdot = np.linalg.pinv(J_a).dot((scale - min_dist) * min_normal[:2])
            qnext[arm_joints] += qdot

    q_sim_py.update_mbp_positions_from_vector(qnext)

    sg = q_sim_py.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
    collision_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(scale)
    print("colls:")
    for model, arm_joints in arm_infos.items():
        for collision in collision_pairs:
            f_id = inspector.GetFrameId(collision.id_A)
            body_A = plant.GetBodyFromFrameId(f_id)
            f_id = inspector.GetFrameId(collision.id_B)
            body_B = plant.GetBodyFromFrameId(f_id)

            if body_A.model_instance() == model or body_B.model_instance() == model:
                print(collision.distance)

    return qnext


def main(
    setup,
    traj_file_path: str,
    step_out_distance: float = 0.02,
    step_in_frames: int = 5,
    contact_buffer: float = 0.0,
    max_rrt_size: int = 200000,
    shortcut_tries: int = 1000,
):
    data = np.load(traj_file_path)
    q_knots_trimmed = data["q_trj"]
    u_knots_trimmed = data["u_trj"]
    h_input = float(data["h"]) if "h" in data.files else setup.rrt_params.h

    if len(u_knots_trimmed) == len(q_knots_trimmed):
        u_knots_trimmed = u_knots_trimmed[:-1]
    elif len(q_knots_trimmed) != len(u_knots_trimmed) + 1:
        raise ValueError(
            f"Unexpected q/u length: {len(q_knots_trimmed)} vs {len(u_knots_trimmed)}"
        )

    segments = IrsRrt.get_regrasp_segments(u_knots_trimmed)
    print(f"Found {len(segments)} contact segments: {segments}")

    idx_q_a = setup.idx_q_a
    idx_q_u = setup.idx_q_u
    dim_x = setup.dim_x
    q_sim_py = setup.q_sim_py
    plant = setup.plant
    joint_limits = setup.joint_limits_ur5e

    actuated_models = list(setup.q_sim.get_actuated_models())
    n_arms = len(actuated_models)
    joints_per_arm = len(idx_q_a) // n_arms

    # Tile joint limits for multi-arm
    if n_arms > 1:
        joint_limits = np.tile(joint_limits, (n_arms, 1))

    q_knots_ref_list = [[setup.q0]]
    u_knots_ref_list = [[]]

    for t_start, t_end in segments:
        q_knots_ref_list.append(q_knots_trimmed[t_start:t_end + 1])
        u_knots_ref_list.append(u_knots_trimmed[t_start:t_end])

    q_knots_patched_list = []
    u_knots_patched_list = []

    for n in range(len(q_knots_ref_list) - 1):
        q_start = q_knots_ref_list[n][-1]
        q_end = q_knots_ref_list[n + 1][0]

        qu_start = q_start[idx_q_u]
        qu_end = q_end[idx_q_u]

        assert np.allclose(qu_start, qu_end, atol=0.005), (
            f"Object pose changed between segments {n} and {n+1}"
        )

        qin = step_out(setup, q_start, scale=step_out_distance)
        qout = step_out(setup, q_end, scale=step_out_distance)

        setup.q_vis.draw_configuration(qin)
        input()
        setup.q_vis.draw_configuration(qout)
        input()

        qa_in = qin[idx_q_a]
        qa_out = qout[idx_q_a]

        cf_params = RrtParams()
        cf_params.goal = qa_out
        cf_params.root_node = CFNode(qa_in)
        cf_params.termination_tolerance = 1e-3
        cf_params.goal_as_subgoal_prob = 0.1
        cf_params.stepsize = 0.5 * h_input
        cf_params.max_size = max_rrt_size

        cf_rrt = CollisionFreeRRT(
            cf_params, qu_start, idx_q_a, idx_q_u, dim_x,
            actuated_models, q_sim_py, plant, joint_limits, contact_buffer
        )

        print(f"Segment {n}->{n+1}: planning collision-free path...")
        cf_rrt.iterate()
        cf_trj = cf_rrt.shortcut_path(
            cf_rrt.get_final_path_q(), num_tries=shortcut_tries
        )

        # Build patch: step out -> collision-free path -> step in
        patch_trj = np.zeros((0, dim_x))

        if step_in_frames > 0:
            patch_trj = np.vstack([
                patch_trj, np.linspace(q_start, qin, step_in_frames)
            ])

        patch_trj = np.vstack([patch_trj, cf_trj])

        if step_in_frames > 0:
            patch_trj = np.vstack([
                patch_trj, np.linspace(qout, q_end, step_in_frames)
            ])

        q_knots_patched_list.append(q_knots_ref_list[n])
        q_knots_patched_list.append(patch_trj[1:-1])

        if len(u_knots_ref_list[n]) > 0:
            u_knots_patched_list.append(u_knots_ref_list[n])
        u_knots_patched_list.append(patch_trj[1:, idx_q_a])

    q_knots_patched_list.append(q_knots_ref_list[-1])
    u_knots_patched_list.append(u_knots_ref_list[-1])

    q_trj_final = np.concatenate(q_knots_patched_list)
    u_trj_final = np.concatenate(u_knots_patched_list)

    assert len(q_trj_final) == len(u_trj_final) + 1, (
        f"Expected len(q) == len(u) + 1, got {len(q_trj_final)} vs {len(u_trj_final)}"
    )

    setup.q_vis.publish_trajectory(q_trj_final, h_input)

    src_name = os.path.basename(traj_file_path)
    ts = src_name[len("traj_"):-len(".npz")]
    out_path = os.path.join(setup.data_folder, f"traj_full_{ts}.npz")

    out = {k: data[k] for k in data.files}
    out["q_trj"] = q_trj_final
    out["u_trj"] = u_trj_final
    out["h"] = h_input
    out["q_u_indices_into_x"] = idx_q_u
    out["q_a_indices_into_x"] = idx_q_a

    np.savez_compressed(out_path, **out)
    print(f"Saved full trajectory to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file_path", type=str)
    parser.add_argument("--step_out_distance", type=float, default=0.02)
    parser.add_argument("--step_in_frames", type=int, default=5)
    parser.add_argument("--contact_buffer", type=float, default=0.0)
    parser.add_argument("--max_rrt_size", type=int, default=200000)
    parser.add_argument("--shortcut_tries", type=int, default=1000)
    parser.add_argument("--prepend_q0", action="store_true")
    args = parser.parse_args()

    data = np.load(args.traj_file_path)
    task_name = str(data["task_name"])
    data.close()

    setup = deduce_setup(task_name)
    main(
        setup,
        args.traj_file_path,
        step_out_distance=args.step_out_distance,
        step_in_frames=args.step_in_frames,
        contact_buffer=args.contact_buffer,
        max_rrt_size=args.max_rrt_size,
        shortcut_tries=args.shortcut_tries,
    )
