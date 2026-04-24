import numpy as np
from pydrake.solvers import GurobiSolver
from pydrake.solvers import MathematicalProgram

from pydrake.all import Quaternion, RollPitchYaw, AngleAxis, RigidTransform, JacobianWrtVariable

from qsim.simulator import QuasistaticSimulator
from qsim_cpp import QuasistaticSimulatorCpp

from irs_rrt.contact_sampler import ContactSampler
from irs_rrt.irs_rrt import IrsRrt, IrsNode, IrsEdge
from irs_rrt.rrt_base import Node
from irs_rrt.rrt_params import DuStarMode, DistanceMetric, IrsRrtTrajectoryParams

from scripts import utils

# For prettier tqdm bar in jupyter notebooks.
from tqdm import tqdm

if "get_ipython" in locals() or "get_ipython" in globals():
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        print("Running in a jupyter notebook!")
        from tqdm.notebook import tqdm


class IrsTrajectoryNode(IrsNode):
    """
    IrsNode. Each node is responsible for keeping a copy of the bundled dynamics
    and the Gaussian parametrized by the bundled dynamics.
    """

    def __init__(self, q: np.array):
        super().__init__(q)

        # Contact mode id for grouping
        self.contact_mode_id = None

        # Closest obj_pose(t) to q
        self.t = None
        self.distance_from_traj = None
        self.is_static = False

        self.is_active = True
        self.extensions_since_regrasp = 0

class IrsRrtTrajectory(IrsRrt):
    def __init__(
        self,
        rrt_params: IrsRrtTrajectoryParams,
        contact_sampler: ContactSampler,
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator,
        get_obj_pose_from_t,
        q_sim_smooth: QuasistaticSimulatorCpp | None = None,
    ):
        self.get_obj_pose_from_t = get_obj_pose_from_t

        self.contact_sampler = contact_sampler
        super().__init__(rrt_params, q_sim, q_sim_py, q_sim_smooth)
        self.solver = GurobiSolver()
        self.root_node.contact_mode_id = 0
        self.root_node.is_static = self.is_static(self.root_node.q)
        self.root_node.t, self.root_node.distance_from_traj = self.closest_t_in_trajectory(self.root_node.q[self.q_u_indices_into_x])
        self.next_contact_id = 1
        self.subgoal_ts = rrt_params.subgoal_ts
        self.min_t = 0.0
        self.curr_goal_i = 1
        self.q_curr_goal = np.zeros(self.dim_x)
        self.q_curr_goal[self.q_u_indices_into_x] = get_obj_pose_from_t(self.subgoal_ts[self.curr_goal_i])

        self.max_static_t = 0

        self.idx_u = self.py_plant.GetModelInstanceByName("box")

        # Detect arms dynamically from actuated models
        actuated_models = list(q_sim.get_actuated_models())
        n_arms = len(actuated_models)
        joints_per_arm = len(self.q_a_indices_into_x) // n_arms

        self.arms = []
        for i, model in enumerate(actuated_models):
            model_name = self.py_plant.GetModelInstanceName(model)
            start_idx = i * joints_per_arm
            end_idx = (i + 1) * joints_per_arm
            arm_pose = rrt_params.arm_poses.get(
                model_name, np.array([1, 0, 0, 0, 0, 0, 0])
            )
            self.arms.append({
                'model_idx': model,
                'joint_indices': self.q_a_indices_into_x[start_idx:end_idx],
                'arm_pose': arm_pose,
            })

        # Backwards compat for bimanual code
        if n_arms >= 2:
            self.idx_a_l = self.arms[0]['model_idx']
            self.idx_a_r = self.arms[1]['model_idx']
            self.idx_q_a_l = self.arms[0]['joint_indices']
            self.idx_q_a_r = self.arms[1]['joint_indices']

    def sample_subgoal(self):
        """
        Sample a subgoal from the configuration space.
        """
        # sample robots (doesn't matter)
        subgoal = np.random.rand(self.dim_x)
        subgoal = self.q_lb + (self.q_ub - self.q_lb) * subgoal

        # sample obj pose
        t0 = max(self.min_t, self.subgoal_ts[self.curr_goal_i - 1])
        t1 = self.subgoal_ts[self.curr_goal_i]
        t = np.random.rand() * (t1 - t0) + t0

        obj_pose_t = self.get_obj_pose_from_t(t)
        obj_quat_t = Quaternion(obj_pose_t[:4])
        obj_trans_t = obj_pose_t[4:]

        quat_noise = RollPitchYaw(subgoal[self.q_u_indices_into_x[:3]]).ToQuaternion()

        obj_quat = obj_quat_t.multiply(quat_noise)

        trans_noise = subgoal[self.q_u_indices_into_x[4:]]
        obj_trans = obj_trans_t + trans_noise

        obj_pose = np.concatenate((obj_quat.wxyz(), obj_trans))

        subgoal[self.q_u_indices_into_x] = obj_pose

        return subgoal, t
    
    def calc_distance(self, query, node):
        # 1 x n
        mu_batch = node.chat_u[None, :]
        # 1 x n x n
        covinv_tensor = node.covinv_u[None, :, :]
        error_batch = query - mu_batch
        int_batch = np.einsum("Bij,Bi -> Bj", covinv_tensor, error_batch)
        metric_batch = np.einsum("Bi,Bi -> B", int_batch, error_batch)

        return metric_batch[0]


    def select_closest_nodes(
        self,
        subgoal: np.array,
        max_t: float,
        k_closest: int,
        d_threshold: float = np.inf,
        print_distance: bool = False,
    ):
        """
        Given a subgoal, this function finds the node that is closest from the
         subgoal.
        None is returned if the distances of all nodes are greater than
         d_treshold.
        """
        d_batch = self.calc_distance_batch(subgoal)

        i_min = np.argsort(d_batch)
        i_min = i_min[d_batch[i_min] < d_threshold]

        selected_nodes = []
        selected_contact_modes = set()
        i = 0
        while i < len(i_min) and len(selected_nodes) < k_closest:
            node = self.get_node_from_id(i_min[i])
            i += 1
            if self.rrt_params.connect_from_behind and node.t > max_t:
                continue
            if node.contact_mode_id not in selected_contact_modes and node.is_active:
                selected_nodes.append(node)
                selected_contact_modes.add(node.contact_mode_id)

        if print_distance:
            print("closest distances to subgoal", d_batch[i_min])

        return selected_nodes
    
    
    def select_closest_static_nodes(
        self,
        subgoal: np.array,
        max_t: float,
        k_closest: int,
        d_threshold: float = np.inf,
        print_distance: bool = False,
    ):
        """
        Given a subgoal, this function finds the nodes that are closest to the
         subgoal using 8 corner difference.
        """
        if self.rrt_params.static_distance_metric == DistanceMetric.Corner:
            pose_batch = self.get_q_matrix_up_to()[:, self.q_u_indices_into_x]
            subgoal_pose = subgoal[self.q_u_indices_into_x]
            d_batch = self.calc_distance_batch_corners(subgoal_pose, pose_batch)
        elif self.rrt_params.static_distance_metric == DistanceMetric.Mahalabonis:
            d_batch = self.calc_distance_batch(subgoal)
        else:
            raise Exception(f"Invalid Distance Metric {self.rrt_params.static_distance_metric}")

        
        i_min = np.argsort(d_batch)
        i_min = i_min[d_batch[i_min] < d_threshold]

        selected_nodes = []
        i = 0
        while i < len(i_min) and len(selected_nodes) < k_closest:
            node = self.get_node_from_id(i_min[i])
            i += 1
            if self.rrt_params.connect_from_behind and node.t > max_t:
                continue
            if node.is_static and node.is_active and node.extensions_since_regrasp >= self.rrt_params.regrasp_cooldown:
                if print_distance:
                    print("closest distance to subgoal", d_batch[i_min[i]])
                selected_nodes.append(node)

        return selected_nodes
    
    def closest_t_in_trajectory(self, obj_pose):
        ts = np.linspace(0,1,101)
        poses_t = np.array([self.get_obj_pose_from_t(t) for t in ts])

        dists = self.calc_distance_batch_corners(obj_pose, poses_t)

        i_min = np.argmin(dists)

        return ts[i_min], dists[i_min]
    
    def dist_to_q(self, q):
        pose_batch = self.get_q_matrix_up_to()[:, self.q_u_indices_into_x]
        q_pose = q[self.q_u_indices_into_x]
        return np.min(self.calc_distance_batch_corners(q_pose, pose_batch))

    def find_node_closest_to_goal(self):
        """
        Override the base-class Mahalanobis metric with the 8-corner
        distance metric used by the termination check in `iterate`, so that
        the "closest node" seen by the saved trajectory agrees with the
        node that actually triggered termination.

        Without this override, `select_closest_node` uses the local
        Mahalanobis metric (rrt_params.distance_metric = "local_u"), which
        can pick a different node than the one that minimizes the corner
        distance — leaving the final saved state further from the goal
        than the termination_tolerance would suggest.
        """
        pose_batch = self.get_q_matrix_up_to()[:, self.q_u_indices_into_x]
        q_pose = self.rrt_params.goal[self.q_u_indices_into_x]
        d_batch = self.calc_distance_batch_corners(q_pose, pose_batch)
        i_min = int(np.argmin(d_batch))
        print(
            f"find_node_closest_to_goal: corner dist = {d_batch[i_min]:.5f} "
            f"(node {i_min})"
        )
        return self.get_node_from_id(i_min)

    def get_trimmed_q_and_u_knots_to_goal(self):
        """
        Override the base-class version with two changes:
          1. Final-node selection uses the 8-corner distance metric (via
             our overridden `find_node_closest_to_goal`) so it agrees with
             the termination check in `iterate`.
          2. Appends one extra "go-to-goal" action at the end and the
             corresponding post-state, so `len(q_knots) == len(u_knots) + 1`.
             The extra action is computed at the goal-closest node via the
             bundled gradient dynamics + configured du* mode, with the target
             set to the RRT goal, so any residual offset between the final
             node and the goal is corrected on the last step. The post-state
             is computed by forward-simulating that action so consumers do
             not need to reconstruct it.
        """
        final_node = self.find_node_closest_to_goal()
        node_idx_path = np.array(self.trace_nodes_to_root_from(final_node.id))

        q_knots = self.q_matrix[node_idx_path]
        u_knots = self.get_u_knots_from_node_idx_path(node_idx_path)

        node_idx_path_to_keep = self.trim_regrasps(u_knots)
        q_knots_trimmed = q_knots[node_idx_path_to_keep]
        u_knots_trimmed = u_knots[node_idx_path_to_keep[1:]]

        # Append a final go-to-goal action and its post-state. After this
        # block, len(q_knots_trimmed) == len(u_knots_trimmed) + 1, which is
        # the natural MPC convention. Using the configured du* mode keeps
        # this consistent with the rest of the planned trajectory's
        # extension steps.
        du_star = self.calc_du_star_towards_q(final_node, self.rrt_params.goal)
        u_star = final_node.ubar + du_star
        u_knots_trimmed = np.concatenate(
            [u_knots_trimmed, u_star[None, :]], axis=0
        )
        # Forward-simulate u_star to get the post-state and append it.
        q_post_star = self.q_sim.calc_dynamics(
            q_knots_trimmed[-1], u_star, self.sim_params
        )
        q_knots_trimmed = np.concatenate(
            [q_knots_trimmed, q_post_star[None, :]], axis=0
        )

        return q_knots_trimmed, u_knots_trimmed

    def cast_to_cone(self, joints, normal, arm_pose, deg):
        rad = np.deg2rad(deg)
        eef_offset = self.rrt_params.eef_offset

        ee_pose = utils.get_ee_pose(joints, arm_pose, eef_offset=eef_offset)
        ee_dir = utils.quat_apply(ee_pose[:4], np.array([0,0,1]))
        dot = np.dot(ee_dir, -normal)
        theta = np.arccos(dot)

        if theta <= rad:
            return joints

        tilt_axis = np.cross(-normal, ee_dir)
        tilt_axis /= np.linalg.norm(tilt_axis)
        ee_quat = Quaternion(ee_pose[:4])

        d_theta = rad - theta
        q_correction = AngleAxis(d_theta, tilt_axis).quaternion()
        ee_quat_new: Quaternion = q_correction.multiply(ee_quat)

        ee_pose_new = np.concatenate([ee_quat_new.wxyz(), ee_pose[4:]])

        # TODO: get all configurations and filter
        joints_new = utils.get_joints(ee_pose_new, arm_pose, joints, eef_offset=eef_offset)

        return joints_new
    
    def step_in(self, q):
        """
        Given a near-contact configuration, give a q that steps in contact.
        Works with any number of arms.
        """
        self.q_sim_py.update_mbp_positions_from_vector(q)

        sg = self.q_sim_py.get_scene_graph()
        query_object = sg.GetOutputPort("query").Eval(self.q_sim_py.context_sg)
        collision_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(
            0.02
        )

        inspector = query_object.inspector()

        q_next = np.copy(q)

        for arm in self.arms:
            model_idx = arm['model_idx']
            joint_indices = arm['joint_indices']
            arm_pose = arm['arm_pose']

            # Find closest collision between this arm and the box
            min_dist = np.inf
            min_body = None
            min_normal = None

            for collision in collision_pairs:
                f_id = inspector.GetFrameId(collision.id_A)
                body_A = self.py_plant.GetBodyFromFrameId(f_id)
                f_id = inspector.GetFrameId(collision.id_B)
                body_B = self.py_plant.GetBodyFromFrameId(f_id)

                # Only collisions involving the box
                if body_A.model_instance() != self.idx_u and body_B.model_instance() != self.idx_u:
                    continue

                # Only collisions involving this arm
                if body_A.model_instance() != model_idx and body_B.model_instance() != model_idx:
                    continue

                if collision.distance < min_dist:
                    min_dist = collision.distance
                    min_body = body_A if body_A.model_instance() == model_idx else body_B
                    # nhat_BA_W points from B to A; we want normal pointing from box toward arm
                    normal_sign = 1 if body_A.model_instance() == model_idx else -1
                    min_normal = normal_sign * collision.nhat_BA_W

            if min_body:
                J = self.py_plant.CalcJacobianTranslationalVelocity(
                    self.q_sim_py.context_plant,
                    JacobianWrtVariable.kV,
                    min_body.body_frame(),
                    np.array([0, 0, 0]),
                    self.py_plant.world_frame(),
                    self.py_plant.world_frame(),
                )

                J_a = J[:2, joint_indices]
                qdot = np.linalg.pinv(J_a).dot(min_dist * -min_normal[:2])

                q_next[joint_indices] += qdot
                # q_next[joint_indices] = self.cast_to_cone(
                #     q_next[joint_indices], min_normal, arm_pose, 30
                # )

        return q_next
    
    def iterate(self):
        """
        Main method for iteration.
        """
        pbar = tqdm(total=self.max_size)

        time_to_dist_to_goal = []

        if self.root_node.is_static:
            initial_nodes = []
            initial_edges = []

            for _ in range(self.rrt_params.initial_contact_samples):
                try:
                    child_node, edge = self.sample_contact(self.root_node)
                    initial_nodes.append(child_node)
                    initial_edges.append(edge)
                except RuntimeError:
                    continue
        
            initial_nodes = np.array(initial_nodes)
            initial_edges = np.array(initial_edges)

            is_valid = self.add_nodes(initial_nodes, draw_first_node=True)
            initial_nodes = initial_nodes[is_valid]
            initial_edges = initial_edges[is_valid]

            for node, edge in zip(initial_nodes, initial_edges):
                node.value = edge.parent.value + edge.cost
                self.add_edge(edge)

            pbar.update(len(initial_nodes))

        while self.size < self.rrt_params.max_size:
            # 1. Sample a subgoal.
            if self.cointoss_for_goal():
                subgoal = self.q_curr_goal
                subgoal_t = self.subgoal_ts[self.curr_goal_i]
            else:
                subgoal, subgoal_t = self.sample_subgoal()

            print(f"Sampling in range [{self.subgoal_ts[self.curr_goal_i-1]};{self.subgoal_ts[self.curr_goal_i]}] -> {subgoal_t}")

            regrasp = np.random.rand() < self.rrt_params.grasp_prob

            # 2. Sample closest nodes to subgoal
            if regrasp:
                parent_nodes = self.select_closest_static_nodes(
                    subgoal, subgoal_t, self.rrt_params.batch_size, d_threshold=self.rrt_params.distance_threshold
                )
            else:
                parent_nodes = self.select_closest_nodes(
                    subgoal, subgoal_t, self.rrt_params.batch_size, d_threshold=self.rrt_params.distance_threshold
                )

            # update progress only if a valid parent_node is chosen.
            if len(parent_nodes) == 0:
                continue

            child_nodes = []
            edges = []
            for parent_node in parent_nodes:
                # 3. Extend to subgoal.
                try:
                    if regrasp:
                        child_node, edge = self.sample_contact(parent_node)
                    else:
                        child_node, edge = self.extend_towards_q(parent_node, subgoal)
                        if child_node.is_static:
                            self.max_static_t = max(self.max_static_t, child_node.t)
                except RuntimeError as e:
                    print(e)
                    continue

                if self.rrt_params.connect_to_front and child_node.t < parent_node.t:
                    continue

                child_nodes.append(child_node)
                edges.append(edge)

            child_nodes = np.array(child_nodes)
            edges = np.array(edges)
            
            if self.size + len(child_nodes) > self.max_size:
                nodes_to_keep = self.max_size - self.size
                child_nodes = child_nodes[:nodes_to_keep]
                edges = edges[:nodes_to_keep]

            # 4. Register the new node to the graph.
            try:
                is_valid = self.add_nodes(child_nodes, draw_first_node=True)
                child_nodes = child_nodes[is_valid]
                edges = edges[is_valid]
            except RuntimeError as e:
                print(e)
                continue

            print(f"max static t: {self.max_static_t}")

            for child_node, edge in zip(child_nodes, edges):
                child_node.value = edge.parent.value + edge.cost
                self.add_edge(edge)

            pbar.update(len(child_nodes))

            # 5. Check for termination.
            dist_to_goal = self.dist_to_q(self.goal)
            dist_to_curr_goal = self.dist_to_q(self.q_curr_goal)

            time_to_dist_to_goal.append([(pbar.last_print_t - pbar.start_t), dist_to_goal])

            print(dist_to_curr_goal, dist_to_goal)
            if dist_to_curr_goal < self.rrt_params.subgoal_tolerance and self.curr_goal_i < len(self.subgoal_ts) - 1:
                self.curr_goal_i += 1
                self.q_curr_goal = np.zeros(self.dim_x)
                self.q_curr_goal[self.q_u_indices_into_x] = self.get_obj_pose_from_t(self.subgoal_ts[self.curr_goal_i])
                print("FOUND A PATH TO SUBGOAL!!!!!")
        
            if dist_to_goal < self.rrt_params.termination_tolerance:
                self.goal_node_idx = child_node.id
                print("FOUND A PATH TO GOAL!!!!!")
                break

        pbar.close()

        return time_to_dist_to_goal

    def calc_du_star_towards_q_qp(self, parent_node: Node, q: np.ndarray):
        prog = MathematicalProgram()
        n_a = self.q_dynamics.dim_u
        du = prog.NewContinuousVariables(n_a)
        idx_obj = self.q_sim.get_q_u_indices_into_q()
        idx_robot = self.q_sim.get_q_a_indices_into_q()
        q_a_lb = self.q_lb[idx_robot]
        q_a_ub = self.q_ub[idx_robot]
        B_obj = parent_node.Bhat[idx_obj, :]

        Q = B_obj.T @ B_obj + 1e-2 * np.eye(n_a)
        b = (q - parent_node.chat)[idx_obj]
        b_combined = -B_obj.T @ b
        prog.AddQuadraticCost(Q, b_combined, du)
        prog.AddBoundingBoxConstraint(
            -self.rrt_params.stepsize, self.rrt_params.stepsize, du
        )
        prog.AddBoundingBoxConstraint(
            q_a_lb - parent_node.ubar, q_a_ub - parent_node.ubar, du
        )

        result = self.solver.Solve(prog)
        if not result.is_success():
            raise RuntimeError

        du_star = result.GetSolution(du)
        return du_star

    def calc_du_star_towards_q_lstsq(self, parent_node: Node, q: np.ndarray):
        # Compute least-squares solution.
        # NOTE(terry-suh): it is important to only do this on the submatrix
        # of B that has to do with u.

        idx_obj = self.q_sim.get_q_u_indices_into_q()

        du_star = np.linalg.lstsq(
            parent_node.Bhat[idx_obj, :],
            (q - parent_node.chat)[idx_obj],
            rcond=None,
        )[0]

        du_norm = np.linalg.norm(du_star)
        step_size = min(du_norm, self.rrt_params.stepsize)
        du_star = du_star / du_norm
        u_star = parent_node.ubar + step_size * du_star

        if self.rrt_params.enforce_robot_joint_limits:
            idx_robot = self.q_sim.get_q_a_indices_into_q()
            q_a_lb = self.q_lb[idx_robot]
            q_a_ub = self.q_ub[idx_robot]
            u_star = np.clip(u_star, q_a_lb, q_a_ub)

        return u_star - parent_node.ubar
    
    def calc_du_star_towards_q_diff(self, parent_node: Node, q: np.ndarray):
        idxs_a = self.q_sim.get_q_a_indices_into_q()
        du_star = q[idxs_a] - parent_node.q[idxs_a]

        du_norm = np.linalg.norm(du_star)
        step_size = min(du_norm, self.rrt_params.stepsize)
        du_star = du_star * step_size / du_norm

        return du_star

    def build_stickiness_block(self, parent_node: Node, A_box, lam_rotate=1.0, lam_slide=1.0):
        """
        Build stickiness rows that penalize relative EE-box twist.
        A_box: (6, n_a) tangent-space box Jacobian [angular; translational].
        lam_rotate: scale for the angular (first 3) rows.
        lam_slide: scale for the translational (last 3) rows.
        Returns (A_stick, b_stick) with shape (6*n_arms, n_a) and (6*n_arms,).
        """
        self.q_sim_py.update_mbp_positions_from_vector(parent_node.q)
        ctx = self.q_sim_py.context_plant

        A_rows = []
        b_rows = []
        n_a = len(self.q_a_indices_into_x)
        ee_name = self.rrt_params.ee_body_name
        scale = np.array([lam_rotate]*3 + [lam_slide]*3)

        for arm in self.arms:
            model_idx = arm['model_idx']
            joint_indices = arm['joint_indices']

            ee_body = self.py_plant.GetBodyByName(ee_name, model_idx)
            J_ee_full = self.py_plant.CalcJacobianSpatialVelocity(
                ctx,
                JacobianWrtVariable.kV,
                ee_body.body_frame(),
                np.zeros(3),
                self.py_plant.world_frame(),
                self.py_plant.world_frame(),
            )
            # Build a (6, n_a) matrix with only this arm's columns nonzero
            J_ee_a = np.zeros((6, n_a))
            J_ee_a[:, joint_indices - self.q_a_indices_into_x[0]] = J_ee_full[:, joint_indices]

            rel_J = (J_ee_a - A_box) * scale[:, None]
            A_rows.append(rel_J)
            b_rows.append(np.zeros(6))

        return np.vstack(A_rows), np.concatenate(b_rows)

    def bhat_to_tangent(self, Bhat, chat):
        """Convert Bhat unactuated rows from 7D quat-space to 6D tangent-space."""
        idx_u = self.q_u_indices_into_x
        w, x, y, z = chat[idx_u[:4]]
        ET = 2.0 * np.array([
            [-x,  w,  z, -y],
            [-y, -z,  w,  x],
            [-z,  y, -x,  w],
        ])
        J_ang = ET @ Bhat[idx_u[:4], :]
        J_pos = Bhat[idx_u[4:], :]
        return np.vstack([J_ang, J_pos])

    def calc_du_star_towards_q_constrained_lstsq(self, parent_node: Node, q: np.ndarray):
        """
        Solve min ||Ax - b|| subject to ||Wx|| <= d,
        where W = diag(du_weights) penalizes specific joints.
        Optionally augments with stickiness rows.
        """
        tol = 1e-10
        max_iter = 100

        A = parent_node.Bhat_u
        b = (q[self.q_u_indices_into_x] - parent_node.chat_u)

        lam_ang = self.rrt_params.stickiness_scale_angular
        lam_lin = self.rrt_params.stickiness_scale_linear
        if lam_ang > 0 or lam_lin > 0:
            A_box_tangent = self.bhat_to_tangent(parent_node.Bhat, parent_node.chat)
            residual = np.linalg.norm(b)
            A_stick, b_stick = self.build_stickiness_block(
                parent_node, A_box_tangent,
                lam_slide=lam_lin * residual,
                lam_rotate=lam_ang * residual,
            )
            A = np.vstack([A, A_stick])
            b = np.concatenate([b, b_stick])

        n_a = A.shape[1]
        AtA = A.T @ A
        Atb = A.T @ b

        try:
            x0 = np.linalg.solve(AtA, Atb)
            if np.linalg.norm(x0) <= self.rrt_params.stepsize:
                return x0
        except:
            pass

        lam_low, lam_high = 0, 1.0
        I = np.eye(n_a)

        for _ in range(50):
            x = np.linalg.solve(AtA + lam_high * I, Atb)
            if np.linalg.norm(x) < self.rrt_params.stepsize:
                break
            lam_high *= 2

        for _ in range(max_iter):
            lam = 0.5 * (lam_low + lam_high)
            x = np.linalg.solve(AtA + lam * I, Atb)

            if np.linalg.norm(x) > self.rrt_params.stepsize:
                lam_low = lam
            else:
                lam_high = lam

            if lam_high - lam_low < tol:
                break

        return x
    
    def calc_du_star_towards_q(self, parent_node: Node, q: np.ndarray):
        if self.rrt_params.du_star_mode == DuStarMode.EEFDiff:
            du_star = self.calc_du_star_towards_q_diff(parent_node, q)
        elif self.rrt_params.du_star_mode == DuStarMode.LSTSQ:
            du_star = self.calc_du_star_towards_q_lstsq(parent_node, q)
        elif self.rrt_params.du_star_mode == DuStarMode.ConstrainedLSTSQ:
            du_star = self.calc_du_star_towards_q_constrained_lstsq(parent_node, q)
        elif self.rrt_params.du_star_mode == DuStarMode.QP:
            du_star = self.calc_du_star_towards_q_qp(parent_node, q)
        else:
            raise Exception(f"Unknown du_star_mode {self.rrt_params.du_star_mode}")
        
        return du_star
    


    def sample_contact(self, parent_node: Node):
        """
        Sample contact and return a new node
        """
        assert parent_node.is_static

        x_next = self.contact_sampler.sample_contact(parent_node.q)

        child_node = IrsTrajectoryNode(x_next)

        child_node.contact_mode_id = self.next_contact_id
        self.next_contact_id += 1

        child_node.is_static = True
        child_node.t, child_node.distance_from_traj = self.closest_t_in_trajectory(x_next[self.q_u_indices_into_x])

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = 1.0

        edge.du = np.nan
        edge.u = np.nan

        return child_node, edge

    def extend_towards_q(self, parent_node: Node, q: np.array):
        """
        Extend towards a specified configuration q and return a new
        node,
        """
        du_star = self.calc_du_star_towards_q(parent_node, q)
        u_star = parent_node.ubar + du_star

        x_next = self.q_sim.calc_dynamics(
            parent_node.q, u_star, self.sim_params
        )

        if self.rrt_params.step_in:
            x_next = self.step_in(x_next)

        cost = 0.0

        child_node = IrsTrajectoryNode(x_next)
        child_node.subgoal = q
        child_node.contact_mode_id = parent_node.contact_mode_id
        child_node.extensions_since_regrasp = parent_node.extensions_since_regrasp + 1
        child_node.is_static = self.is_static(x_next)

        child_node.t, child_node.distance_from_traj = self.closest_t_in_trajectory(x_next[self.q_u_indices_into_x])

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = cost

        edge.du = du_star
        edge.u = u_star

        return child_node, edge
    
