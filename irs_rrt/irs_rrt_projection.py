import numpy as np
from pydrake.solvers import GurobiSolver
from pydrake.solvers import MathematicalProgram

from pydrake.all import Quaternion, RollPitchYaw, AngleAxis, RigidTransform

from qsim.simulator import QuasistaticSimulator
from qsim_cpp import QuasistaticSimulatorCpp

from irs_rrt.contact_sampler import ContactSampler
from irs_rrt.irs_rrt import IrsRrt, IrsNode, IrsEdge
from irs_rrt.rrt_base import Node
from irs_rrt.rrt_params import DuStarMode, IrsRrtProjectionParams

# For prettier tqdm bar in jupyter notebooks.
from tqdm import tqdm

if "get_ipython" in locals() or "get_ipython" in globals():
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        print("Running in a jupyter notebook!")
        from tqdm.notebook import tqdm


class IrsRrtProjection(IrsRrt):
    def __init__(
        self,
        rrt_params: IrsRrtProjectionParams,
        contact_sampler: ContactSampler,
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator,
        get_obj_pose_from_t,
        q_sim_smooth: QuasistaticSimulatorCpp | None = None,
        fixed_eef_1 = None,
        fixed_eef_2 = None
    ):
        self.get_obj_pose_from_t = get_obj_pose_from_t
        self.fixed_eef_1 = fixed_eef_1
        self.fixed_eef_2 = fixed_eef_2

        self.contact_sampler = contact_sampler
        super().__init__(rrt_params, q_sim, q_sim_py, q_sim_smooth)
        self.solver = GurobiSolver()
        self.root_node.contact_mode_id = 0
        self.next_contact_id = 1
        self.trajectory_ts = [0.0, 0.5, 1.0]
        self.min_t = 0.0
        self.curr_goal_i = 1
        self.q_curr_goal = np.zeros(self.dim_x)
        self.q_curr_goal[self.q_u_indices_into_x] = get_obj_pose_from_t(self.trajectory_ts[self.curr_goal_i])

    def sample_subgoal(self, t=None):
        """
        Sample a subgoal from the configuration space.
        """
        # sample robots (doesn't matter)
        subgoal = np.random.rand(self.dim_x)
        subgoal = self.q_lb + (self.q_ub - self.q_lb) * subgoal

        # sample obj pose
        if t is None:
            t0 = max(self.min_t, self.trajectory_ts[self.curr_goal_i - 1])
            t1 = self.trajectory_ts[self.curr_goal_i]
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

        return subgoal

    # def sample_subgoal(self):
    #     # Sample translation
    #     subgoal = np.random.rand(self.dim_x)
    #     subgoal = self.q_lb + (self.q_ub - self.q_lb) * subgoal

    #     rpy = RollPitchYaw(subgoal[self.q_u_indices_into_x][:3])
    #     subgoal[self.q_u_indices_into_x[:4]] = rpy.ToQuaternion().wxyz()

    #     return subgoal
    
    # def sample_subgoal(self, t=None):
    #     """
    #     Sample a subgoal from the configuration space.
    #     """
    #     # sample robots
    #     subgoal = np.random.rand(self.dim_x)
    #     subgoal = self.q_lb + (self.q_ub - self.q_lb) * subgoal

    #     # sample obj pose
    #     if t is None:
    #         t0 = self.trajectory_ts[self.curr_goal_i - 1]
    #         t1 = self.trajectory_ts[self.curr_goal_i]
    #         t = np.random.rand() * (t1 - t0) + t0
    #     obj_pose_t = self.get_obj_pose_from_t(t)
    #     subgoal[self.q_u_indices_into_x] = obj_pose_t

    #     obj_rot_mat = Quaternion(obj_pose_t[:4]).rotation()
    #     obj_trans =  obj_pose_t[4:]

    #     if self.fixed_eef_1 is not None:
    #         subgoal[self.q_a_indices_into_x[:3]] = obj_rot_mat @ self.fixed_eef_1 + obj_trans
        
    #     if self.fixed_eef_2 is not None:
    #         subgoal[self.q_a_indices_into_x[3:]] = obj_rot_mat @ self.fixed_eef_2 + obj_trans        

    #     return subgoal
    
    def calc_distance(self, query, node):
        # 1 x n
        mu_batch = node.chat_u[None, :]
        # 1 x n x n
        covinv_tensor = node.covinv_u[None, :, :]
        error_batch = query - mu_batch
        int_batch = np.einsum("Bij,Bi -> Bj", covinv_tensor, error_batch)
        metric_batch = np.einsum("Bi,Bi -> B", int_batch, error_batch)

        return metric_batch[0]

    def select_closest_node(
        self,
        subgoal: np.array,
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
        i_min = np.argmin(d_batch)
        if d_batch[i_min] < d_threshold:
            selected_node = self.get_node_from_id(i_min)
            if print_distance:
                print("closest distance to subgoal", d_batch[selected_node.id])
        else:
            selected_node = None

        return selected_node

    def closest_t_in_trajectory(self, obj_pose):
        def distance_metric(pose1, pose2):
            # TODO: remove hard coded obj_dims
            obj_dims = np.array([0.4, 0.6, 0.06])
            obj_dims_half = obj_dims / 2
            corners = np.array([
                [x, y, z]
                for x in (-obj_dims_half[0], obj_dims_half[0])
                for y in (-obj_dims_half[1], obj_dims_half[1])
                for z in (-obj_dims_half[2], obj_dims_half[2])
            ])

            
            RT1 = RigidTransform(Quaternion(pose1[:4]), pose1[4:])
            RT2 = RigidTransform(Quaternion(pose2[:4]), pose2[4:])

            total = 0.0

            for p_c in corners:
                p_W1 = RT1.multiply(p_c)
                p_W2 = RT2.multiply(p_c)
                total += np.linalg.norm(p_W1 - p_W2)

            return total

        best_t = 0
        min_dist = 10e9

        for t in np.linspace(0, 1, 101):
            pose_t = self.get_obj_pose_from_t(t)
            dist_t = distance_metric(pose_t, obj_pose)
            if dist_t < min_dist:
                min_dist = dist_t
                best_t = t

        return best_t, min_dist

    
    def select_closest_nodes(
        self,
        subgoal: np.array,
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

        # i_min = np.argsort(d_batch)[:k_closest]
        # i_min = i_min[d_batch[i_min] < d_threshold]

        # selected_nodes = [self.get_node_from_id(i) for i in i_min]

        i_min = np.argsort(d_batch)
        i_min = i_min[d_batch[i_min] < d_threshold]

        selected_nodes = []
        selected_contact_modes = set()
        i = 0
        while i < len(i_min) and len(selected_nodes) < k_closest:
            node = self.get_node_from_id(i_min[i])
            if node.contact_mode_id not in selected_contact_modes:
                selected_nodes.append(node)
                selected_contact_modes.add(node.contact_mode_id)
            i += 1

        if print_distance:
            print("closest distances to subgoal", d_batch[i_min])

        return selected_nodes

    def iterate(self):
        """
        Main method for iteration.
        """

        pbar = tqdm(total=self.max_size)

        time_to_dist_to_goal = []

        while self.size < self.rrt_params.max_size:
            # 1. Sample a subgoal.
            if self.cointoss_for_goal():
                subgoal = self.rrt_params.goal
            else:
                subgoal = self.sample_subgoal()

            # 2. Sample closest node to subgoal
            parent_node = self.select_closest_node(
                subgoal, d_threshold=self.rrt_params.distance_threshold
            )
            if parent_node is None:
                continue
            # update progress only if a valid parent_node is chosen.

            # 3. Extend to subgoal.
            try:
                child_node, edge = self.extend(parent_node, subgoal)
            except RuntimeError:
                continue

            # 4. Attempt to rewire a candidate child node.
            if self.rrt_params.rewire:
                parent_node, child_node, edge = self.rewire(
                    parent_node, child_node
                )

            # 5. Register the new node to the graph.
            try:
                # Drawing every new node in meshcat seems to slow down
                #  tree building by quite a bit.
                self.add_node(child_node, draw_node=self.size % 3 == 0)
            except RuntimeError as e:
                print(e)
                continue
            pbar.update(1)

            child_node.value = parent_node.value + edge.cost
            self.add_edge(edge)

            # 6. Check for termination.
            dist_to_goal = self.dist_to_goal()
            if dist_to_goal < self.rrt_params.termination_tolerance:
                self.goal_node_idx = child_node.id
                print("FOUND A PATH TO GOAL!!!!!")
                break

            time_to_dist_to_goal.append([(pbar.last_print_t - pbar.start_t), dist_to_goal])

        pbar.close()

        return time_to_dist_to_goal
    
    def iterate_batched(self, batch_size):
        """
        Main method for iteration.
        """
        pbar = tqdm(total=self.max_size / batch_size)

        time_to_dist_to_goal = []

        # TODO: make a parameter
        for _ in range(100):
            try:
                child_node, edge = self.sample_contact(self.root_node)
                self.add_node(child_node, draw_node=False)
                child_node.value = self.root_node.value + edge.cost
                self.add_edge(edge)
            except RuntimeError:
                continue

        while self.size <= (self.rrt_params.max_size - batch_size):
            # 1. Sample a subgoal.
            if self.cointoss_for_goal():
                subgoal = self.q_curr_goal
            else:
                subgoal = self.sample_subgoal()

            # 2. Sample closest node to subgoal
            parent_nodes = self.select_closest_nodes(
                subgoal, batch_size, d_threshold=self.rrt_params.distance_threshold
            )
            if len(parent_nodes) == 0:
                continue
            # update progress only if a valid parent_node is chosen.

            # TODO: Make parallel
            for parent_node in parent_nodes:
                # 3. Extend to subgoal.
                try:
                    child_node, edge = self.extend_towards_q(parent_node, subgoal)
                except RuntimeError as e:
                    print(e)
                    continue

                # 4. Attempt to rewire a candidate child node.
                if self.rrt_params.rewire:
                    parent_node, child_node, edge = self.rewire(
                        parent_node, child_node
                    )

                # 5. Register the new node to the graph.
                try:
                    # Drawing every new node in meshcat seems to slow down
                    #  tree building by quite a bit.
                    closest_t, dist_to_trajectory = self.closest_t_in_trajectory(child_node.q[self.q_u_indices_into_x])
                    self.min_t = min(max(closest_t - 0.05, self.min_t), self.trajectory_ts[self.curr_goal_i])
                    if dist_to_trajectory < 0.2:
                        print(closest_t, dist_to_trajectory, self.min_t)
                    self.add_node(child_node, draw_node=self.size % (3 * batch_size) == 0)
                except RuntimeError as e:
                    print(e)
                    continue

                child_node.value = parent_node.value + edge.cost
                self.add_edge(edge)

            pbar.update(1)

            # 6. Check for termination.
            dist_to_goal = self.dist_to_goal()
            dist_to_curr_goal = np.min(self.calc_distance_batch(self.q_curr_goal))
            if dist_to_curr_goal < self.rrt_params.termination_tolerance:
                if self.curr_goal_i == len(self.trajectory_ts) - 1:
                    self.goal_node_idx = child_node.id
                    print("FOUND A PATH TO GOAL!!!!!")
                    break
                else:
                    self.curr_goal_i += 1
                    self.q_curr_goal = np.zeros(self.dim_x)
                    self.q_curr_goal[self.q_u_indices_into_x] = self.get_obj_pose_from_t(self.trajectory_ts[self.curr_goal_i])
                    print("FOUND A PATH TO SUBGOAL!!!!!")
            
            time_to_dist_to_goal.append([(pbar.last_print_t - pbar.start_t), dist_to_goal])

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

        # We need |A * x - b| ^2 + epsilon * |x|^2, but MathematicalProgram
        # requires every term in the quadratic cost to be PD.
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

        # Normalize least-squares solution.
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

    def calc_du_star_towards_q_constrained_lstsq(self, parent_node: Node, q: np.ndarray):
        """
        Solve min ||Ax - b|| subject to ||x|| <= d.
        """
        tol = 1e-10
        max_iter = 100

        A = parent_node.Bhat_u
        b = (q[self.q_u_indices_into_x] - parent_node.chat_u)

        AtA = A.T @ A
        Atb = A.T @ b

        try:
            # Unconstrained LS solution
            x0 = np.linalg.solve(AtA, Atb)
            if np.linalg.norm(x0) <= self.rrt_params.stepsize:
                return x0  # constraint inactive
        except:
            pass
                
        lam_low, lam_high = 0, 1.0
        
        # Increase lam_high until norm(x) < stepsize
        for _ in range(50):
            x = np.linalg.solve(AtA + lam_high * np.eye(A.shape[1]), Atb)
            if np.linalg.norm(x) < self.rrt_params.stepsize:
                break
            lam_high *= 2
        
        # Binary search λ
        for _ in range(max_iter):
            lam = 0.5 * (lam_low + lam_high)
            x = np.linalg.solve(AtA + lam * np.eye(A.shape[1]), Atb)
            
            if np.linalg.norm(x) > self.rrt_params.stepsize:
                lam_low = lam
            else:
                lam_high = lam
            
            if lam_high - lam_low < tol:
                break
            
        return x 


    def sample_contact(self, parent_node: Node):
        """
        Sample contact and return a new node
        """
        x_next = self.contact_sampler.sample_contact(parent_node.q)

        child_node = IrsNode(x_next)

        child_node.contact_mode_id = self.next_contact_id
        self.next_contact_id += 1

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
        regrasp = np.random.rand() < self.rrt_params.grasp_prob

        if regrasp:
            return self.sample_contact(parent_node)
    
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
        u_star = parent_node.ubar + du_star

        u_star, _ = self.rrt_params.robot_state_clamp_func(u_star)

        x_next = self.q_sim.calc_dynamics(
            parent_node.q, u_star, self.sim_params
        )

        cost = 0.0

        child_node = IrsNode(x_next)
        child_node.subgoal = q
        child_node.contact_mode_id = parent_node.contact_mode_id

        edge = IrsEdge()
        edge.parent = parent_node
        edge.child = child_node
        edge.cost = cost

        edge.du = du_star
        edge.u = u_star

        return child_node, edge
