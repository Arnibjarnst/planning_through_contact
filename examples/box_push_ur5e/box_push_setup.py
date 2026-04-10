import os
import numpy as np

from scripts import utils

from qsim.model_paths import models_dir

from pydrake.all import Quaternion, RollPitchYaw

from irs_rrt.rrt_params import IrsRrtTrajectoryParams, DuStarMode
from irs_rrt.irs_rrt_trajectory import IrsTrajectoryNode
from irs_rrt.contact_sampler import ContactSampler


from qsim.parser import QuasistaticParser
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer
from irs_mpc2.irs_mpc_params import SmoothingMode

# Names
eef_name = "ur5e"
object_name = "box"

# Robot poses
# [qw, qx, qy, qz, x, y, z]
arm_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
eef_offset = 0.0  # no sphere EEF (ur5e.sdf); use 0.02 for ur5e_sphere.sdf

# Box parameters
object_dims = np.array([0.34, 0.235, 0.27])
box_mass = 4.4
box_friction = 0.35        # cardboard
ground_friction = 0.35     # wood
ground_offset = 0.018      # wooden plate 1.8cm above robot base plane

# Generate model files from the parameters above
q_model_path = utils.generate_ur5e_box_models(
    models_dir, object_dims, {eef_name: arm_pose},
    mass=box_mass, friction=box_friction,
    ground_friction=ground_friction, ground_offset=ground_offset,
    prefix="box_push",
)
q_model_path_gradients = q_model_path
q_u0 = np.array([1, 0, 0, 0, 0, 0.5, ground_offset + object_dims[-1] / 2])
q_u1 = np.array([1, 0, 0, 0, 0.2, 0.5, ground_offset + object_dims[-1] / 2])
t0 = 0.0
t1 = 1.0

q_a0 = np.array([0, -np.pi/2, 0, 0, 0, 0])


def get_obj_pose_from_t(t: float):
    t = max(min(t, 1), 0)
    # fine since no rotation
    pose = q_u0 + (q_u1 - q_u0) * t
    return pose


pose_sampling_function = get_obj_pose_from_t

# Goal conditions
goal_u = pose_sampling_function(t1)

# data collection.
data_folder = "ptc_data/box_push_ur5e"

q_parser = QuasistaticParser(q_model_path)
q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py


q_parser_smooth = QuasistaticParser(q_model_path_gradients)
q_sim_smooth = q_parser_smooth.make_simulator_cpp()

plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(eef_name)
idx_u = plant.GetModelInstanceByName(object_name)
idx_ground = plant.GetModelInstanceByName("ground")

dim_x = plant.num_positions()

idx_q_a = q_sim.get_q_a_indices_into_q()
idx_q_u = q_sim.get_q_u_indices_into_q()


joint_limits_ur5e = np.array([
    [-6.28319, 6.28319],
    [-3.39, 0.25], # One continuous segment above ground
    [-2.8,  2.8],
    [-6.28319, 6.28319],
    [-6.28319, 6.28319],
    [-6.28319, 6.28319],
])

joint_limits = {
    idx_u: np.array(
        [
            [-0.0, 0.0], # roll
            [-0.0, 0.0], # pitch
            [-0.0, 0.0], # yaw
            [0, 0], # placeholder
            [-0.0, 0.0], # x
            [-0.0, 0.0], # y
            [0.0, 0.0], # z
        ]
    ),
    idx_a: joint_limits_ur5e,
}

# start from t
obj_pose_t0 = pose_sampling_function(t0)

q0_dict = {idx_a: q_a0, idx_u: obj_pose_t0}
q0 = q_sim.get_q_vec_from_dict(q0_dict)


rrt_params = IrsRrtTrajectoryParams(q_model_path, joint_limits)
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
rrt_params.root_node = IrsTrajectoryNode(q0)
rrt_params.max_size = 50000
rrt_params.goal = np.zeros(dim_x)
rrt_params.goal[idx_q_u] = goal_u
rrt_params.termination_tolerance = 0.05
rrt_params.subgoal_ts = [0.0, 1.0]
rrt_params.subgoal_tolerance = 0.2
rrt_params.goal_as_subgoal_prob = 0.4
rrt_params.enforce_robot_joint_limits = True
rrt_params.quat_metric = 5
rrt_params.distance_threshold = np.inf
rrt_params.regularization = 1e-3
# Randomized Parameters:
rrt_params.std_u = 0.1 * np.ones(6)
rrt_params.n_samples = 100

rrt_params.du_star_mode = DuStarMode.ConstrainedLSTSQ
rrt_params.stepsize = 0.1
rrt_params.rewire = False
rrt_params.distance_metric = "local_u"
rrt_params.grasp_prob = 0.0
rrt_params.h = 1.0 / 10.0
rrt_params.log_barrier_weight_for_bundling = 1000

rrt_params.max_static_angle_diff = 0.005
rrt_params.max_static_pos_diff = 0.005

rrt_params.obj_dims = object_dims

rrt_params.batch_size = 32
rrt_params.initial_contact_samples = 128

rrt_params.use_free_solvers = False

rrt_params.arm_poses = {eef_name: arm_pose}
rrt_params.eef_offset = eef_offset


def get_best_joint_configurations(joint_configs, q_, model_idx, joint_idx):
    q = np.copy(q_)

    best_joints = None
    best_sol_quality = np.inf

    original_joints = q[joint_idx]

    for joints in joint_configs:
        q[joint_idx] = joints

        q_sim_py.update_mbp_positions_from_vector(q)

        sg = q_sim_py.get_scene_graph()
        query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
        collision_pairs = (
            query_object.ComputeSignedDistancePairwiseClosestPoints(0.0)
        )
        inspector = query_object.inspector()

        is_collision = False
        for collision in collision_pairs:
            f_id = inspector.GetFrameId(collision.id_A)
            body_A = plant.GetBodyFromFrameId(f_id)
            f_id = inspector.GetFrameId(collision.id_B)
            body_B = plant.GetBodyFromFrameId(f_id)

            if body_A.name() == "ee_link" or body_B.name() == "ee_link":
                continue

            if body_A.model_instance() == model_idx or body_B.model_instance() == model_idx:
                is_collision = True
                break

        if not is_collision:
            test_sol = np.ones(6) * 9999.
            for i in range(6):
                for add_ang in [-2. * np.pi, 0, 2. * np.pi]:
                    test_ang = joints[i] + add_ang
                    if (
                        test_ang >= joint_limits_ur5e[i, 0] and
                        test_ang <= joint_limits_ur5e[i, 1] and
                        abs(test_ang - original_joints[i]) < abs(test_sol[i] - original_joints[i])
                    ):
                        test_sol[i] = test_ang
            if np.all(test_sol != 9999.):
                sol_distance_from_original = np.sum((test_sol - original_joints)**2)
                if sol_distance_from_original < best_sol_quality:
                    best_joints = test_sol
                    best_sol_quality = sol_distance_from_original

    return best_joints

class MagicContactSampler(ContactSampler):
    def __init__(self):
        super().__init__(q_sim, q_sim_py)

        self.box_dims = object_dims
        self.box_dims_half = self.box_dims / 2.0
        self.eef_radius = 0.02
        self.box_faces_area = np.array([
            self.box_dims[0]*self.box_dims[1],
            self.box_dims[0]*self.box_dims[2],
            self.box_dims[1]*self.box_dims[2],
        ])

        self.face_collision_area = np.sum(self.box_faces_area) * 2
        self.corner_collision_area = 4 * np.pi * (self.eef_radius ** 2)
        self.edge_collision_area = 2 * np.pi * self.eef_radius * np.sum(self.box_dims)
        self.total_collision_area = self.face_collision_area + self.corner_collision_area + self.edge_collision_area

        self.max_angle_rad = np.radians(30)
        self.z_to_y = RollPitchYaw(-np.pi/2, 0, 0).ToRotationMatrix()
        self.z_to_x = RollPitchYaw(0, np.pi/2, 0).ToRotationMatrix()

    def rand_sign(self):
        if np.random.rand() < 0.5:
            return 1
        else:
            return -1

    def sample_contact_point(self):
        r = np.random.rand() * self.total_collision_area

        if r < self.face_collision_area:
            return self.sample_face()
        elif r < self.face_collision_area + self.edge_collision_area:
            return self.sample_edge()

        return self.sample_corner()

    def sample_face(self):
        p = -self.box_dims_half + np.random.rand(3) * self.box_dims
        r = np.random.rand() * self.face_collision_area
        sign = self.rand_sign()
        if r < self.box_faces_area[0] * 2:
            p[2] = sign * (self.box_dims_half[2] + self.eef_radius)
        elif r < (self.box_faces_area[0] + self.box_faces_area[1]) * 2:
            p[1] = sign * (self.box_dims_half[1] + self.eef_radius)
        else:
            p[0] = sign * (self.box_dims_half[0] + self.eef_radius)
        return p

    def sample_edge(self):
        p = -self.box_dims_half + np.random.rand(3) * self.box_dims
        total_edge_length = np.sum(self.box_dims)
        r = np.random.rand() * total_edge_length
        rad = np.random.rand() * np.pi / 2.0
        d_rad_1 = np.cos(rad) * self.eef_radius
        d_rad_2 = np.sin(rad) * self.eef_radius
        sign1 = self.rand_sign()
        sign2 = self.rand_sign()
        if r < self.box_dims[0]:
            p[1] = sign1 * (self.box_dims_half[1] + d_rad_1)
            p[2] = sign2 * (self.box_dims_half[2] + d_rad_2)
        elif r < self.box_dims[0] + self.box_dims[1]:
            p[0] = sign1 * (self.box_dims_half[0] + d_rad_1)
            p[2] = sign2 * (self.box_dims_half[2] + d_rad_2)
        else:
            p[0] = sign1 * (self.box_dims_half[0] + d_rad_1)
            p[1] = sign2 * (self.box_dims_half[1] + d_rad_2)
        return p

    def sample_corner(self):
        v_normal = np.random.normal(0, 1, 3)
        p = v_normal / np.linalg.norm(v_normal) * self.eef_radius
        p[0] += np.sign(p[0]) * self.box_dims_half[0]
        p[1] += np.sign(p[1]) * self.box_dims_half[1]
        p[2] += np.sign(p[2]) * self.box_dims_half[2]
        return p

    def sample_contact_point_with_direction(self):
        """Sample a single contact point with approach direction on a box face."""
        p = -self.box_dims_half + np.random.rand(3) * self.box_dims

        # sample face (all faces equally probable)
        r = np.random.rand() * 3
        sign = self.rand_sign()

        # direction into the face
        direction = -sign * utils.cosine_weighted_cone(self.max_angle_rad)

        face_i = int(r)
        p[face_i] = sign * (self.box_dims_half[face_i] + self.eef_radius)

        # Rotate direction to face frame
        if r < 1:
            direction = self.z_to_x @ direction
        elif r < 2:
            direction = self.z_to_y @ direction

        return p, direction

    def sample_contact(self, q):
        q0 = np.copy(q)
        q0_quat = Quaternion(q0[idx_q_u[:4]])
        q0_rot_mat = Quaternion(q0_quat).rotation()
        q0_pos = q0[idx_q_u[4:]]

        for _ in range(1000):
            ee_pos_local, dir_local = self.sample_contact_point_with_direction()

            ee_pos_global = q0_rot_mat @ ee_pos_local + q0_pos

            if ee_pos_global[2] < self.eef_radius:
                continue

            dir_global = q0_rot_mat @ dir_local

            ee_quat_global = utils.dir_to_quat(dir_global)
            ee_pose_global = np.concatenate([ee_quat_global, ee_pos_global])

            all_joints = utils.get_joints(
                ee_pose_global, arm_pose, q0[idx_q_a], True, "ur5e",
                eef_offset=eef_offset,
            )
            joints = get_best_joint_configurations(all_joints, q0, idx_a, idx_q_a)

            if joints is None:
                continue

            q0[idx_q_a] = joints
            return q0

        raise Exception("Failed to sample contact points")

contact_sampler = MagicContactSampler()
