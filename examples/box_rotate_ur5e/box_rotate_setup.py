"""
Setup for the box_rotate_ur5e task: rotate a box by a configurable angle
about its vertical (z) axis using a single UR5e arm.

The box stays at the same position; only the yaw changes. For a 90° rotation
the arm needs to push against an edge/face with friction to spin the box in
place — the quasi-static model captures this via contact forces.

To change the rotation, edit ROTATION_DEGREES below.
"""

# -------------------------------------------------------------------------
# Task parameters — edit these to change the task
# -------------------------------------------------------------------------
ROTATION_DEGREES = 90.0   # rotate the box by this many degrees about z-axis

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
eef_offset = 0.013 #0.013  # 1.3cm offset for the black tool attacher thing
arm_kp = np.array([800, 600, 300, 200, 100, 100])

# Box parameters
object_dims = np.array([0.34, 0.235, 0.27])
box_mass = 4.4
box_friction = 0.35        # cardboard
ground_friction = 0.01     # wood
ground_offset = 0.018      # wooden plate 1.8cm above robot base plane

# Generate model files from the parameters above
q_model_path = utils.generate_ur5e_box_models(
    models_dir, object_dims, {eef_name: arm_pose},
    mass=box_mass, friction=box_friction,
    ground_friction=ground_friction, ground_offset=ground_offset,
    prefix="box_rotate", joint_kp=arm_kp,
)
q_model_path_gradients = q_model_path
t0 = 0.0
t1 = 1.0

q_a0 = np.array([0, -np.pi/2, 0, 0, 0, 0])

box_position = np.array([0, 0.5, ground_offset + object_dims[-1] / 2])

def get_obj_pose_from_t(t: float):
    """Interpolate yaw linearly from 0 to ROTATION_DEGREES."""
    t = max(min(t, 1), 0)
    rad = t * np.deg2rad(ROTATION_DEGREES)
    quat = RollPitchYaw(0.0, 0.0, rad).ToQuaternion()
    return np.concatenate([quat.wxyz(), box_position])

pose_sampling_function = get_obj_pose_from_t

# Goal conditions
goal_u = pose_sampling_function(t1)

# data collection.
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_folder = os.path.join(_repo_root, "ptc_data", "box_rotate_ur5e")

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

# Tree structure
rrt_params.root_node = IrsTrajectoryNode(q0)
rrt_params.max_size = 50000
rrt_params.goal = np.zeros(dim_x)
rrt_params.goal[idx_q_u] = goal_u
rrt_params.termination_tolerance = 0.05
rrt_params.connect_from_behind = False
rrt_params.connect_to_front = False
rrt_params.regrasp_cooldown = 7

# Subgoal / trajectory
rrt_params.subgoal_ts = [0.0, 1.0]
rrt_params.subgoal_tolerance = 0.2
rrt_params.goal_as_subgoal_prob = 0.4

# Distance / metrics
rrt_params.distance_metric = "local_u"
rrt_params.distance_threshold = np.inf
rrt_params.quat_metric = 5
rrt_params.regularization = 1e-3
rrt_params.obj_dims = object_dims
rrt_params.max_static_angle_diff = 0.005
rrt_params.max_static_pos_diff = 0.005

# Dynamics / stepping
rrt_params.h = 1.0 / 10.0
rrt_params.stepsize = 0.05
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
rrt_params.du_star_mode = DuStarMode.ConstrainedLSTSQ
rrt_params.log_barrier_weight_for_bundling = 1000
rrt_params.step_in = True
rrt_params.stickiness_scale_angular = 15.0
rrt_params.stickiness_scale_linear = 2.5


# Sampling
rrt_params.batch_size = 32
rrt_params.initial_contact_samples = 256
rrt_params.grasp_prob = 0.2

# Robot / contact
rrt_params.arm_poses = {eef_name: arm_pose}
rrt_params.eef_offset = eef_offset
rrt_params.enforce_robot_joint_limits = True

# Solver
rrt_params.use_free_solvers = False


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
            query_object.ComputeSignedDistancePairwiseClosestPoints(0.01) # Non EE need to be 1cm away from collision
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
        self.box_faces_area = np.array([
            self.box_dims[0]*self.box_dims[1],
            self.box_dims[0]*self.box_dims[2],
            self.box_dims[1]*self.box_dims[2],
        ])

        self.max_angle_rad = np.radians(0)
        self.z_to_y = RollPitchYaw(-np.pi/2, 0, 0).ToRotationMatrix()
        self.z_to_x = RollPitchYaw(0, np.pi/2, 0).ToRotationMatrix()

    def rand_sign(self):
        if np.random.rand() < 0.5:
            return 1
        else:
            return -1

    def sample_contact_point_with_direction(self):
        """Sample a single contact point with approach direction on a box face."""
        p = -self.box_dims_half + np.random.rand(3) * self.box_dims

        # sample face (all faces equally probable)
        r = np.random.rand() * 3
        sign = self.rand_sign()

        # direction into the face
        direction = -sign * utils.cosine_weighted_cone(self.max_angle_rad)

        face_i = int(r)
        p[face_i] = sign * (self.box_dims_half[face_i])

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

            dir_global = q0_rot_mat @ dir_local

            ee_quat_global = utils.dir_to_quat(dir_global)
            ee_pose_global = np.concatenate([ee_quat_global, ee_pos_global])

            all_joints = utils.get_joints(
                ee_pose_global, arm_pose, q0[idx_q_a], True,
                eef_offset=eef_offset,
            )
            joints = get_best_joint_configurations(all_joints, q0, idx_a, idx_q_a)

            if joints is None:
                continue

            q0[idx_q_a] = joints
            return q0

        raise Exception("Failed to sample contact points")

contact_sampler = MagicContactSampler()
