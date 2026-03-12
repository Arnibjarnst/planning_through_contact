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


q_model_path = os.path.join(models_dir, "q_sys", "magic_finger_bimanual_box.yml")
q_model_path_gradients = os.path.join(models_dir, "q_sys", "magic_finger_bimanual_box_gradients.yml") # Can use this instead to calculate gradients (reachability_set)

# Names
eef_l_name = "finger_left"
eef_r_name = "finger_right"
object_name = "box"


# Robot poses
# [qw, qx, qy, qz, x, y, z]
arm_l_pose = np.array([0.0, 0.0, 0.0, 1.0, -0.6, -0.2, 0.0])
arm_r_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.6, -0.2, 0.0])

# conservative reach distance
arm_reach = 0.7


# initial conditions.
object_dims = np.array([0.4, 0.6, 0.06])
q_u0 = np.array([1, 0, 0, 0, 0, 0, 0.03])
t0 = 0.0
t1 = 1.0

q_a0_l_local = np.array([-0.22, 0.25, 0.0])
q_a0_r_local = np.array([0.22, 0.25, 0.0])
q_a0_l = q_a0_r_local + q_u0[4:]
q_a0_r = q_a0_l_local + q_u0[4:]

def get_obj_pose_from_t_part_1(t: float):
    t = max(min(t, 1), 0)
    rad = t * np.pi / 2
    quat = RollPitchYaw(rad, 0, 0).ToQuaternion()
    obj_pos = q_u0[4:]
    rotation_origin = np.array([0, obj_pos[1] - object_dims[1] / 2, obj_pos[2] - object_dims[2] / 2])
    rotation_origin_to_obj_center = obj_pos - rotation_origin
    pos_diff = (quat.rotation() @ rotation_origin_to_obj_center) - rotation_origin_to_obj_center
    curr_pos = q_u0[4:] + pos_diff
    pose = np.concatenate([quat.wxyz(), curr_pos])
    return pose

start_pose = get_obj_pose_from_t_part_1(1.0)
def get_obj_pose_from_t_part_2(t: float):
    t = max(min(t, 1), 0)
    rad = t * np.pi / 2
    quat = RollPitchYaw(0, rad, 0).ToQuaternion()
    next_quat = Quaternion(start_pose[:4]).multiply(quat)
    pose = np.concatenate([next_quat.wxyz(), start_pose[4:]])
    return pose


def get_obj_pose_from_t(t: float):
    t = max(min(t, 1), 0)
    t = t0 + t * (t1 - t0)
    if t <= 0.5:
        pose = get_obj_pose_from_t_part_1(2 * t)
    else:
        pose = get_obj_pose_from_t_part_2(2 * t - 1)
    return pose


pose_sampling_function = get_obj_pose_from_t

# Goal conditions
goal_u = pose_sampling_function(1.0)

# data collection.
data_folder = "ptc_data/box_lift"

q_parser = QuasistaticParser(q_model_path)
q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py

q_parser_smooth = QuasistaticParser(q_model_path_gradients)
q_sim_smooth = q_parser_smooth.make_simulator_cpp()

plant = q_sim_py.get_plant()
idx_a_l = plant.GetModelInstanceByName(eef_l_name)
idx_a_r = plant.GetModelInstanceByName(eef_r_name)
idx_u = plant.GetModelInstanceByName(object_name)

dim_x = plant.num_positions()

idx_q_a = q_sim.get_q_a_indices_into_q()
idx_q_a_l = idx_q_a[:3]
idx_q_a_r = idx_q_a[3:]
idx_q_u = q_sim.get_q_u_indices_into_q()


joint_limits = {
    idx_u: np.zeros((7,2)),
    idx_a_l: np.array([
        [-1.0, 1.0],
        [-1.0, 1.0],
        [0, 1]
    ]),
    idx_a_r: np.array([
        [-1.0, 1.0],
        [-1.0, 1.0],
        [0, 1]
    ]),
}


# start from t
obj_pose_t0 = pose_sampling_function(0)

obj_rot_mat = Quaternion(obj_pose_t0[:4]).rotation()
obj_trans =  obj_pose_t0[4:]

left_finger_t0 = obj_rot_mat @ q_a0_l_local + obj_trans
right_finger_t0= obj_rot_mat @ q_a0_r_local + obj_trans        

q0_dict = {idx_a_l: left_finger_t0, idx_a_r: right_finger_t0, idx_u: obj_pose_t0}
# q0_dict = {idx_a_l: q_a0_l, idx_a_r: q_a0_r, idx_u: q_u0}
q0 = q_sim.get_q_vec_from_dict(q0_dict)

rrt_params = IrsRrtTrajectoryParams(q_model_path, joint_limits)
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
rrt_params.root_node = IrsTrajectoryNode(q0)
rrt_params.max_size = 50000
rrt_params.goal = np.zeros(dim_x)
rrt_params.goal[idx_q_u] = goal_u
rrt_params.termination_tolerance = 0.05
rrt_params.subgoal_tolerance = 0.2
rrt_params.goal_as_subgoal_prob = 0.2
rrt_params.enforce_robot_joint_limits = True
rrt_params.quat_metric = 5
rrt_params.distance_threshold = np.inf
rrt_params.regularization = 1e-3
# Randomized Parameters:
rrt_params.std_u = 0.1 * np.ones(6)
rrt_params.n_samples = 100

rrt_params.du_star_mode = DuStarMode.ConstrainedLSTSQ
rrt_params.stepsize = 0.15
rrt_params.rewire = False
rrt_params.distance_metric = "local_u"
rrt_params.grasp_prob = 0.2
rrt_params.h = 1.0 / 10.0
rrt_params.log_barrier_weight_for_bundling = 1000

rrt_params.max_static_angle_diff = 0.005
rrt_params.max_static_pos_diff = 0.005

rrt_params.obj_dims = object_dims 

rrt_params.use_free_solvers = False


def robot_state_clamp(q_a):
    q_a_l = q_a[idx_q_a_l]
    q_a_r = q_a[idx_q_a_r]

    dq_a_l = q_a_l - arm_l_pose[4:]
    dq_a_r = q_a_r - arm_r_pose[4:]

    dist_l = np.linalg.norm(dq_a_l) 
    dist_r = np.linalg.norm(dq_a_r)

    factor_l = min(dist_l, arm_reach) / dist_l
    factor_r = min(dist_r, arm_reach) / dist_r

    clamped_q_a_l = arm_l_pose[4:] + factor_l * dq_a_l
    clamped_q_a_r = arm_r_pose[4:] + factor_r * dq_a_r
    clamped_q_a = np.zeros(6)
    clamped_q_a[idx_q_a_l] = clamped_q_a_l
    clamped_q_a[idx_q_a_r] = clamped_q_a_r
    clamped = factor_l < 1.0 or factor_r < 1.0

    return clamped_q_a, clamped


rrt_params.robot_state_clamp_func = robot_state_clamp


class MagicContactSampler(ContactSampler):
    def __init__(self):
        super().__init__(q_sim, q_sim_py)

        self.box_dims = np.array([0.4, 0.6, 0.06])
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


    def sample_face(self):
        # sample point in cube
        p = -self.box_dims_half + np.random.rand(3) * self.box_dims

        # sample face probabilities
        r = np.random.rand() * self.face_collision_area
        sign = self.rand_sign()
        if r < self.box_faces_area[0] * 2:
            p[2] = sign * (self.box_dims_half[2]+ self.eef_radius)
        elif r < (self.box_faces_area[0] + self.box_faces_area[1]) * 2:
            p[1] = sign * (self.box_dims_half[1] + self.eef_radius)
        else:
            p[0] = sign * (self.box_dims_half[0] + self.eef_radius)

        return p

    def sample_edge(self):
        # sample point in cube
        p = -self.box_dims_half + np.random.rand(3) * self.box_dims

        # sample edge probabilities
        total_edge_length = np.sum(self.box_dims)
        r = np.random.rand() * total_edge_length
        
        # sample collision direction from edge
        rad = np.random.rand() * np.pi / 2.0
        d_rad_1 = np.cos(rad) * self.eef_radius
        d_rad_2 = np.sin(rad) * self.eef_radius

        sign1 = self.rand_sign()
        sign2 = self.rand_sign()

        # cast to edge collision
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
        # sample point on sphere
        v_normal = np.random.normal(0, 1, 3)
        p = v_normal / np.linalg.norm(v_normal) * self.eef_radius

        # cast to corner depending on where on sphere the point is
        p[0] += np.sign(p[0]) * self.box_dims_half[0]
        p[1] += np.sign(p[1]) * self.box_dims_half[1]
        p[2] += np.sign(p[2]) * self.box_dims_half[2]
        
        return p
    

    def sample_faces(self):
        # sample point in cube
        p1 = -self.box_dims_half + np.random.rand(3) * self.box_dims
        p2 = np.copy(p1)

        # sample face probabilities
        r = np.random.rand() * self.face_collision_area
        sign = self.rand_sign()
        if r < self.box_faces_area[0] * 2:
            p1[2] = sign * (self.box_dims_half[2]+ self.eef_radius)
            p2[2] = -p1[2]
        elif r < (self.box_faces_area[0] + self.box_faces_area[1]) * 2:
            p1[1] = sign * (self.box_dims_half[1] + self.eef_radius)
            p2[1] = -p1[1]
        else:
            p1[0] = sign * (self.box_dims_half[0] + self.eef_radius)
            p2[0] = -p1[0]

        return p1, p2

    def sample_edges(self):
        # sample point in cube
        p1 = -self.box_dims_half + np.random.rand(3) * self.box_dims
        p2 = np.copy(p1)

        # sample edge probabilities
        total_edge_length = np.sum(self.box_dims)
        r = np.random.rand() * total_edge_length
        
        # sample collision direction from edge
        rad = np.random.rand() * np.pi / 2.0
        d_rad_1 = np.cos(rad) * self.eef_radius
        d_rad_2 = np.sin(rad) * self.eef_radius

        sign1 = self.rand_sign()
        sign2 = self.rand_sign()

        # cast to edge collision
        if r < self.box_dims[0]:
            p1[1] = sign1 * (self.box_dims_half[1] + d_rad_1)
            p1[2] = sign2 * (self.box_dims_half[2] + d_rad_2)
            p2[[1,2]] = -p1[[1,2]]
        elif r < self.box_dims[0] + self.box_dims[1]:
            p1[0] = sign1 * (self.box_dims_half[0] + d_rad_1)
            p1[2] = sign2 * (self.box_dims_half[2] + d_rad_2)
            p2[[0,2]] = -p1[[0,2]]
        else:
            p1[0] = sign1 * (self.box_dims_half[0] + d_rad_1)
            p1[1] = sign2 * (self.box_dims_half[1] + d_rad_2)
            p2[[0,1]] = -p1[[0,1]]

        return p1, p2

    def sample_corners(self):
        # sample point on sphere
        v_normal = np.random.normal(0, 1, 3)
        p1 = v_normal / np.linalg.norm(v_normal) * self.eef_radius

        # cast to corner depending on where on sphere the point is
        p1[0] += np.sign(p1[0]) * self.box_dims_half[0]
        p1[1] += np.sign(p1[1]) * self.box_dims_half[1]
        p1[2] += np.sign(p1[2]) * self.box_dims_half[2]
        p2 = -np.copy(p1)
        
        return p1, p2

    def sample_contact_point(self):
        r = np.random.rand() * self.total_collision_area

        if r < self.face_collision_area:
            return self.sample_face()
        elif r < self.face_collision_area + self.edge_collision_area:
            return self.sample_edge()
        
        return self.sample_corner()

    def sample_contact_points(self):
        r = np.random.rand() * self.total_collision_area

        if r < self.face_collision_area:
            return self.sample_faces()
        elif r < self.face_collision_area + self.edge_collision_area:
            return self.sample_edges()
        
        return self.sample_corners()

    # Only supports face sampling
    def sample_contact_points_with_direction(self):
        # sample point in cube
        p1 = -self.box_dims_half + np.random.rand(3) * self.box_dims
        p2 = np.copy(p1)

        # sample face probabilities (all faces equally probable)
        r = np.random.rand() * 3
        sign = self.rand_sign()

        # sample contact_direction in z frame
        # negative since we want direction into the face
        dir1 = -sign * utils.cosine_weighted_cone(self.max_angle_rad)
        dir2 = np.array([dir1[0], dir1[1], -dir1[2]]) # mirror around z axis

        # Cast point to face
        if r < 1:
            p1[2] = sign * (self.box_dims_half[2]+ self.eef_radius)
            p2[2] = -p1[2]
        elif r < 2:
            p1[1] = sign * (self.box_dims_half[1] + self.eef_radius)
            p2[1] = -p1[1]
            # rotate to y face
            dir1 = self.z_to_y @ dir1
            dir2 = self.z_to_y @ dir2
        else:
            p1[0] = sign * (self.box_dims_half[0] + self.eef_radius)
            p2[0] = -p1[0]
            # rotate to y face
            dir1 = self.z_to_x @ dir1
            dir2 = self.z_to_x @ dir2
        

        return p1, p2, dir1, dir2

    # def sample_contact(self, q):
    #     q0 = np.copy(q)
    #     q0_rot_mat = Quaternion(q[self.q_sim.get_q_u_indices_into_q()[:4]]).rotation()
    #     q0_pos = q[self.q_sim.get_q_u_indices_into_q()[4:]]
        
    #     for _ in range(100):
    #         contact_1_local = self.sample_contact_point()
    #         contact_1_global = q0_rot_mat @ contact_1_local + q0_pos

    #         if contact_1_global[2] >= self.eef_radius:
    #             q0[self.q_sim.get_q_a_indices_into_q()[:3]] = contact_1_global
                
    #             for _ in range(100):
    #                 contact_2_local = self.sample_contact_point()
    #                 eef_dist = np.linalg.norm(contact_1_local - contact_2_local)
    #                 contact_2_global = q0_rot_mat @ contact_2_local + q0_pos
            
    #                 if eef_dist > 2 * self.eef_radius and contact_2_global[2] >= self.eef_radius:
    #                     q0[self.q_sim.get_q_a_indices_into_q()[3:]] = contact_2_global
    #                     return q0

    #     raise Exception("Failed to sample Contact points")


    # def sample_contact(self, q):
    #     q0 = np.copy(q)
    #     q0_rot_mat = Quaternion(q[self.q_sim.get_q_u_indices_into_q()[:4]]).rotation()
    #     q0_pos = q[self.q_sim.get_q_u_indices_into_q()[4:]]
        
    #     for _ in range(1000):
    #         contact_1_local, contact_2_local = self.sample_contact_points()

    #         contact_1_global = q0_rot_mat @ contact_1_local + q0_pos
    #         contact_2_global = q0_rot_mat @ contact_2_local + q0_pos

    #         if contact_1_global[2] >= self.eef_radius and contact_2_global[2] >= self.eef_radius:
    #             q0[idx_q_a_l] = contact_1_global
    #             q0[idx_q_a_r] = contact_2_global
    #             return q0

    #     raise Exception("Failed to sample Contact points")
    
    def sample_contact(self, q):
        q0 = np.copy(q)
        q0_quat = Quaternion(q[idx_q_u[:4]])
        q0_rot_mat = Quaternion(q0_quat).rotation()
        q0_pos = q[idx_q_u[4:]]
        
        for _ in range(1000):
            contact_1_local, contact_2_local, dir_1_local, dir_2_local = self.sample_contact_points_with_direction()

            contact_1_global = q0_rot_mat @ contact_1_local + q0_pos
            contact_2_global = q0_rot_mat @ contact_2_local + q0_pos

            if contact_1_global[2] < self.eef_radius or contact_2_global[2] < self.eef_radius:
                continue

            dir_1_global = q0_rot_mat @ dir_1_local
            dir_2_global = q0_rot_mat @ dir_2_local

            # Convert dirs to quats
            quat_1_global = utils.dir_to_quat(dir_1_global)
            quat_2_global = utils.dir_to_quat(dir_2_global)

            pose_1_global = np.concatenate([quat_1_global, contact_1_global])
            pose_2_global = np.concatenate([quat_2_global, contact_2_global])

            # TODO: use these joint values in the RL instead of recomputing
            # Try using them in the RRT

            contact_1_to_arm_l = np.linalg.norm(arm_l_pose[4:] - contact_1_global)
            contact_1_to_arm_r = np.linalg.norm(arm_r_pose[4:] - contact_1_global)
            contact_2_to_arm_l = np.linalg.norm(arm_l_pose[4:] - contact_2_global)
            contact_2_to_arm_r = np.linalg.norm(arm_r_pose[4:] - contact_2_global)

            # swap if swapping would decrese the distance between contact points and arm origins
            if contact_1_to_arm_r + contact_2_to_arm_l < contact_1_to_arm_l + contact_2_to_arm_r:
                pose_1_global, pose_2_global = pose_2_global, pose_1_global

            joints_1 = utils.get_joints(pose_1_global, arm_l_pose)
            joints_2 = utils.get_joints(pose_2_global, arm_r_pose)

            if joints_1 is None or joints_2 is None:
                # First fail, try swapping contacts
                pose_1_global, pose_2_global = pose_2_global, pose_1_global
                joints_1 = utils.get_joints(pose_1_global, arm_l_pose)
                joints_2 = utils.get_joints(pose_2_global, arm_r_pose)
                if joints_1 is None or joints_2 is None:
                    # Second fail, resample contacts
                    continue

            q0[idx_q_a_l] = pose_1_global[4:]
            q0[idx_q_a_r] = pose_2_global[4:]
            return q0

        raise Exception("Failed to sample Contact points")
    
contact_sampler = MagicContactSampler()
