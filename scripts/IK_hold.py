import numpy as np
import copy

import utils

from qsim.parser import QuasistaticParser
from examples.box_lift.box_lift_setup import *

from qsim.simulator import (
    GradientMode,
    ForwardDynamicsMode,
)

np.set_printoptions(precision=8, suppress=True)


obj_quat = RollPitchYaw(np.pi / 2.0, 0.0, 0.0).ToQuaternion()
obj_trans = np.array([0.0, 0.0, 1.0])
obj_pose = np.concatenate([obj_quat.wxyz(), obj_trans])

obj_rot_mat = obj_quat.rotation()

left_finger = obj_rot_mat @ q_a0_l_local + obj_trans
right_finger= obj_rot_mat @ q_a0_r_local + obj_trans        

q0_dict = {idx_a_l: left_finger, idx_a_r: right_finger, idx_u: obj_pose}
q0 = q_sim.get_q_vec_from_dict(q0_dict)


q_sim.update_mbp_positions_from_vector(q0)
q_sim_py.update_mbp_positions_from_vector(q0)
q_sim_py.draw_current_configuration()

sim_params = copy.deepcopy(q_sim.get_sim_params())
sim_params.calc_contact_forces = False
sim_params.h = rrt_params.h
sim_params.log_barrier_weight = (
    rrt_params.log_barrier_weight_for_bundling
)
sim_params.forward_mode = ForwardDynamicsMode.kSocpMp
sim_params.use_free_solvers = rrt_params.use_free_solvers


q_parser_slippery = QuasistaticParser(q_model_path_gradients)
q_sim_slippery = q_parser_slippery.make_simulator_cpp()


gradient_sim_params = copy.deepcopy(sim_params)
gradient_sim_params.gradient_mode = GradientMode.kBOnly
gradient_sim_params.forward_mode = sim_params.forward_mode


def calc_du_star_towards_q_constrained_lstsq(Bhat: np.ndarray, chat: np.ndarray, q: np.ndarray):
        """
        Solve min ||Ax - b|| subject to ||x|| <= d.
        """
        tol = 1e-10
        max_iter = 100

        A = Bhat[idx_u, :]
        b = (q - chat)[idx_u]

        AtA = A.T @ A
        Atb = A.T @ b

        try:
            # Unconstrained LS solution
            x0 = np.linalg.solve(AtA, Atb)
            if np.linalg.norm(x0) <= rrt_params.stepsize:
                return x0  # constraint inactive
        except:
            pass
                
        lam_low, lam_high = 0, 1.0
        
        # Increase lam_high until norm(x) < d
        for _ in range(50):
            x = np.linalg.solve(AtA + lam_high * np.eye(A.shape[1]), Atb)
            if np.linalg.norm(x) < rrt_params.stepsize:
                break
            lam_high *= 2
        
        # Binary search λ
        for _ in range(max_iter):
            lam = 0.5 * (lam_low + lam_high)
            x = np.linalg.solve(AtA + lam * np.eye(A.shape[1]), Atb)
            
            if np.linalg.norm(x) > rrt_params.stepsize:
                lam_low = lam
            else:
                lam_high = lam
            
            if lam_high - lam_low < tol:
                break
            
        return x 


idx_u = q_sim.get_q_u_indices_into_q()
idx_a = q_sim.get_q_a_indices_into_q()

q_a = q0[idx_a]

chat = q_sim.calc_dynamics(
    q=q0, u=q_a, sim_params=gradient_sim_params
)

Bhat = q_sim.get_Dq_nextDqa_cmd()

du_star = calc_du_star_towards_q_constrained_lstsq(Bhat, chat, q0)

u_star = q_a + du_star

q_next = q_sim.calc_dynamics(q0, u_star, sim_params)

f_unit_l = du_star[:3] / np.linalg.norm(du_star[:3])
f_unit_r = du_star[3:] / np.linalg.norm(du_star[3:])

pose_l = utils.get_pose(q_a[:3], f_unit_l)
pose_r = utils.get_pose(q_a[3:], f_unit_r)
pose_l_target = utils.get_pose(u_star[:3], f_unit_l)
pose_r_target = utils.get_pose(u_star[3:], f_unit_r)

arm_l_pose = np.array([1.0, 0.0, 0.0, 0.0, -0.3, -0.5, 0.5])
arm_r_pose = np.array([1.0, 0.0, 0.0, 0.0,  0.3, -0.5, 0.5])


joints_l = utils.get_joints(pose_l, arm_l_pose)
joints_r = utils.get_joints(pose_r, arm_r_pose)
joints_l_target = utils.get_joints(pose_l_target, arm_l_pose, joints_l)
joints_r_target = utils.get_joints(pose_r_target, arm_r_pose, joints_r)

np.savez_compressed(
    "test_data/IK_hold.npz",
    obj_poses=obj_pose[None, ...],
    joints_l=joints_l[None, ...],
    joints_l_target=joints_l_target[None, ...],
    joints_r=joints_r[None, ...],
    joints_r_target=joints_r_target[None, ...],
    arm_l_pose=arm_l_pose,
    arm_r_pose=arm_r_pose,
    dt = rrt_params.h
)
