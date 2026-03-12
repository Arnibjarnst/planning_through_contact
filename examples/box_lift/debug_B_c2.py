import numpy as np
import matplotlib.pyplot as plt

from irs_rrt.irs_rrt import IrsNode
from irs_rrt.irs_rrt_projection import IrsRrtProjection
from irs_rrt.rrt_params import IrsRrtProjectionParams

from qsim_cpp import ForwardDynamicsMode
from qsim.parser import QuasistaticParser
from irs_rrt.contact_sampler import ContactSampler
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer
from irs_mpc2.irs_mpc_params import SmoothingMode
from box_lift_setup2 import *

from irs_mpc2.irs_mpc_params import (
    kSmoothingMode2ForwardDynamicsModeMap,
    kNoSmoothingModes,
    k0RandomizedSmoothingModes,
    k1RandomizedSmoothingModes,
    kAnalyticSmoothingModes,
)

# %%
q_parser = QuasistaticParser(q_model_path)
q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py

plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(eef_name)
idx_u = plant.GetModelInstanceByName(object_name)

dim_x = plant.num_positions()
dim_u = q_sim.num_actuated_dofs()

contact_sampler = ContactSampler(q_sim, q_sim_py)

q0_dict = {idx_a: q_a0, idx_u: q_u0}
q0 = q_sim.get_q_vec_from_dict(q0_dict)


q_sim.update_mbp_positions_from_vector(q0)
q_sim_py.update_mbp_positions_from_vector(q0)
q_sim_py.draw_current_configuration()


joint_limits = {
    idx_u: np.zeros((7,2)),
    idx_a: np.array([
        [-1, 1],
        [-1, 1],
        [0, 1]
    ]),
}


rrt_params = IrsRrtProjectionParams(q_model_path, joint_limits)
rrt_params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
rrt_params.root_node = IrsNode(q0)
rrt_params.max_size = 10000
rrt_params.goal = np.copy(q0)
rrt_params.goal[q_sim.get_q_u_indices_into_q()] = goal_u
rrt_params.goal[q_sim.get_q_a_indices_into_q()] = goal_a
rrt_params.termination_tolerance = 0.1  # used in irs_rrt.iterate() as cost
rrt_params.goal_as_subgoal_prob = 0.2
rrt_params.global_metric = np.ones(q0.shape) * 0.1
rrt_params.global_metric[q_sim.get_q_u_indices_into_q()[:4]] = 0
rrt_params.quat_metric = 5
rrt_params.distance_threshold = np.inf
rrt_params.regularization = 1e-3
# Randomized Parameters:
rrt_params.std_u = 0.02 * np.ones(6)
rrt_params.n_samples = 10000
rrt_params.log_barrier_weight_for_bundling = 100

rrt_params.stepsize = 0.2
rrt_params.rewire = False
rrt_params.distance_metric = "local_u"
rrt_params.grasp_prob = 0.1
rrt_params.h = 0.02

# %% use free solvers?
use_free_solvers = False
rrt_params.use_free_solvers = use_free_solvers


prob_rrt = IrsRrtProjection(rrt_params, contact_sampler, q_sim, q_sim_py, get_obj_pose_from_t, q_a0_local)



def constrained_least_squares(A, b, d, tol=1e-10, max_iter=100):
    """
    Solve min ||Ax - b|| subject to ||x|| <= d.
    """
    AtA = A.T @ A
    Atb = A.T @ b
    
    # Unconstrained LS solution
    x0 = np.linalg.solve(AtA, Atb)
    if np.linalg.norm(x0) <= d:
        return x0  # constraint inactive
        
    lam_low, lam_high = 0, 1.0
    
    # Increase lam_high until norm(x) < d
    for _ in range(50):
        x = np.linalg.solve(AtA + lam_high * np.eye(A.shape[1]), Atb)
        if np.linalg.norm(x) < d:
            break
        lam_high *= 2
    
    # Binary search λ
    for _ in range(max_iter):
        lam = 0.5 * (lam_low + lam_high)
        x = np.linalg.solve(AtA + lam * np.eye(A.shape[1]), Atb)
        
        if np.linalg.norm(x) > d:
            lam_low = lam
        else:
            lam_high = lam
        
        if lam_high - lam_low < tol:
            break
    
    return x


start_t = 0.0
end_t = 0.5
constrained = False

while True:
    idx_u = q_sim.get_q_u_indices_into_q()
    idx_a = q_sim.get_q_a_indices_into_q()

    q_goal = prob_rrt.sample_subgoal(end_t)
    q_vis.draw_configuration(q_goal)

    if rrt_params.smoothing_mode in kAnalyticSmoothingModes:
        Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_analytic(q_goal, q_goal[idx_a])
    elif rrt_params.smoothing_mode in k1RandomizedSmoothingModes:
        Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_randomized(q_goal, q_goal[idx_a])
    elif rrt_params.smoothing_mode in kNoSmoothingModes:
        Bhat, chat = prob_rrt.reachable_set.calc_exact_Bc(q_goal, q_goal[idx_a])

    np.set_printoptions(suppress=True, precision=3)

    print(q_goal[idx_a])
    print(Bhat[idx_u])
    
    input("start")

    qi = prob_rrt.sample_subgoal(start_t)
    q_vis.draw_configuration(qi)

    while True:
        q_a = qi[idx_a]
        if rrt_params.smoothing_mode in kAnalyticSmoothingModes:
            Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_analytic(qi, q_a)
        elif rrt_params.smoothing_mode in k1RandomizedSmoothingModes:
            Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_randomized(qi, q_a)
        elif rrt_params.smoothing_mode in kNoSmoothingModes:
            Bhat, chat = prob_rrt.reachable_set.calc_exact_Bc(qi, q_a)


        if not constrained:
            du_star = np.linalg.lstsq(
                Bhat[idx_u, :],
                (q_goal - chat)[idx_u],
                rcond=None,
            )[0]
        else:
            try:
                du_star = constrained_least_squares(Bhat[idx_u], (q_goal - chat)[idx_u], rrt_params.stepsize)
            except:
                du_star = np.zeros(dim_u)
        # Normalize least-squares solution.
        du_norm = max(np.linalg.norm(du_star), 1e-6)
        step_size = min(du_norm, rrt_params.stepsize)
        du_star = du_star / du_norm

        u_star = q_a + step_size * du_star

        print(Bhat)
        print(du_star)

        input("next")

        qi = q_sim.calc_dynamics(qi, u_star, prob_rrt.sim_params)
        q_vis.draw_configuration(qi)