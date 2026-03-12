import matplotlib.pyplot as plt

import numpy as np

from irs_rrt.irs_rrt_projection import IrsRrtProjection

from irs_mpc2.irs_mpc_params import SmoothingMode
from box_lift_setup import *


from irs_mpc2.irs_mpc_params import (
    kNoSmoothingModes,
    k1RandomizedSmoothingModes,
    kAnalyticSmoothingModes,
)

prob_rrt = IrsRrtProjection(
    rrt_params,
    contact_sampler,
    q_sim,
    q_sim_py,
    pose_sampling_function,
    # q_sim_smooth=q_sim_smooth
)


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

def calc_mahalabonis_distance(q, q_goal):
    Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_analytic(q, q[idx_q_a])

    (
        cov_u,
        mu_u,
    ) = prob_rrt.reachable_set.calc_unactuated_metric_parameters(Bhat, chat)
    covinv_u = np.linalg.inv(cov_u)

    # 1 x n
    mu_batch = mu_u[None, :]
    # 1 x n x n
    covinv_tensor = covinv_u[None, :, :]
    error_batch = q_goal[idx_q_u] - mu_batch
    int_batch = np.einsum("Bij,Bi -> Bj", covinv_tensor, error_batch)
    metric_batch = np.einsum("Bi,Bi -> B", int_batch, error_batch)

    return metric_batch[0]


while True:
    start_t = float(input("start:"))
    end_t = float(input("end:"))

    q_start = np.zeros(dim_x)
    q_start[idx_q_u] = get_obj_pose_from_t(start_t)
    q_goal = np.zeros(dim_x)
    q_goal[idx_q_u] = get_obj_pose_from_t(end_t)

    rrt_params.goal = q_goal

    np.set_printoptions(suppress=True, precision=3)

    x = []
    y = []

    for i in range(20):
        print(i)
        qi = contact_sampler.sample_contact(q_start)

        originial_mahalabonis_dist = calc_mahalabonis_distance(qi, q_goal)

        best_mahalabonis_dist= originial_mahalabonis_dist

        it = 0
        while True:
            try:
                q_a = qi[idx_q_a]
                if rrt_params.smoothing_mode in kAnalyticSmoothingModes:
                    Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_analytic(qi, q_a)
                elif rrt_params.smoothing_mode in k1RandomizedSmoothingModes:
                    Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_randomized(qi, q_a)
                elif rrt_params.smoothing_mode in kNoSmoothingModes:
                    Bhat, chat = prob_rrt.reachable_set.calc_exact_Bc(qi, q_a)


                du_star = constrained_least_squares(Bhat[idx_q_u], (q_goal - chat)[idx_q_u], rrt_params.stepsize)

                # Normalize least-squares solution.
                du_norm = max(np.linalg.norm(du_star), 1e-6)
                step_size = min(du_norm, rrt_params.stepsize)
                du_star = du_star / du_norm

                u_star = q_a + step_size * du_star
                
                qi = q_sim.calc_dynamics(qi, u_star, prob_rrt.sim_params)

                mahalabonis_distance = calc_mahalabonis_distance(qi, q_goal)

                best_mahalabonis_dist = min(mahalabonis_distance, best_mahalabonis_dist)

                it += 1

            except Exception as e:
                break

            if it > 50:
                break

        x.append(originial_mahalabonis_dist)
        y.append(best_mahalabonis_dist)

    plt.scatter(x, y)
    plt.show()