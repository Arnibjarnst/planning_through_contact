import numpy as np

from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory

from box_lift_setup import *

from pydrake.all import (
    JacobianWrtVariable,
)

# rrt_params.log_barrier_weight_for_bundling = 100

prob_rrt = IrsRrtTrajectory(
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


def get_m_dist(q, q_goal):
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

    mahalabonis_distance = metric_batch[0]

    return mahalabonis_distance

def step_in(q):
    """
    Given a near-contact configuration, give a q that steps in contact.
    """
    q_sim_py.update_mbp_positions_from_vector(q)

    sg = q_sim_py.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
    collision_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(
        0.2
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

        # We only care about collisions with the box
        if body_A.model_instance() != idx_u and body_B.model_instance() != idx_u:
            continue

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

    J_R = plant.CalcJacobianTranslationalVelocity(
        q_sim_py.context_plant,
        JacobianWrtVariable.kV,
        min_body_right.body_frame(),
        np.array([0, 0, 0]),
        plant.world_frame(),
        plant.world_frame(),
    )

    J_Ra = J_R[:2, idx_q_a_r]

    qdot_La = np.linalg.pinv(J_La).dot(min_dist_left * -min_normal_left[:2])
    qdot_Ra = np.linalg.pinv(J_Ra).dot(min_dist_right * -min_normal_right[:2])

    qdot = np.zeros_like(q)
    qdot[idx_q_a_l] = qdot_La
    qdot[idx_q_a_r] = qdot_Ra
    qnext = q + qdot

    return qnext

def step(q, q_goal):
    Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_analytic(q, q[idx_q_a])

    du_star = constrained_least_squares(Bhat[idx_q_u], (q_goal - chat)[idx_q_u], rrt_params.stepsize)

    u_star = q[idx_q_a] + du_star
    
    q_next = q_sim.calc_dynamics(q, u_star, prob_rrt.sim_params)

    q_next_step_in = step_in(q_next)

    delta_du_star = q_next_step_in[idx_q_a] - q_next[idx_q_a]

    du_star_2 = du_star + delta_du_star

    u_star_2 = q[idx_q_a] + du_star_2

    print(np.linalg.norm(du_star_2), np.linalg.norm(du_star))

    print(delta_du_star)
    print(du_star[:3])
    print(du_star_2[:3])

    q_next_2 = q_sim.calc_dynamics(q, u_star_2, prob_rrt.sim_params)

    return q_next, q_next_2

while True:
    start_t = float(input("start:"))
    end_t = float(input("end:"))
    n_contacts = int(input("contact samples:"))

    q_start = np.zeros(dim_x)
    q_start[idx_q_u] = get_obj_pose_from_t(start_t)
    q_goal = np.zeros(dim_x)
    q_goal[idx_q_u] = get_obj_pose_from_t(end_t)

    rrt_params.goal = q_goal

    np.set_printoptions(suppress=True, precision=3)

    qs = []
    m_dists = []

    for i in range(n_contacts if n_contacts > 0 else 100000):
        qi = contact_sampler.sample_contact(q_start)

        m_dist = get_m_dist(qi, q_goal)

        qs.append(qi)

        q_next, _ = step(qi, q_goal)
        m_dist_next = get_m_dist(q_next, q_goal)

        m_dists.append(m_dist_next)
        
        if n_contacts == 0:
            print(m_dist, m_dist_next)
            q_vis.draw_configuration(qi)

            i = input()
                
            if i == '0':
                break
    

    if n_contacts > 0:
        i_min = np.argmin(m_dists)
        qi = qs[i_min]
        print(f"mahalabonis distance from goal {m_dists[i_min]}")
        q_vis.draw_configuration(qi)

    while True:
        next_action = input("next")

        if next_action == '0':
            break

        try:
            qi, qi_2 = step(qi, q_goal)

            q_vis.draw_configuration(qi)

            input("Normal")
            q_vis.draw_configuration(qi_2)
            input("step in du")

            m_dist = get_m_dist(qi, q_goal)
            t, dist = prob_rrt.closest_t_in_trajectory(qi[idx_q_u])

            print(f"Is static: {prob_rrt.is_static(qi)}")
            print(f"mahalabonis distance from goal {m_dist}")
            print(f"closest t: {t} ({dist})")

        except Exception as e:
            print(e)
            q_vis.draw_configuration(qi)
            continue