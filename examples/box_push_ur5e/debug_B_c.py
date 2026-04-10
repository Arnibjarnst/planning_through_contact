import numpy as np

from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory

from pydrake.all import AngleAxis
from box_push_setup import *

from pydrake.all import (
    JacobianWrtVariable,
)

prob_rrt = IrsRrtTrajectory(
    rrt_params,
    contact_sampler,
    q_sim,
    q_sim_py,
    pose_sampling_function,
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


def step(q, q_goal):
    try:
        Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_analytic(q, q[idx_q_a])

        du_star = constrained_least_squares(Bhat[idx_q_u], (q_goal - chat)[idx_q_u], rrt_params.stepsize)

        u_star = q[idx_q_a] + du_star

        q_next = q_sim.calc_dynamics(q, u_star, prob_rrt.sim_params)

        print(q_next[idx_q_a] - u_star)

        return q_next
    except:
        return q

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


def cast_to_cone(joints, normal, arm_pose_val, deg):
    rad = np.deg2rad(deg)

    ee_pose = utils.get_ee_pose(joints, arm_pose_val, eef_offset=eef_offset)
    ee_dir = utils.quat_apply(ee_pose[:4], np.array([0,0,1]))
    dot = np.dot(ee_dir, -normal)
    theta = np.arccos(dot)

    print(ee_dir, -normal, np.rad2deg(theta))

    if theta <= rad:
        return joints

    print("CAST JOINTS")

    tilt_axis = np.cross(-normal, ee_dir)
    tilt_axis /= np.linalg.norm(tilt_axis)
    ee_quat = Quaternion(ee_pose[:4])

    d_theta = rad - theta
    q_correction = AngleAxis(d_theta, tilt_axis).quaternion()
    ee_quat_new: Quaternion = q_correction.multiply(ee_quat)

    ee_pose_new = np.concatenate([ee_quat_new.wxyz(), ee_pose[4:]])

    joints_new = utils.get_joints(ee_pose_new, arm_pose_val, joints, robot='ur5e', eef_offset=eef_offset)

    print(ee_pose, utils.get_ee_pose(joints_new, arm_pose_val, 'ur5e', eef_offset=eef_offset), ee_pose_new)

    return joints_new

def step_in(q):
    """
    Given a near-contact configuration, give a q that steps in contact.
    Single-arm version.
    """
    q_sim_py.update_mbp_positions_from_vector(q)

    sg = q_sim_py.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
    collision_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(
        0.2
    )

    inspector = query_object.inspector()

    min_dist = np.inf
    min_body = None
    min_normal = None

    for collision in collision_pairs:
        f_id = inspector.GetFrameId(collision.id_A)
        body_A = plant.GetBodyFromFrameId(f_id)
        f_id = inspector.GetFrameId(collision.id_B)
        body_B = plant.GetBodyFromFrameId(f_id)

        # Only collisions with the box
        if body_A.model_instance() != idx_u and body_B.model_instance() != idx_u:
            continue

        # Only collisions with our arm
        if body_A.model_instance() != idx_a and body_B.model_instance() != idx_a:
            continue

        if collision.distance < min_dist:
            min_dist = collision.distance
            min_body = body_A if body_A.model_instance() == idx_a else body_B
            normal_sign = 1 if body_A.model_instance() == idx_a else -1
            min_normal = normal_sign * collision.nhat_BA_W

    q_next = np.copy(q)

    if min_body:
        J = plant.CalcJacobianTranslationalVelocity(
            q_sim_py.context_plant,
            JacobianWrtVariable.kV,
            min_body.body_frame(),
            np.array([0, 0, 0]),
            plant.world_frame(),
            plant.world_frame(),
        )

        J_a = J[:2, idx_q_a]
        qdot = np.linalg.pinv(J_a).dot(min_dist * -min_normal[:2])

        q_next[idx_q_a] += qdot
        q_next[idx_q_a] = cast_to_cone(q_next[idx_q_a], min_normal, arm_pose, 30)

    return q_next

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

        q_next = step(qi, q_goal)
        m_dist_next = get_m_dist(q_next, q_goal)

        m_dists.append(m_dist_next)

        if n_contacts == 0:
            print(m_dist, m_dist_next)
            q_vis.draw_configuration(qi)

            i = input()

            if i == '0':
                break


    qs = np.array(qs)
    m_dists = np.array(m_dists)

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
            qi = step(qi, q_goal)

            qi = step_in(qi)

            q_vis.draw_configuration(qi)

            m_dist = get_m_dist(qi, q_goal)
            t, dist = prob_rrt.closest_t_in_trajectory(qi[idx_q_u])

            print(f"Is static: {prob_rrt.is_static(qi)}")
            print(f"mahalabonis distance from goal {m_dist}")
            print(f"closest t: {t} ({dist})")

        except Exception as e:
            print(e)
            q_vis.draw_configuration(qi)
            continue
