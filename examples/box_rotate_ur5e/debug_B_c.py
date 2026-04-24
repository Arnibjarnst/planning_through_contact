import numpy as np

from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory

from pydrake.all import AngleAxis, Quaternion, RotationMatrix
from box_rotate_setup import *

from pydrake.all import (
    JacobianWrtVariable,
)

from irs_mpc2.irs_mpc_params import kSmoothingMode2ForwardDynamicsModeMap


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

    print(A)
    print(b)

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
    
    print(f"lam_low: {lam_low}, lam_high: {lam_high}")
    return x

from pydrake.all import JacobianWrtVariable                                                                                                
                                                            
def get_ee_jacobian(q):
    """Get EE spatial Jacobian (6, n_a) at current config. Rows: [ang; lin]."""
    q_sim_py.update_mbp_positions_from_vector(q)
    ctx = q_sim_py.context_plant
    ee_body = plant.GetBodyByName("ee_link")
    J_ee_full = plant.CalcJacobianSpatialVelocity(
        ctx, JacobianWrtVariable.kV,
        ee_body.body_frame(), np.zeros(3),
        plant.world_frame(), plant.world_frame(),
    )
    return J_ee_full[:, idx_q_a]


def get_contact_normal(q):
    """Get contact normal pointing from box toward EE, in world frame."""
    q_sim_py.update_mbp_positions_from_vector(q)
    sg = q_sim_py.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
    collision_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(0.02)
    inspector = query_object.inspector()

    min_dist = np.inf
    min_normal = None

    for collision in collision_pairs:
        f_id = inspector.GetFrameId(collision.id_A)
        body_A = plant.GetBodyFromFrameId(f_id)
        f_id = inspector.GetFrameId(collision.id_B)
        body_B = plant.GetBodyFromFrameId(f_id)

        if body_A.model_instance() != idx_u and body_B.model_instance() != idx_u:
            continue
        if body_A.model_instance() != idx_a and body_B.model_instance() != idx_a:
            continue

        if collision.distance < min_dist:
            min_dist = collision.distance
            normal_sign = 1 if body_A.model_instance() == idx_a else -1
            min_normal = normal_sign * collision.nhat_BA_W

    return min_normal


def build_stickiness_block(q, J_box_a, lam_ang=5.0, lam_lin=5.0):
    """Full stickiness: penalize all relative EE-box twist."""
    J_ee_a = get_ee_jacobian(q)
    rel_J = J_ee_a - J_box_a

    A_stick = np.vstack([lam_ang * rel_J[0:3], lam_lin * rel_J[3:6]])
    b_stick = np.zeros(6)
    return A_stick, b_stick


def build_tangential_stickiness_block(q, J_box_a, lam_ang=5.0, lam_lin=5.0):
    """Tangential stickiness: only penalize sliding/rotating along the surface,
    not the normal direction (which step_in / contact solver handles)."""
    J_ee_a = get_ee_jacobian(q)
    rel_J = J_ee_a - J_box_a

    contact_normal = get_contact_normal(q)
    if contact_normal is None:
        return build_stickiness_block(q, J_box_a, lam_ang, lam_lin)

    n = contact_normal / np.linalg.norm(contact_normal)
    # Tangential projector: I - nnᵀ
    P_tang = np.eye(3) - np.outer(n, n)

    # Project linear relative velocity onto tangential plane
    rel_lin_tang = P_tang @ rel_J[3:6]
    # Angular: project rotation axis onto normal (penalize rotation around non-normal axes)
    # Actually, rotation around the contact normal = spinning in place = sliding rotation
    # Rotation around tangent axes = tilting = changes contact angle
    # We want to penalize spinning (around normal) and sliding (tangential linear)
    P_normal = np.outer(n, n)
    rel_ang_spinning = P_normal @ rel_J[0:3]  # spinning around contact normal
    rel_ang_tilting = P_tang @ rel_J[0:3]     # tilting away from contact

    # Penalize tangential sliding + spinning (not normal linear or tilting)
    A_stick = np.vstack([
        lam_ang * rel_ang_spinning,   # (1 effective DOF via projection)
        lam_ang * rel_ang_tilting,    # (2 effective DOFs)
        lam_lin * rel_lin_tang,       # (2 effective DOFs)
    ])
    # Drop: lam_lin * normal linear velocity (handled by contact solver)
    b_stick = np.zeros(A_stick.shape[0])
    return A_stick, b_stick


def build_nullspace_stickiness_block(q, A_box, J_box_a, lam_ang=5.0, lam_lin=5.0):
    """Null-space stickiness: project stickiness into the null space of A_box
    so it cannot fight the primary box-motion objective."""
    J_ee_a = get_ee_jacobian(q)
    rel_J = J_ee_a - J_box_a

    A_stick_full = np.vstack([lam_ang * rel_J[0:3], lam_lin * rel_J[3:6]])

    # Null-space projector of the primary objective
    A_pinv = np.linalg.pinv(A_box)
    N = np.eye(A_box.shape[1]) - A_pinv @ A_box

    A_stick = A_stick_full @ N
    b_stick = np.zeros(6)
    return A_stick, b_stick

def quat_log(q):
    """Log map: unit quaternion [w,x,y,z] -> rotation vector (3,)."""
    w, xyz = q[0], q[1:]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-10:
        return 2.0 * xyz
    half_angle = np.arctan2(sin_half, w)
    return 2.0 * half_angle / sin_half * xyz


STICK_MODE = "full"  # "none", "full", "tangential", "nullspace"
LAM_ANG = 50.0
LAM_LIN = 0.0
SCALE_BY_RESIDUAL = False


def bhat_to_tangent(Bhat, chat):
    w, x, y, z = chat[idx_q_u[:4]]
    ET = 2.0 * np.array([
        [-x,  w,  z, -y],
        [-y, -z,  w,  x],
        [-z,  y, -x,  w],
    ])
    J_ang = ET @ Bhat[idx_q_u[:4], :]
    J_pos = Bhat[idx_q_u[4:], :]
    return np.vstack([J_ang, J_pos])


def prepare_lstsq(Bhat, chat, q, q_goal, mode=None):
    if mode is None:
        mode = STICK_MODE

    chat_quat = chat[idx_q_u[:4]]
    q_goal_quat = q_goal[idx_q_u[:4]]

    dq = Quaternion(q_goal_quat).multiply(Quaternion(chat_quat).inverse())
    b_ang = quat_log(dq.wxyz())
    b_pos = q_goal[idx_q_u[4:]] - chat[idx_q_u[4:]]

    A_box = bhat_to_tangent(Bhat, chat)
    b_box = np.concatenate([b_ang, b_pos])

    residual = np.linalg.norm(b_box) if SCALE_BY_RESIDUAL else 1.0
    lam_ang = LAM_ANG * residual
    lam_lin = LAM_LIN * residual

    if mode == "none":
        return A_box, b_box
    elif mode == "full":
        A_stick, b_stick = build_stickiness_block(q, A_box, lam_ang, lam_lin)
    elif mode == "tangential":
        A_stick, b_stick = build_tangential_stickiness_block(q, A_box, lam_ang, lam_lin)
    elif mode == "nullspace":
        A_stick, b_stick = build_nullspace_stickiness_block(q, A_box, A_box, lam_ang, lam_lin)
    else:
        raise ValueError(f"Unknown stickiness mode: {mode}")

    A = np.vstack([A_box, A_stick])
    b = np.concatenate([b_box, b_stick])

    return A, b


def step(q, q_goal, mode=None, verbose=True):
    try:
        Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_analytic(q, q[idx_q_a])

        A, b = prepare_lstsq(Bhat, chat, q, q_goal, mode=mode)
        du_star = constrained_least_squares(A, b, rrt_params.stepsize)

        u_star = q[idx_q_a] + du_star
        # prob_rrt.sim_params.log_barrier_weight = 100
        prob_rrt.sim_params.forward_mode = kSmoothingMode2ForwardDynamicsModeMap[rrt_params.smoothing_mode]
        q_next = q_sim.calc_dynamics(q, u_star, prob_rrt.sim_params)

        if verbose:
            print(f"  du norm: {np.linalg.norm(du_star):.4f}")
            print(f"  du: {du_star}")

        return q_next
    except Exception as e:
        print(f"  step failed: {e}")
        return q


def step_compare(q, q_goal):
    """Take one step with each stickiness mode and compare."""
    Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_analytic(q, q[idx_q_a])
    A_box = bhat_to_tangent(Bhat, chat)
    J_ee_a = get_ee_jacobian(q)
    contact_normal = get_contact_normal(q)

    print(f"\n{'='*60}")
    print(f"  Contact normal: {contact_normal}")
    print(f"  A_box rank: {np.linalg.matrix_rank(A_box)}")
    print(f"  A_box singular values: {np.linalg.svd(A_box, compute_uv=False)}")

    modes = ["none", "full", "tangential", "nullspace"]
    results = {}

    for mode in modes:
        try:
            A, b = prepare_lstsq(Bhat, chat, q, q_goal, mode=mode)
            du = constrained_least_squares(A, b, rrt_params.stepsize)

            u_star = q[idx_q_a] + du
            q_next = q_sim.calc_dynamics(q, u_star, prob_rrt.sim_params)

            # Compute relative EE-box twist for this du
            rel_J = J_ee_a - A_box
            rel_twist = rel_J @ du
            rel_ang_vel = rel_twist[:3]
            rel_lin_vel = rel_twist[3:]

            # Decompose into normal/tangential if we have contact info
            if contact_normal is not None:
                n = contact_normal / np.linalg.norm(contact_normal)
                lin_normal = np.dot(rel_lin_vel, n)
                lin_tangent = np.linalg.norm(rel_lin_vel - lin_normal * n)
                ang_spinning = np.dot(rel_ang_vel, n)
                ang_tilting = np.linalg.norm(rel_ang_vel - ang_spinning * n)
            else:
                lin_normal = lin_tangent = ang_spinning = ang_tilting = float('nan')

            # Box motion achieved
            box_motion = A_box @ du
            box_err_before = np.linalg.norm(b[:6] if len(b) >= 6 else b)

            m_dist = get_m_dist(q_next, q_goal)

            results[mode] = {
                'du': du, 'q_next': q_next, 'du_norm': np.linalg.norm(du),
                'rel_ang': np.linalg.norm(rel_ang_vel),
                'rel_lin': np.linalg.norm(rel_lin_vel),
                'lin_normal': lin_normal, 'lin_tangent': lin_tangent,
                'ang_spinning': ang_spinning, 'ang_tilting': ang_tilting,
                'box_motion_norm': np.linalg.norm(box_motion),
                'm_dist': m_dist,
            }
        except Exception as e:
            print(f"  {mode}: FAILED ({e})")
            results[mode] = None

    # Print comparison table
    print(f"\n  {'Mode':<12} {'||du||':>7} {'m_dist':>8} "
          f"{'|ω_rel|':>7} {'|v_rel|':>7} "
          f"{'v_norm':>7} {'v_tang':>7} {'ω_spin':>7} {'ω_tilt':>7} "
          f"{'box_mot':>7}")
    print(f"  {'-'*90}")

    for mode in modes:
        r = results[mode]
        if r is None:
            print(f"  {mode:<12} FAILED")
            continue
        print(f"  {mode:<12} {r['du_norm']:7.4f} {r['m_dist']:8.2f} "
              f"{r['rel_ang']:7.4f} {r['rel_lin']:7.4f} "
              f"{r['lin_normal']:7.4f} {r['lin_tangent']:7.4f} "
              f"{r['ang_spinning']:7.4f} {r['ang_tilting']:7.4f} "
              f"{r['box_motion_norm']:7.4f}")

    return results

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

    joints_new = utils.get_joints(ee_pose_new, arm_pose_val, joints, eef_offset=eef_offset)

    print(ee_pose, utils.get_ee_pose(joints_new, arm_pose_val, eef_offset=eef_offset), ee_pose_new)

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
        # q_next[idx_q_a] = cast_to_cone(q_next[idx_q_a], min_normal, arm_pose, 30)

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

        m_dists.append(m_dist)

        if n_contacts == 0:
            print(m_dist)
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

    q0 = np.copy(qi)

    while True:
        next_action = input("action (enter=step, c=compare, 0=quit, mode=(mode) lam=(lam_ang, lam_lin), r=reset): ").strip()

        if next_action == '0':
            break
        elif next_action == 'c':
            results = step_compare(qi, q_goal)
            pick = input("  pick mode to apply (none/full/tangential/nullspace, enter=skip): ").strip()
            if pick in results and results[pick] is not None:
                qi = results[pick]['q_next']
                qi = step_in(qi)
                q_vis.draw_configuration(qi)
            continue
        elif next_action.startswith("mode="):
            STICK_MODE = next_action.split("=")[1]
            print(f"  Stickiness mode set to: {STICK_MODE}")
            continue
        elif next_action.startswith("lam="):
            parts = next_action.split("=")[1].split(",")
            LAM_ANG = float(parts[0])
            LAM_LIN = float(parts[1]) if len(parts) > 1 else LAM_ANG
            print(f"  Lambda set to: ang={LAM_ANG}, lin={LAM_LIN}")
            continue
        elif next_action == 'r':
            qi = np.copy(q0)
            q_vis.draw_configuration(qi)
            continue

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
