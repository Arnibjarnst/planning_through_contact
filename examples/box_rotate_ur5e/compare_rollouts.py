import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import AngleAxis, Quaternion, RotationMatrix, JacobianWrtVariable

from box_rotate_setup import *
from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory

DEBUG = "--debug" in sys.argv

prob_rrt = IrsRrtTrajectory(
    rrt_params,
    contact_sampler,
    q_sim,
    q_sim_py,
    pose_sampling_function,
)

reachable_set = prob_rrt.reachable_set
reg = rrt_params.regularization
obj_corners = prob_rrt.obj_corners


def constrained_least_squares(A, b, d, tol=1e-10, max_iter=100):
    AtA = A.T @ A
    Atb = A.T @ b

    x0 = np.linalg.solve(AtA, Atb)
    if np.linalg.norm(x0) <= d:
        return x0

    lam_low, lam_high = 0, 1.0
    for _ in range(50):
        x = np.linalg.solve(AtA + lam_high * np.eye(A.shape[1]), Atb)
        if np.linalg.norm(x) < d:
            break
        lam_high *= 2

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


def quat_log(q):
    w, xyz = q[0], q[1:]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-10:
        return 2.0 * xyz
    half_angle = np.arctan2(sin_half, w)
    return 2.0 * half_angle / sin_half * xyz


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


def tangent_error(chat, q_goal):
    dq = Quaternion(q_goal[idx_q_u[:4]]).multiply(
        Quaternion(chat[idx_q_u[:4]]).inverse()
    )
    e_ang = quat_log(dq.wxyz())
    e_pos = q_goal[idx_q_u[4:]] - chat[idx_q_u[4:]]
    return np.concatenate([e_ang, e_pos])


def build_stickiness_block(q, J_box_a, lam_slide=5.0, lam_rotate=5.0):
    q_sim_py.update_mbp_positions_from_vector(q)
    ctx = q_sim_py.context_plant

    ee_body = plant.GetBodyByName("ee_link")
    J_ee_full = plant.CalcJacobianSpatialVelocity(
        ctx,
        JacobianWrtVariable.kV,
        ee_body.body_frame(),
        np.zeros(3),
        plant.world_frame(),
        plant.world_frame(),
    )
    J_ee_a = J_ee_full[:, idx_q_a]

    rel_J = J_ee_a - J_box_a

    A_stick = np.vstack([lam_rotate * rel_J[0:3], lam_slide * rel_J[3:6]])
    b_stick = np.zeros(6)
    return A_stick, b_stick


def corner_distance(pose_a, pose_b):
    R_a = Quaternion(pose_a[:4]).rotation()
    R_b = Quaternion(pose_b[:4]).rotation()
    corners_a = (R_a @ obj_corners.T).T + pose_a[4:]
    corners_b = (R_b @ obj_corners.T).T + pose_b[4:]
    return np.sum(np.linalg.norm(corners_a - corners_b, axis=1))


def compute_m_dist_quat(Bhat, chat, q_goal):
    Bhat_u = Bhat[idx_q_u]
    cov_u = Bhat_u @ Bhat_u.T + reg * np.eye(len(idx_q_u))
    error = q_goal[idx_q_u] - chat[idx_q_u]
    return error @ np.linalg.inv(cov_u) @ error


def compute_m_dist_tangent(Bhat_tangent, chat, q_goal):
    cov_t = Bhat_tangent @ Bhat_tangent.T + reg * np.eye(6)
    error = tangent_error(chat, q_goal)
    return error @ np.linalg.inv(cov_t) @ error


def step_in(q):
    q_sim_py.update_mbp_positions_from_vector(q)
    sg = q_sim_py.get_scene_graph()
    query_object = sg.GetOutputPort("query").Eval(q_sim_py.context_sg)
    collision_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(0.2)
    inspector = query_object.inspector()

    min_dist = np.inf
    min_body = None
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

    return q_next


def step(q, q_goal, method, lam_rot=0.0, lam_slide=0.0):
    Bhat, chat = reachable_set.calc_bundled_Bc_analytic(q, q[idx_q_a])
    use_stickiness = lam_rot > 0 or lam_slide > 0

    if method == "7x6":
        A = Bhat[idx_q_u]
        b = (q_goal - chat)[idx_q_u]
        if use_stickiness:
            A_box_6 = bhat_to_tangent(Bhat, chat)
            residual = np.linalg.norm(b)
            A_stick, b_stick = build_stickiness_block(
                q, A_box_6, lam_slide * residual, lam_rot * residual)
            A = np.vstack([A, A_stick])
            b = np.concatenate([b, b_stick])
    elif method == "6x6":
        A_box = bhat_to_tangent(Bhat, chat)
        b_box = tangent_error(chat, q_goal)
        if use_stickiness:
            residual = np.linalg.norm(b_box)
            A_stick, b_stick = build_stickiness_block(
                q, A_box, lam_slide * residual, lam_rot * residual)
            A = np.vstack([A_box, A_stick])
            b = np.concatenate([b_box, b_stick])
        else:
            A, b = A_box, b_box

    du = constrained_least_squares(A, b, rrt_params.stepsize)
    u_star = q[idx_q_a] + du
    q_next = q_sim.calc_dynamics(q, u_star, prob_rrt.sim_params)
    return q_next, Bhat, chat


def rollout(q_start, q_goal, n_steps, method, lam_rot=0.0, lam_slide=0.0, label=""):
    q = np.copy(q_start)
    q_traj = [np.copy(q)]
    m_dists_q = []
    m_dists_t = []
    c_dists = []
    goal_pose = q_goal[idx_q_u]

    for i in range(n_steps):
        try:
            Bhat, chat = reachable_set.calc_bundled_Bc_analytic(q, q[idx_q_a])
            Bt = bhat_to_tangent(Bhat, chat)

            mq = compute_m_dist_quat(Bhat, chat, q_goal)
            mt = compute_m_dist_tangent(Bt, chat, q_goal)
            cd = corner_distance(q[idx_q_u], goal_pose)
            m_dists_q.append(mq)
            m_dists_t.append(mt)
            c_dists.append(cd)

            if DEBUG:
                q_vis.draw_configuration(q)
                print(f"  [{label}] step {i}/{n_steps}  "
                      f"m_quat={mq:.4f}  m_tang={mt:.4f}  corner={cd:.4f}")
                inp = input("    (enter=next, 0=abort): ").strip()
                if inp == '0':
                    break

            q_next, _, _ = step(q, q_goal, method, lam_rot, lam_slide)
            q_next = step_in(q_next)
            q = q_next
            q_traj.append(np.copy(q))
        except Exception as e:
            print(f"  {label} failed at step {i}: {e}")
            m_dists_q.append(np.nan)
            m_dists_t.append(np.nan)
            c_dists.append(np.nan)
            break

    # Final distances
    try:
        Bhat, chat = reachable_set.calc_bundled_Bc_analytic(q, q[idx_q_a])
        Bt = bhat_to_tangent(Bhat, chat)
        mq = compute_m_dist_quat(Bhat, chat, q_goal)
        mt = compute_m_dist_tangent(Bt, chat, q_goal)
        cd = corner_distance(q[idx_q_u], goal_pose)
        m_dists_q.append(mq)
        m_dists_t.append(mt)
        c_dists.append(cd)

        if DEBUG:
            q_vis.draw_configuration(q)
            print(f"  [{label}] final  "
                  f"m_quat={mq:.4f}  m_tang={mt:.4f}  corner={cd:.4f}")
    except Exception:
        m_dists_q.append(np.nan)
        m_dists_t.append(np.nan)
        c_dists.append(np.nan)

    return np.array(m_dists_q), np.array(m_dists_t), np.array(c_dists), np.array(q_traj)


def visualize_trajectory(q_traj, label, debug=False):
    print(f"\n  Visualizing: {label} ({len(q_traj)} frames)")
    for i, q in enumerate(q_traj):
        q_vis.draw_configuration(q)
        if debug:
            inp = input(f"    [{label}] frame {i}/{len(q_traj)-1} (enter=next, 0=skip): ").strip()
            if inp == '0':
                break
        else:
            time.sleep(0.1)


# --- Interactive loop ---
# (method, lam_rot, lam_slide, label, color, linestyle)
METHODS = [
    ("6x6",   0,    0,   "6x6",              "tab:gray",    "-"),
    ("6x6", 2.5,  2.5,   "6x6 r2.5 s2.5",   "tab:green",   "-"),
    ("6x6", 2.5,    0,   "6x6 r2.5 s0",      "tab:green",   "--"),
    ("6x6",   0,  2.5,   "6x6 r0 s2.5",      "tab:green",   ":"),
    ("6x6",   5,    5,   "6x6 r5 s5",         "tab:blue",    "-"),
    ("6x6",   5,    0,   "6x6 r5 s0",         "tab:blue",    "--"),
    ("6x6",   0,    5,   "6x6 r0 s5",         "tab:blue",    ":"),
    ("6x6", 7.5,  7.5,   "6x6 r7.5 s7.5",    "tab:orange",  "-"),
    ("6x6", 7.5,    0,   "6x6 r7.5 s0",       "tab:orange",  "--"),
    ("6x6",   0,  7.5,   "6x6 r0 s7.5",       "tab:orange",  ":"),
]

np.set_printoptions(suppress=True, precision=3)

while True:
    start_t = float(input("start t: "))
    end_t = float(input("end t: "))
    n_contacts = int(input("contact samples (0=interactive): "))
    top_k = int(input("top k: "))
    n_steps = int(input("N steps: "))

    q_start = np.zeros(dim_x)
    q_start[idx_q_u] = get_obj_pose_from_t(start_t)
    q_goal = np.zeros(dim_x)
    q_goal[idx_q_u] = get_obj_pose_from_t(end_t)

    rrt_params.goal = q_goal

    # Sample contacts and compute initial distances
    if n_contacts > 0:
        qs = []
        init_m_q = []
        init_m_t = []
        init_c = []
        goal_pose = q_goal[idx_q_u]
        for i in range(n_contacts):
            try:
                qi = contact_sampler.sample_contact(q_start)
                Bhat, chat = reachable_set.calc_bundled_Bc_analytic(qi, qi[idx_q_a])
                Bt = bhat_to_tangent(Bhat, chat)
                mq = compute_m_dist_quat(Bhat, chat, q_goal)
                mt = compute_m_dist_tangent(Bt, chat, q_goal)
                cd = corner_distance(qi[idx_q_u], goal_pose)
                qs.append(qi)
                init_m_q.append(mq)
                init_m_t.append(mt)
                init_c.append(cd)
            except Exception:
                continue

        qs = np.array(qs)
        init_m_q = np.array(init_m_q)
        init_m_t = np.array(init_m_t)
        init_c = np.array(init_c)

        # Select top-k by tangent Mahalanobis
        top_k = min(top_k, len(qs))
        top_indices = np.argsort(init_m_t)[:top_k]
        print(f"Sampled {len(qs)} contacts, selecting top {top_k}")
        for rank, idx in enumerate(top_indices):
            print(f"  #{rank}: m_quat={init_m_q[idx]:.4f}  m_tang={init_m_t[idx]:.4f}  corner={init_c[idx]:.4f}")

        selected_qs = qs[top_indices]
        selected_m_q = init_m_q[top_indices]
        selected_m_t = init_m_t[top_indices]
        selected_c = init_c[top_indices]
    else:
        selected_qs = []
        selected_m_q = []
        selected_m_t = []
        selected_c = []
        goal_pose = q_goal[idx_q_u]
        while len(selected_qs) < top_k:
            try:
                qi = contact_sampler.sample_contact(q_start)
                Bhat, chat = reachable_set.calc_bundled_Bc_analytic(qi, qi[idx_q_a])
                Bt = bhat_to_tangent(Bhat, chat)
                mq = compute_m_dist_quat(Bhat, chat, q_goal)
                mt = compute_m_dist_tangent(Bt, chat, q_goal)
                cd = corner_distance(qi[idx_q_u], goal_pose)
                print(f"m_quat={mq:.4f}  m_tang={mt:.4f}  corner={cd:.4f}")
                q_vis.draw_configuration(qi)

                choice = input("accept? (enter=next, y=accept, 0=done): ").strip()
                if choice == 'y':
                    selected_qs.append(qi)
                    selected_m_q.append(mq)
                    selected_m_t.append(mt)
                    selected_c.append(cd)
                    print(f"  Accepted {len(selected_qs)}/{top_k}")
                elif choice == '0':
                    break
            except Exception as e:
                print(f"sample failed: {e}")
                continue

        if len(selected_qs) == 0:
            continue
        selected_qs = np.array(selected_qs)
        selected_m_q = np.array(selected_m_q)
        selected_m_t = np.array(selected_m_t)
        selected_c = np.array(selected_c)
        top_k = len(selected_qs)

    # Run rollouts for all selected contacts x all methods
    # all_results[contact_idx][method_label] = (m_q, m_t, c_d, q_traj)
    all_results = []
    for ci in range(top_k):
        qi = selected_qs[ci]
        print(f"\n--- Contact #{ci} (m_q={selected_m_q[ci]:.4f} m_t={selected_m_t[ci]:.4f} corner={selected_c[ci]:.4f}) ---")
        contact_results = {}
        for method, lr, ls_val, label, _, _ in METHODS:
            print(f"  Rolling out: {label}...")
            m_q, m_t, c_d, q_traj = rollout(qi, q_goal, n_steps, method, lam_rot=lr, lam_slide=ls_val, label=f"c{ci}/{label}")
            contact_results[label] = (m_q, m_t, c_d, q_traj)

            if not DEBUG:
                visualize_trajectory(q_traj, f"c{ci}/{label}")

        all_results.append(contact_results)

    # --- Plot 1: rollout curves per contact (one figure per contact) ---
    for ci in range(top_k):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for method, lr, ls_val, label, color, ls in METHODS:
            m_q, m_t, c_d, _ = all_results[ci][label]
            steps = np.arange(len(m_q))
            axes[0].plot(steps, m_q, label=label, color=color, linestyle=ls, linewidth=2)
            axes[1].plot(steps, m_t, label=label, color=color, linestyle=ls, linewidth=2)
            axes[2].plot(steps, c_d, label=label, color=color, linestyle=ls, linewidth=2)

        axes[0].set_title("Mahalanobis (7D quat)")
        axes[1].set_title("Mahalanobis (6D tangent)")
        axes[2].set_title("Corner distance")
        for ax in axes:
            ax.set_xlabel("Step")
            ax.set_ylabel("Distance")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Contact #{ci}  |  t: {start_t}->{end_t}  |  "
                     f"init m_q={selected_m_q[ci]:.2f} m_t={selected_m_t[ci]:.2f} corner={selected_c[ci]:.3f}",
                     fontsize=12)
        plt.tight_layout()
        fname = f"rollout_c{ci}_t{start_t:.2f}_t{end_t:.2f}_n{n_steps}.png"
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")

    # --- Plot 2: initial metric vs final corner distance (predictive power) ---
    if top_k >= 3:
        from scipy.stats import spearmanr

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for method, lr, ls_val, label, color, ls in METHODS:
            final_corners = []
            for ci in range(top_k):
                c_d = all_results[ci][label][2]
                final_corners.append(np.nanmin(c_d) if len(c_d) > 0 else np.nan)
            final_corners = np.array(final_corners)

            valid = np.isfinite(final_corners)
            if np.sum(valid) < 3:
                continue

            marker = 'o' if (lr > 0 and ls_val > 0) else ('x' if lr > 0 else 's')

            rho_q, p_q = spearmanr(selected_m_q[valid], final_corners[valid])
            axes[0].scatter(selected_m_q[valid], final_corners[valid],
                          label=f"{label} (ρ={rho_q:.2f})", color=color,
                          marker=marker, s=60)

            rho_t, p_t = spearmanr(selected_m_t[valid], final_corners[valid])
            axes[1].scatter(selected_m_t[valid], final_corners[valid],
                          label=f"{label} (ρ={rho_t:.2f})", color=color,
                          marker=marker, s=60)

        axes[0].set_xlabel("Initial Mahalanobis (7D quat)")
        axes[0].set_ylabel("Min corner distance")
        axes[0].set_title("7D quat metric as predictor")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Initial Mahalanobis (6D tangent)")
        axes[1].set_ylabel("Min corner distance")
        axes[1].set_title("6D tangent metric as predictor")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Metric predictive power  |  t: {start_t}->{end_t}, {n_steps} steps, {top_k} contacts",
                     fontsize=12)
        plt.tight_layout()
        fname = f"metric_prediction_t{start_t:.2f}_t{end_t:.2f}_n{n_steps}.png"
        plt.savefig(fname, dpi=150)
        print(f"Saved: {fname}")

    plt.show()
