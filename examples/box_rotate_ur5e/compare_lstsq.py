import numpy as np
from tqdm import tqdm
from pydrake.all import Quaternion

from box_rotate_setup import *
from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory

prob_rrt = IrsRrtTrajectory(
    rrt_params,
    contact_sampler,
    q_sim,
    q_sim_py,
    pose_sampling_function,
)

reachable_set = prob_rrt.reachable_set
reg = rrt_params.regularization


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
    """Convert 7-row unactuated Bhat to 6-row tangent-space Jacobian."""
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
    """Compute 6D error: log-map angular + Euclidean position."""
    dq = Quaternion(q_goal[idx_q_u[:4]]).multiply(
        Quaternion(chat[idx_q_u[:4]]).inverse()
    )
    e_ang = quat_log(dq.wxyz())
    e_pos = q_goal[idx_q_u[4:]] - chat[idx_q_u[4:]]
    return np.concatenate([e_ang, e_pos])


def compute_du_7x6(Bhat, chat, q_goal):
    A = Bhat[idx_q_u]
    b = (q_goal - chat)[idx_q_u]
    return constrained_least_squares(A, b, rrt_params.stepsize)


def compute_du_6x6(Bhat_tangent, chat, q_goal):
    b = tangent_error(chat, q_goal)
    return constrained_least_squares(Bhat_tangent, b, rrt_params.stepsize)


def compute_m_dists(Bhat, chat, q_goal, Bhat_tangent=None):
    """Compute both 7D and 6D Mahalanobis distances."""
    # 7D quat-space
    Bhat_u = Bhat[idx_q_u]
    cov_u = Bhat_u @ Bhat_u.T + reg * np.eye(len(idx_q_u))
    error_7 = q_goal[idx_q_u] - chat[idx_q_u]
    d_quat = error_7 @ np.linalg.inv(cov_u) @ error_7

    # 6D tangent-space
    if Bhat_tangent is None:
        Bhat_tangent = bhat_to_tangent(Bhat, chat)
    cov_t = Bhat_tangent @ Bhat_tangent.T + reg * np.eye(6)
    error_6 = tangent_error(chat, q_goal)
    d_tangent = error_6 @ np.linalg.inv(cov_t) @ error_6

    return d_quat, d_tangent


N = 200
eps = 0.1
np.set_printoptions(suppress=True, precision=4)

# --- Phase 1: sample contact configurations (sequential) ---
print("Sampling contact configurations...")
q_batch_list = []
q_goal_list = []
for i in tqdm(range(N)):
    t0 = np.random.uniform(0, 1 - eps)
    t1 = np.random.uniform(t0 + eps, 1)

    q_start = np.zeros(dim_x)
    q_start[idx_q_u] = get_obj_pose_from_t(t0)
    q_goal = np.zeros(dim_x)
    q_goal[idx_q_u] = get_obj_pose_from_t(t1)

    try:
        qi = contact_sampler.sample_contact(q_start)
    except Exception:
        continue

    q_batch_list.append(qi)
    q_goal_list.append(q_goal)

q_batch = np.array(q_batch_list)
q_goal_batch = np.array(q_goal_list)
n_valid = len(q_batch)
print(f"Sampled {n_valid} valid contact configurations")

# --- Phase 2: batched Bc computation ---
print("Computing bundled dynamics (batched)...")
u_batch = q_batch[:, idx_q_a]
B_batch, chat_batch, is_valid = reachable_set.calc_bundled_Bc_analytic_batch(
    q_batch, u_batch
)

valid_mask = np.array(is_valid)
q_batch = q_batch[valid_mask]
q_goal_batch = q_goal_batch[valid_mask]
B_batch = np.array(B_batch)[valid_mask]
chat_batch = np.array(chat_batch)[valid_mask]
u_batch = u_batch[valid_mask]
n_valid = len(q_batch)
print(f"{n_valid} valid after Bc computation")

# --- Phase 3: precompute tangent Bhats + compute du for both methods ---
print("Computing tangent Bhats and du* for both methods...")
Bt_batch = np.zeros((n_valid, 6, len(idx_q_a)))
du_batch_7 = np.zeros((n_valid, len(idx_q_a)))
du_batch_6 = np.zeros((n_valid, len(idx_q_a)))
du_valid = np.ones(n_valid, dtype=bool)

for i in range(n_valid):
    try:
        Bt_batch[i] = bhat_to_tangent(B_batch[i], chat_batch[i])
        du_batch_7[i] = compute_du_7x6(B_batch[i], chat_batch[i], q_goal_batch[i])
        du_batch_6[i] = compute_du_6x6(Bt_batch[i], chat_batch[i], q_goal_batch[i])
    except Exception:
        du_valid[i] = False

q_batch = q_batch[du_valid]
q_goal_batch = q_goal_batch[du_valid]
B_batch = B_batch[du_valid]
Bt_batch = Bt_batch[du_valid]
chat_batch = chat_batch[du_valid]
u_batch = u_batch[du_valid]
du_batch_7 = du_batch_7[du_valid]
du_batch_6 = du_batch_6[du_valid]
n_valid = len(q_batch)
print(f"{n_valid} valid after du computation")

# --- Phase 4: "before" Mahalanobis distances (reuse B_batch/Bt_batch/chat_batch) ---
print("Computing 'before' Mahalanobis distances...")
m_before_q = np.full(n_valid, np.nan)
m_before_t = np.full(n_valid, np.nan)
for i in range(n_valid):
    m_before_q[i], m_before_t[i] = compute_m_dists(
        B_batch[i], chat_batch[i], q_goal_batch[i], Bt_batch[i]
    )

# --- Phase 5: batched forward dynamics for both methods ---
print("Stepping forward (batched)...")
u_star_7 = u_batch + du_batch_7
u_star_6 = u_batch + du_batch_6

q_next_7, _, _, is_valid_7 = reachable_set.q_sim_batch.calc_dynamics_parallel(
    q_batch, u_star_7, reachable_set.sim_params
)
q_next_6, _, _, is_valid_6 = reachable_set.q_sim_batch.calc_dynamics_parallel(
    q_batch, u_star_6, reachable_set.sim_params
)

q_next_7 = np.array(q_next_7)
q_next_6 = np.array(q_next_6)
is_valid_7 = np.array(is_valid_7)
is_valid_6 = np.array(is_valid_6)

both_valid = is_valid_7 & is_valid_6
q_batch = q_batch[both_valid]
q_goal_batch = q_goal_batch[both_valid]
q_next_7 = q_next_7[both_valid]
q_next_6 = q_next_6[both_valid]
m_before_q = m_before_q[both_valid]
m_before_t = m_before_t[both_valid]
n_valid = len(q_batch)
print(f"{n_valid} valid after stepping")

# --- Phase 6: "after" Mahalanobis distances (need new Bc at q_next) ---
print("Computing 'after' Mahalanobis distances (batched Bc)...")

def get_m_dist_batch_after(q_from_batch, q_goal_batch):
    n = len(q_from_batch)
    u_from = q_from_batch[:, idx_q_a]
    B_b, chat_b, is_v = reachable_set.calc_bundled_Bc_analytic_batch(
        q_from_batch, u_from
    )
    B_b = np.array(B_b)
    chat_b = np.array(chat_b)
    is_v = np.array(is_v)

    dists_q = np.full(n, np.nan)
    dists_t = np.full(n, np.nan)
    for i in range(n):
        if not is_v[i]:
            continue
        Bt = bhat_to_tangent(B_b[i], chat_b[i])
        dists_q[i], dists_t[i] = compute_m_dists(
            B_b[i], chat_b[i], q_goal_batch[i], Bt
        )
    return dists_q, dists_t


m_after_7_q, m_after_7_t = get_m_dist_batch_after(q_next_7, q_goal_batch)
m_after_6_q, m_after_6_t = get_m_dist_batch_after(q_next_6, q_goal_batch)

# --- Filter valid ---
all_finite = (
    np.isfinite(m_before_q) & np.isfinite(m_after_7_q) & np.isfinite(m_after_6_q) &
    np.isfinite(m_before_t) & np.isfinite(m_after_7_t) & np.isfinite(m_after_6_t)
)
m_before_q = m_before_q[all_finite]
m_after_7_q = m_after_7_q[all_finite]
m_after_6_q = m_after_6_q[all_finite]
m_before_t = m_before_t[all_finite]
m_after_7_t = m_after_7_t[all_finite]
m_after_6_t = m_after_6_t[all_finite]


def print_results(label, before, after_7, after_6):
    dec_7 = before - after_7
    dec_6 = before - after_6
    n = len(before)

    print(f"\n  {label}:")
    print(f"    Decrease (higher = better):")
    print(f"      7x6 (naive quat):  mean={dec_7.mean():.4f}  median={np.median(dec_7):.4f}  std={dec_7.std():.4f}")
    print(f"      6x6 (tangent):     mean={dec_6.mean():.4f}  median={np.median(dec_6):.4f}  std={dec_6.std():.4f}")

    wins_6 = np.sum(dec_6 > dec_7)
    wins_7 = np.sum(dec_7 > dec_6)
    print(f"    6x6 wins: {wins_6}/{n} ({100*wins_6/n:.1f}%)  |  7x6 wins: {wins_7}/{n} ({100*wins_7/n:.1f}%)")

    neg_7 = np.sum(dec_7 < 0)
    neg_6 = np.sum(dec_6 < 0)
    print(f"    Steps that increased distance:  7x6: {neg_7}/{n}  |  6x6: {neg_6}/{n}")

    imp = dec_6 - dec_7
    print(f"    6x6 improvement over 7x6:  mean={imp.mean():.4f}  median={np.median(imp):.4f}")

    return dec_7, dec_6


n_all = len(m_before_q)
print(f"\n{'='*60}")
print(f"ALL TRIALS: {n_all}")
dec_7_q, dec_6_q = print_results("Quat-space Mahalanobis (7D)", m_before_q, m_after_7_q, m_after_6_q)
dec_7_t, dec_6_t = print_results("Tangent-space Mahalanobis (6D)", m_before_t, m_after_7_t, m_after_6_t)

# --- Filtered: both methods decreased distance under BOTH metrics ---
both_positive = (dec_7_q > 0) & (dec_6_q > 0) & (dec_7_t > 0) & (dec_6_t > 0)
n_pos = np.sum(both_positive)

print(f"\n{'='*60}")
print(f"FILTERED (both methods decreased distance in both metrics): {n_pos}/{n_all}")
print_results("Quat-space Mahalanobis (7D)", m_before_q[both_positive], m_after_7_q[both_positive], m_after_6_q[both_positive])
print_results("Tangent-space Mahalanobis (6D)", m_before_t[both_positive], m_after_7_t[both_positive], m_after_6_t[both_positive])
