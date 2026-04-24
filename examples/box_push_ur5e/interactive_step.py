import numpy as np
from box_push_setup import *

from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory

prob_rrt = IrsRrtTrajectory(
    rrt_params,
    contact_sampler,
    q_sim,
    q_sim_py,
    pose_sampling_function,
)

np.set_printoptions(suppress=True, precision=4)


def constrained_least_squares(A, b, d, tol=1e-10, max_iter=100):
    """Solve min ||Ax - b|| subject to ||x|| <= d."""
    AtA = A.T @ A
    Atb = A.T @ b

    try:
        x0 = np.linalg.solve(AtA, Atb)
        if np.linalg.norm(x0) <= d:
            return x0
    except np.linalg.LinAlgError:
        pass

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

q_goal = np.zeros(dim_x)
q_goal[idx_q_u] = get_obj_pose_from_t(1.0)
rrt_params.goal = q_goal

qi = None


def sample_contact(t=0.0):
    """Sample a configuration in contact at object pose t."""
    global qi
    q_start = np.zeros(dim_x)
    q_start[idx_q_u] = get_obj_pose_from_t(t)
    qi = contact_sampler.sample_contact(q_start)
    q_vis.draw_configuration(qi)
    print(f"Sampled contact config at t={t}")
    _print_state()
    return qi


def sample_random():
    """Sample a random configuration within joint limits."""
    global qi
    qi = np.random.rand(dim_x)
    qi = prob_rrt.q_lb + (prob_rrt.q_ub - prob_rrt.q_lb) * qi
    qi[idx_q_u] = get_obj_pose_from_t(0.0)
    q_vis.draw_configuration(qi)
    print("Sampled random config")
    _print_state()
    return qi


def step_gradient(stepsize=None):
    """Take a step in the gradient direction toward q_goal."""
    global qi
    if qi is None:
        print("No config set. Use sample_contact() or sample_random() first.")
        return

    if stepsize is None:
        stepsize = rrt_params.stepsize

    try:
        Bhat, chat = prob_rrt.reachable_set.calc_bundled_Bc_analytic(
            qi, qi[idx_q_a]
        )

        x = constrained_least_squares(
            Bhat[idx_q_u], (q_goal - chat)[idx_q_u], stepsize
        )

        u_star = qi[idx_q_a] + x
        q_next = q_sim.calc_dynamics(qi, u_star, prob_rrt.sim_params)

        du_actual = q_next[idx_q_a] - qi[idx_q_a]
        print(f"du commanded: {x}")
        print(f"du actual:    {du_actual}")
        print(f"du diff:      {du_actual - x}")

        qi = q_next
        q_vis.draw_configuration(qi)
        _print_state()
    except Exception as e:
        print(f"Step failed: {e}")

    return qi


def step_random(stepsize=None):
    """Take a step in a random direction."""
    global qi
    if qi is None:
        print("No config set. Use sample_contact() or sample_random() first.")
        return

    if stepsize is None:
        stepsize = rrt_params.stepsize

    try:
        du = np.random.randn(len(idx_q_a))
        du = du / np.linalg.norm(du) * stepsize

        u_star = qi[idx_q_a] + du
        q_next = q_sim.calc_dynamics(qi, u_star, prob_rrt.sim_params)

        du_actual = q_next[idx_q_a] - qi[idx_q_a]
        print(f"du commanded: {du}")
        print(f"du actual:    {du_actual}")
        print(f"du diff:      {du_actual - du}")

        qi = q_next
        q_vis.draw_configuration(qi)
        _print_state()
    except Exception as e:
        print(f"Step failed: {e}")

    return qi


def _print_state():
    if qi is None:
        return
    try:
        t, dist = prob_rrt.closest_t_in_trajectory(qi[idx_q_u])
        is_static = prob_rrt.is_static(qi)
        print(f"  closest t: {t:.3f} (dist={dist:.4f})")
        print(f"  is_static: {is_static}")
        print(f"  q_a: {qi[idx_q_a]}")
        print(f"  q_u: {qi[idx_q_u]}")
    except Exception as e:
        print(f"  (state info failed: {e})")


print("""
Interactive stepping tool
=========================
Commands:
  sample_contact(t=0.0)      - sample config in contact at pose t
  sample_random()             - sample random joint config
  step_gradient(stepsize)     - step toward goal using bundled dynamics
  step_random(stepsize)       - step in random direction

stepsize defaults to rrt_params.stepsize if omitted.
""")
