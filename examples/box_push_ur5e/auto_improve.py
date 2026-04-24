"""
Adaptive iMPC refinement tuning for box_push_ur5e.

Phase 1: Run a small set of 1-parameter probes to understand sensitivities.
Phase 2: Use results to generate promising combinations and perturbations
         around the best config found so far.

The log file is written incrementally as JSONL so you can tail -f it.

Usage:
    python auto_improve.py ptc_data/box_push_ur5e/traj_20260412_145454.npz \
        --duration_minutes 30

Monitor:
    tail -f auto_improve_log.json
"""
import argparse
import copy
import json
import os
import time
import numpy as np

from qsim_cpp import ForwardDynamicsMode
from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.irs_mpc import IrsMpcQuasistatic
from irs_mpc2.irs_mpc_params import SmoothingMode, IrsMpcQuasistaticParameters

from box_push_setup import (
    q_sim, q_sim_py, q_parser, idx_q_u, idx_q_a, idx_u, idx_a,
    rrt_params, pose_sampling_function,
)
from evaluate_trajectory import (
    rollout_trajectory, compute_metrics, compute_score,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "ptc_data", "box_push_ur5e")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("traj_file_path", type=str)
parser.add_argument(
    "--duration_minutes", type=float, default=30,
    help="Total wall-clock time to run (minutes).",
)
parser.add_argument(
    "--log_file", type=str, default="auto_improve_log.json",
    help="Path to JSONL log file (appended incrementally).",
)
args = parser.parse_args()

# --------------------------------------------------------------------------
# Load source trajectory
# --------------------------------------------------------------------------
data = np.load(args.traj_file_path)
q_knots_original = data["q_trj"]
u_knots_original = data["u_trj"]
h_planner = float(data["h"])

segments = IrsRrt.get_regrasp_segments(u_knots_original)
print(f"Loaded {len(q_knots_original)} knots, {len(segments)} segments")

# Extract timestamp from source filename for compatible output naming
src_basename = os.path.basename(args.traj_file_path)
src_ts = src_basename.replace("traj_", "").replace(".npz", "")

dim_u = q_sim.num_actuated_dofs()

# Sim params
sim_params_planner = copy.deepcopy(q_sim.get_sim_params())
sim_params_planner.h = h_planner

sim_params_projection = copy.deepcopy(sim_params_planner)
sim_params_projection.unactuated_mass_scale = 1e-4


def project_to_non_penetration(q):
    return q_sim.calc_dynamics(q, q[idx_q_a], sim_params_projection)


def linear_upsample(u_trj_coarse, n_steps_per_h):
    T0, d = u_trj_coarse.shape
    T_fine = T0 * n_steps_per_h
    t_fine = np.arange(T_fine) / n_steps_per_h
    i_lo = np.clip(np.floor(t_fine).astype(int), 0, T0 - 1)
    i_hi = np.clip(i_lo + 1, 0, T0 - 1)
    alpha = (t_fine - i_lo)[:, None]
    return (1 - alpha) * u_trj_coarse[i_lo] + alpha * u_trj_coarse[i_hi]


# --------------------------------------------------------------------------
# Core: run one refinement with given config
# --------------------------------------------------------------------------
def run_refinement(config):
    """
    Run iMPC refinement with the given config dict.
    Returns (u_trj_refined, q_trj_refined, metrics, elapsed_seconds).
    """
    h_small = config["h_small"]
    max_iterations = config["max_iterations"]
    output_hz = config["output_hz"]

    inner_hz = int(round(1.0 / h_small))
    h_output = 1.0 / output_hz
    inner_to_output_ds = inner_hz // output_hz

    params = IrsMpcQuasistaticParameters()
    params.h = h_small

    params.Q_dict = {
        idx_u: np.array(config["Q_obj"]),
        idx_a: np.ones(dim_u) * config["Q_arm"],
    }

    params.Qd_dict = {}
    for model in q_sim.get_actuated_models():
        params.Qd_dict[model] = params.Q_dict[model]
    for model in q_sim.get_unactuated_models():
        params.Qd_dict[model] = params.Q_dict[model] * config["Qd_multiplier"]

    params.R_dict = {idx_a: np.array(config["R"])}

    u_size = config["u_size"]
    params.u_bounds_abs = np.array([
        -np.ones(dim_u) * u_size * params.h,
        np.ones(dim_u) * u_size * params.h,
    ])

    params.smoothing_mode = SmoothingMode.k1AnalyticIcecream
    params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
    params.std_u_initial = np.ones(dim_u) * 0.3
    params.num_samples = 100

    params.log_barrier_weight_initial = config["log_barrier_initial"]
    log_barrier_final = config["log_barrier_final"]
    base = np.exp(
        np.log(log_barrier_final / params.log_barrier_weight_initial)
        / max(max_iterations, 1)
    )
    params.calc_log_barrier_weight = lambda kappa0, i: kappa0 * (base ** i)

    params.use_A = False
    params.rollout_forward_dynamics_mode = ForwardDynamicsMode.kSocpMp

    prob_mpc = IrsMpcQuasistatic(q_sim=q_sim, parser=q_parser, params=params)

    q_trj_refined_list = []
    u_trj_refined_list = []
    cost_curves = []

    t_start_wall = time.time()

    for i_s, (t_start, t_end) in enumerate(segments):
        u_trj_seg = u_knots_original[t_start:t_end]
        q_trj_seg = q_knots_original[t_start:t_end]

        if len(u_trj_seg) == 0:
            continue

        n_steps_per_h_for_rate = int(round(h_planner / h_small))
        n_steps_per_h_for_length = max(1, int(np.ceil(10 / max(len(u_trj_seg), 1))))
        n_steps_per_h = max(n_steps_per_h_for_rate, n_steps_per_h_for_length)

        q0 = np.array(q_trj_seg[0])
        if len(q_trj_refined_list) > 0:
            q0[idx_q_u] = q_trj_refined_list[-1][-1, idx_q_u]
            q0 = project_to_non_penetration(q0)

        q_last = np.array(q_trj_seg[-1])
        q_final = q_sim.calc_dynamics(q_last, u_trj_seg[-1], sim_params_planner)
        if i_s == len(segments) - 1:
            q_final[idx_q_u] = rrt_params.goal[idx_q_u]

        u_trj_small = linear_upsample(u_trj_seg, n_steps_per_h)

        q_d = np.copy(q0)
        q_d[idx_q_u] = q_final[idx_q_u]
        T_inner = len(u_trj_small)
        q_trj_d = np.tile(q_d, (T_inner + 1, 1))

        prob_mpc.initialize_problem(x0=q0, x_trj_d=q_trj_d, u_trj_0=u_trj_small)
        prob_mpc.iterate(max_iterations=max_iterations, cost_Qu_f_threshold=0)

        q_trj_opt = np.array(prob_mpc.x_trj_best)
        u_trj_opt = np.array(prob_mpc.u_trj_best)

        q_trj_down = q_trj_opt[::inner_to_output_ds]
        u_trj_down = u_trj_opt[::inner_to_output_ds]

        q_trj_refined_list.append(q_trj_down)
        u_trj_refined_list.append(u_trj_down)

        # Capture cost curve for this segment
        cost_curves.append({
            "costs": [float(c) for c in prob_mpc.cost_all_list],
            "cost_Qu": [float(c) for c in prob_mpc.cost_Qu_list],
            "cost_Qu_final": [float(c) for c in prob_mpc.cost_Qu_final_list],
            "cost_R": [float(c) for c in prob_mpc.cost_R_list],
            "idx_best": int(prob_mpc.idx_best),
        })

    elapsed = time.time() - t_start_wall

    # Concatenate
    q_trj_parts = []
    u_trj_parts = []
    for i, (q_seg, u_seg) in enumerate(zip(q_trj_refined_list, u_trj_refined_list)):
        q_trj_parts.append(q_seg[:-1])
        u_trj_parts.append(u_seg)
        if i < len(q_trj_refined_list) - 1:
            q_trj_parts.append(q_seg[-1:])
            u_trj_parts.append(np.full((1, dim_u), np.nan))

    q_trj_save = np.concatenate(q_trj_parts, axis=0)
    u_trj_save = np.concatenate(u_trj_parts, axis=0)

    # Evaluate
    q_trj_rerolled = rollout_trajectory(u_trj_save, q_trj_save[0], q_sim, 1.0 / output_hz)
    metrics = compute_metrics(u_trj_save, q_trj_rerolled, 1.0 / output_hz, idx_q_u, idx_q_a)
    metrics["score"] = compute_score(metrics)

    # Summarize cost curves: convergence behavior
    for i, cc in enumerate(cost_curves):
        costs = cc["costs"]
        n = len(costs)
        if n >= 2:
            # How much improvement in first half vs second half
            mid = n // 2
            first_half_drop = costs[0] - costs[mid]
            second_half_drop = costs[mid] - costs[-1]
            # Fraction of improvement that happened in second half
            total_drop = costs[0] - costs[-1]
            late_improvement_frac = second_half_drop / max(total_drop, 1e-10)
        else:
            late_improvement_frac = 0
            total_drop = 0

        cc["late_improvement_frac"] = late_improvement_frac
        cc["total_drop"] = total_drop
        cc["final_cost"] = costs[-1] if costs else 0

    metrics["cost_curves"] = cost_curves

    return u_trj_save, q_trj_save, metrics, elapsed


# --------------------------------------------------------------------------
# Baseline config
# --------------------------------------------------------------------------
BASE_CONFIG = {
    "h_small": 0.01,
    "max_iterations": 20,
    "output_hz": 50,
    "Q_obj": [1, 1, 1, 1, 50, 50, 5],
    "Q_arm": 1e-3,
    "Qd_multiplier": 200,
    "R": [100.0] * 6,
    "u_size": 10.0,
    "log_barrier_initial": 100,
    "log_barrier_final": 6000,
}

# Parameters we can tune and their reasonable ranges (log-scale where noted)
PARAM_RANGES = {
    "R_uniform":          (10.0, 2000.0, "log"),    # uniform R for all joints
    "R_wrist_ratio":      (1.0, 20.0, "log"),       # wrist R = base R * ratio
    "Q_pos":              (5.0, 500.0, "log"),       # Q weight on box x,y
    "Qd_multiplier":      (20.0, 2000.0, "log"),     # terminal cost multiplier
    "u_size":             (2.0, 30.0, "log"),         # trust region
    "max_iterations":     (5, 50, "linear_int"),
    "log_barrier_initial": (20.0, 500.0, "log"),
    "log_barrier_final":  (1000.0, 20000.0, "log"),
}


def config_from_params(p: dict) -> tuple:
    """Convert a flat param dict to (description, config)."""
    c = copy.deepcopy(BASE_CONFIG)

    r_base = p.get("R_uniform", 100.0)
    r_wrist_ratio = p.get("R_wrist_ratio", 1.0)
    c["R"] = [r_base] * 3 + [r_base * r_wrist_ratio] * 3

    q_pos = p.get("Q_pos", 50.0)
    c["Q_obj"] = [1, 1, 1, 1, q_pos, q_pos, 5]

    c["Qd_multiplier"] = p.get("Qd_multiplier", 200.0)
    c["u_size"] = p.get("u_size", 10.0)
    c["max_iterations"] = int(p.get("max_iterations", 20))
    c["log_barrier_initial"] = p.get("log_barrier_initial", 100.0)
    c["log_barrier_final"] = p.get("log_barrier_final", 6000.0)

    parts = []
    for k, v in sorted(p.items()):
        if isinstance(v, float):
            parts.append(f"{k}={v:.1f}")
        else:
            parts.append(f"{k}={v}")
    desc = " ".join(parts)

    return desc, c


def sample_in_range(lo, hi, scale):
    """Sample a value uniformly in the given range/scale."""
    if scale == "log":
        return np.exp(np.random.uniform(np.log(lo), np.log(hi)))
    elif scale == "linear_int":
        return int(np.random.uniform(lo, hi))
    else:
        return np.random.uniform(lo, hi)


def perturb_params(p: dict, strength=0.3) -> dict:
    """
    Perturb a param dict. strength controls how far from the current
    value we explore (0 = no change, 1 = full range).
    """
    p_new = copy.deepcopy(p)
    # Pick 1-3 params to perturb
    keys = list(p.keys())
    n_perturb = np.random.randint(1, min(4, len(keys) + 1))
    to_perturb = np.random.choice(keys, size=n_perturb, replace=False)

    for key in to_perturb:
        lo, hi, scale = PARAM_RANGES[key]
        current = p[key]

        if scale == "log":
            log_cur = np.log(current)
            log_lo, log_hi = np.log(lo), np.log(hi)
            span = log_hi - log_lo
            noise = np.random.normal(0, strength * span * 0.3)
            new_val = np.exp(np.clip(log_cur + noise, log_lo, log_hi))
            p_new[key] = float(new_val)
        elif scale == "linear_int":
            span = hi - lo
            noise = np.random.normal(0, strength * span * 0.3)
            p_new[key] = int(np.clip(current + noise, lo, hi))
        else:
            span = hi - lo
            noise = np.random.normal(0, strength * span * 0.3)
            p_new[key] = float(np.clip(current + noise, lo, hi))

    return p_new


def decide_next(all_results, best_params, param_ranges, base_params):
    """
    Look at all results so far and decide what to try next.
    Returns (strategy_name, params_dict, reasoning_string).
    """
    n = len(all_results)
    valid = [(p, s, m) for p, s, m in all_results if s < 1e6]
    if not valid:
        return "random", random_params(), "no valid results yet, exploring randomly"

    valid_sorted = sorted(valid, key=lambda x: x[1])
    best_p, best_s, best_m = valid_sorted[0]

    # Look at cost curves from the best result to understand convergence
    cost_curves = best_m.get("cost_curves", [])
    still_improving = False
    for cc in cost_curves:
        if cc.get("late_improvement_frac", 0) > 0.2:
            still_improving = True
            break

    # Check if accuracy or smoothness is the bottleneck
    pos_err = best_m.get("goal_pos_error", 0)
    rms_d2u = best_m.get("rms_d2u", 0)
    max_d2u = best_m.get("max_d2u", 0)

    accuracy_limited = pos_err > 0.03  # > 3cm is still too far
    smoothness_limited = max_d2u > 0.015  # jerky peaks

    # Check which parameters correlated with improvement
    # (simple: compare top 3 vs bottom 3 configs)
    if len(valid_sorted) >= 6:
        top3 = valid_sorted[:3]
        bot3 = valid_sorted[-3:]
        param_hints = {}
        for key in param_ranges:
            top_avg = np.mean([p.get(key, base_params[key]) for p, _, _ in top3])
            bot_avg = np.mean([p.get(key, base_params[key]) for p, _, _ in bot3])
            if abs(top_avg - bot_avg) > 1e-6:
                param_hints[key] = ("higher" if top_avg > bot_avg else "lower",
                                    top_avg, bot_avg)
    else:
        param_hints = {}

    # --- Decision logic based on observations ---

    # If optimizer was still improving at the end, try more iterations
    if still_improving and best_p.get("max_iterations", 20) < 45:
        p = copy.deepcopy(best_p)
        p["max_iterations"] = min(int(p.get("max_iterations", 20) * 1.5), 50)
        return "more_iters", p, (
            f"cost still dropping in late iterations "
            f"(late_improvement_frac > 0.2), increasing iterations to {p['max_iterations']}"
        )

    # If accuracy is the bottleneck, boost Q/Qd
    if accuracy_limited and np.random.rand() < 0.5:
        p = copy.deepcopy(best_p)
        if np.random.rand() < 0.5:
            p["Q_pos"] = min(p.get("Q_pos", 50) * 1.5, 500)
            return "boost_Q", p, (
                f"pos_err={pos_err:.4f} > 0.03, increasing Q_pos to {p['Q_pos']:.0f} "
                f"to push harder toward goal"
            )
        else:
            p["Qd_multiplier"] = min(p.get("Qd_multiplier", 200) * 1.5, 2000)
            return "boost_Qd", p, (
                f"pos_err={pos_err:.4f} > 0.03, increasing Qd_multiplier to "
                f"{p['Qd_multiplier']:.0f} for stronger terminal cost"
            )

    # If smoothness is the bottleneck, boost R
    if smoothness_limited and np.random.rand() < 0.5:
        p = copy.deepcopy(best_p)
        # Check which joints are worst
        max_d2u_joints = best_m.get("max_d2u_per_joint", [0] * 6)
        wrist_jerky = any(max_d2u_joints[i] > 0.01 for i in range(3, min(6, len(max_d2u_joints))))
        if wrist_jerky:
            p["R_wrist_ratio"] = min(p.get("R_wrist_ratio", 1) * 2, 20)
            return "boost_R_wrist", p, (
                f"max_d2u={max_d2u:.4f}, wrist joints jerky "
                f"({[f'{x:.4f}' for x in max_d2u_joints[3:]]}), "
                f"increasing wrist R ratio to {p['R_wrist_ratio']:.1f}"
            )
        else:
            p["R_uniform"] = min(p.get("R_uniform", 100) * 1.5, 2000)
            return "boost_R", p, (
                f"max_d2u={max_d2u:.4f}, increasing R_uniform to {p['R_uniform']:.0f}"
            )

    # Follow param_hints from correlation analysis
    if param_hints and np.random.rand() < 0.4:
        key = np.random.choice(list(param_hints.keys()))
        direction, top_avg, bot_avg = param_hints[key]
        p = copy.deepcopy(best_p)
        lo, hi, scale = param_ranges[key]
        cur = p.get(key, base_params[key])
        if direction == "higher":
            if scale == "log":
                p[key] = min(cur * 1.4, hi)
            else:
                p[key] = min(cur * 1.3, hi)
        else:
            if scale == "log":
                p[key] = max(cur / 1.4, lo)
            else:
                p[key] = max(cur / 1.3, lo)
        if scale == "linear_int":
            p[key] = int(p[key])
        return "follow_hint", p, (
            f"top configs tend to have {direction} {key} "
            f"(top avg={top_avg:.1f} vs bot avg={bot_avg:.1f}), "
            f"moving {key} to {p[key]}"
        )

    # Crossover between good configs
    if len(valid_sorted) >= 4 and np.random.rand() < 0.3:
        idx1 = np.random.randint(0, min(3, len(valid_sorted)))
        idx2 = np.random.randint(0, min(5, len(valid_sorted)))
        p1 = valid_sorted[idx1][0]
        p2 = valid_sorted[idx2][0]
        p = {}
        for key in param_ranges:
            if np.random.rand() < 0.5:
                p[key] = p1.get(key, base_params[key])
            else:
                p[key] = p2.get(key, base_params[key])
        return "crossover", p, (
            f"blending rank {idx1+1} and rank {idx2+1} configs"
        )

    # Default: perturb best with adaptive strength
    # More stagnation = wider perturbation
    recent_scores = [s for _, s, _ in all_results[-5:] if s < 1e6]
    if recent_scores and all(s >= best_s for s in recent_scores):
        strength = 0.5  # stuck, explore wider
        reason = "recent runs haven't improved, exploring wider"
    else:
        strength = 0.25
        reason = "perturbing best config"

    p = perturb_params(best_p, strength)
    return "perturb", p, reason


def random_params() -> dict:
    """Sample completely random params."""
    p = {}
    for key, (lo, hi, scale) in PARAM_RANGES.items():
        p[key] = sample_in_range(lo, hi, scale)
    return p


def params_from_config(config: dict) -> dict:
    """Extract the flat param dict from a config (inverse of config_from_params)."""
    r = config["R"]
    r_base = r[0]
    r_wrist = r[3] if len(r) > 3 else r[0]

    return {
        "R_uniform": r_base,
        "R_wrist_ratio": r_wrist / max(r_base, 1e-10),
        "Q_pos": config["Q_obj"][4],
        "Qd_multiplier": config["Qd_multiplier"],
        "u_size": config["u_size"],
        "max_iterations": config["max_iterations"],
        "log_barrier_initial": config["log_barrier_initial"],
        "log_barrier_final": config["log_barrier_final"],
    }


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def log_result(log_file, run_id, description, config, metrics, elapsed, phase=""):
    entry = {
        "run_id": run_id,
        "phase": phase,
        "description": description,
        "config": config,
        "metrics": metrics,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%H:%M:%S"),
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    score_str = f"{metrics['score']:8.4f}" if metrics['score'] < 1e6 else "FAILED"
    print(
        f"\n>>> Run {run_id:3d} [{phase:>10s}] | {description:<45s} | "
        f"score={score_str} | "
        f"rms_d2u={metrics.get('rms_d2u', 0):8.5f} | "
        f"max_d2u={metrics.get('max_d2u', 0):8.5f} | "
        f"pos_err={metrics.get('goal_pos_error', 0):8.5f} | "
        f"time={elapsed:.0f}s",
        flush=True,
    )


def load_log(log_file):
    """Load all completed runs from the log."""
    try:
        with open(log_file) as f:
            return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return []


def print_leaderboard(log_file):
    runs = load_log(log_file)
    valid = [r for r in runs if r["metrics"].get("score", float("inf")) < 1e6]
    valid.sort(key=lambda r: r["metrics"]["score"])
    print("\n" + "=" * 100, flush=True)
    print("LEADERBOARD (top 5)", flush=True)
    print(f"{'Rank':<5} {'Run':<5} {'Phase':<12} {'Score':>8} {'rms_d2u':>10} "
          f"{'max_d2u':>10} {'pos_err':>10} {'Description'}", flush=True)
    print("-" * 100, flush=True)
    for i, r in enumerate(valid[:5]):
        m = r["metrics"]
        print(
            f"{i+1:<5} {r['run_id']:<5} {r.get('phase',''):<12} "
            f"{m['score']:8.4f} {m.get('rms_d2u',0):10.5f} "
            f"{m.get('max_d2u',0):10.5f} {m.get('goal_pos_error',0):10.5f} "
            f"{r['description'][:50]}", flush=True
        )
    print("=" * 100 + "\n", flush=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Clear old log
    if os.path.exists(args.log_file):
        os.rename(args.log_file, args.log_file + ".bak")

    deadline = time.time() + args.duration_minutes * 60
    run_id = 0

    best_score = float("inf")
    best_params = params_from_config(BASE_CONFIG)
    best_config = BASE_CONFIG

    all_results = []  # (params_dict, score, metrics)

    # ------------------------------------------------------------------
    # Phase 1: Probe — one-at-a-time parameter sweeps (3 values each)
    # to learn which parameters matter most.
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print("PHASE 1: Probing parameter sensitivities", flush=True)
    print("=" * 60, flush=True)

    probe_configs = []

    # Baseline first
    probe_configs.append(("baseline", BASE_CONFIG, params_from_config(BASE_CONFIG)))

    # For each tunable param, try low and high
    base_params = params_from_config(BASE_CONFIG)
    for key, (lo, hi, scale) in PARAM_RANGES.items():
        for frac in [0.15, 0.85]:  # low-ish and high-ish within range
            p = copy.deepcopy(base_params)
            if scale == "log":
                p[key] = np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo)))
            elif scale == "linear_int":
                p[key] = int(lo + frac * (hi - lo))
            else:
                p[key] = lo + frac * (hi - lo)
            desc, config = config_from_params(p)
            probe_configs.append((f"probe {key}={p[key]:.1f}", config, p))

    for desc, config, p in probe_configs:
        if time.time() >= deadline:
            break

        remaining = (deadline - time.time()) / 60
        print(f"\n--- Run {run_id}: {desc} ({remaining:.1f} min left) ---", flush=True)

        try:
            u_ref, q_ref, metrics, elapsed = run_refinement(config)
            log_result(args.log_file, run_id, desc, config, metrics, elapsed, "probe")
            all_results.append((p, metrics["score"], metrics))

            if metrics["score"] < best_score:
                best_score = metrics["score"]
                best_params = copy.deepcopy(p)
                best_config = copy.deepcopy(config)
                for save_name in ["traj_refined_best.npz",
                                  f"traj_refined_{src_ts}.npz"]:
                    np.savez_compressed(
                        os.path.join(DATA_DIR, save_name),
                        q_trj=q_ref, u_trj=u_ref,
                        h=1.0 / config["output_hz"],
                        q_u_indices_into_x=idx_q_u,
                        q_a_indices_into_x=idx_q_a,
                    )
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            log_result(args.log_file, run_id, f"{desc} (FAIL)",
                       config, {"score": float("inf")}, 0, "probe")
            all_results.append((p, float("inf"), {}))

        run_id += 1

    print_leaderboard(args.log_file)

    # ------------------------------------------------------------------
    # Analyze phase 1: which parameters had the most impact?
    # ------------------------------------------------------------------
    if len(all_results) >= 3:
        baseline_score = all_results[0][1]
        sensitivities = {}
        for key in PARAM_RANGES:
            # Find probe runs for this param
            relevant = [(p, s) for p, s, _ in all_results[1:]
                        if abs(p.get(key, base_params[key]) - base_params[key])
                        > 1e-6 * abs(base_params[key])]
            if relevant:
                impact = max(abs(s - baseline_score) for _, s in relevant)
                sensitivities[key] = impact

        if sensitivities:
            sorted_sens = sorted(sensitivities.items(), key=lambda x: -x[1])
            print("\nParameter sensitivity (impact on score):", flush=True)
            for k, v in sorted_sens:
                print(f"  {k:<25s}: {v:.4f}", flush=True)
            print(flush=True)

    # ------------------------------------------------------------------
    # Phase 2: Adaptive — decide next config based on what we've learned
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print(f"PHASE 2: Adaptive search around best (score={best_score:.4f})", flush=True)
    print("=" * 60, flush=True)

    while time.time() < deadline:
        remaining = (deadline - time.time()) / 60

        # Analyze recent results to decide strategy
        strategy, p, reasoning = decide_next(all_results, best_params, PARAM_RANGES, base_params)

        desc, config = config_from_params(p)
        print(f"\n--- Run {run_id} [{strategy}] ({remaining:.1f} min left) ---", flush=True)
        print(f"  Reasoning: {reasoning}", flush=True)

        try:
            u_ref, q_ref, metrics, elapsed = run_refinement(config)

            # Summarize cost curve for logging (don't dump full arrays)
            cost_summary = []
            for cc in metrics.get("cost_curves", []):
                cost_summary.append({
                    "n_iters": len(cc["costs"]),
                    "initial_cost": cc["costs"][0] if cc["costs"] else 0,
                    "final_cost": cc["final_cost"],
                    "late_improvement_frac": round(cc["late_improvement_frac"], 3),
                    "idx_best": cc["idx_best"],
                })
            metrics_for_log = {k: v for k, v in metrics.items() if k != "cost_curves"}
            metrics_for_log["cost_summary"] = cost_summary

            log_result(args.log_file, run_id, f"[{strategy}] {desc[:40]}",
                       config, metrics_for_log, elapsed, strategy)
            all_results.append((p, metrics["score"], metrics))

            if metrics["score"] < best_score:
                improvement = best_score - metrics["score"]
                print(f"  *** NEW BEST! score {best_score:.4f} -> {metrics['score']:.4f} "
                      f"(improved by {improvement:.4f}) ***", flush=True)
                best_score = metrics["score"]
                best_params = copy.deepcopy(p)
                best_config = copy.deepcopy(config)

                for save_name in ["traj_refined_best.npz",
                                  f"traj_refined_{src_ts}.npz"]:
                    np.savez_compressed(
                        os.path.join(DATA_DIR, save_name),
                        q_trj=q_ref, u_trj=u_ref,
                        h=1.0 / config["output_hz"],
                        q_u_indices_into_x=idx_q_u,
                        q_a_indices_into_x=idx_q_a,
                    )

        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            log_result(args.log_file, run_id, f"[{strategy}] FAIL: {e}",
                       config, {"score": float("inf")}, 0, strategy)
            all_results.append((p, float("inf"), {}))

        run_id += 1

        if run_id % 3 == 0:
            print_leaderboard(args.log_file)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print_leaderboard(args.log_file)
    print(f"\nTotal runs: {run_id}", flush=True)
    print(f"Best score: {best_score:.4f}", flush=True)
    print(f"Best params:", flush=True)
    print(json.dumps(best_params, indent=2), flush=True)
    print(f"Best config:", flush=True)
    print(json.dumps(best_config, indent=2), flush=True)
    print(f"\nBest trajectory: {os.path.join(DATA_DIR, 'traj_refined_best.npz')}", flush=True)
