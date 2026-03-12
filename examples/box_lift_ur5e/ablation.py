import time
import copy
from datetime import datetime
import json

from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory
from irs_rrt.rrt_params import DuStarMode, DistanceMetric

from box_lift_setup import *

def run_trial(rrt_params, seed):
    np.random.seed(seed)

    prob_rrt = IrsRrtTrajectory(
        rrt_params,
        contact_sampler,
        q_sim,
        q_sim_py,
        pose_sampling_function,
        # q_sim_smooth=q_sim_smooth # Different scene for gradients
        q_sim_smooth=None # Same scene for gradients
    )

    t0 = time.perf_counter()
    prob_rrt.iterate()
    t1 = time.perf_counter()
    dt = t1 - t0

    success = prob_rrt.goal_node_idx is not None

    return dt, success

def safe_deepcopy_dict(d):
    new_d = {}
    for k,v in d.items():
        try:
            new_d[k] = copy.deepcopy(v)
        except Exception:
            new_d[k] = safe_deepcopy(v)
    return new_d


def safe_deepcopy_obj(obj):
    cls = obj.__class__
    new_obj = cls.__new__(cls)
    for k, v in vars(obj).items():
        try:
            setattr(new_obj, k, copy.deepcopy(v))
        except Exception:
            setattr(new_obj, k, safe_deepcopy(v))
    return new_obj


def safe_deepcopy(x):
    if isinstance(x, dict):
        return safe_deepcopy_dict(x)
    elif isinstance(x, object):
        return safe_deepcopy_obj(x)
    
    try:
        return copy.deepcopy(x)
    except:
        return x

seeds = list(range(10))

rrt_params_no_batch = safe_deepcopy(rrt_params)
rrt_params_no_batch.batch_size = 1

rrt_params_lstsq = safe_deepcopy(rrt_params)
rrt_params_lstsq.du_star_mode = DuStarMode.LSTSQ

rrt_params_no_corner_distance = safe_deepcopy(rrt_params)
rrt_params_no_corner_distance.static_distance_metric = DistanceMetric.Mahalabonis

rrt_params_no_connect_from_behind = safe_deepcopy(rrt_params)
rrt_params_no_connect_from_behind.connect_from_behind = False

rrt_params_no_connect_to_front = safe_deepcopy(rrt_params)
rrt_params_no_connect_to_front.connect_to_front = False

rrt_params_no_sample_noise = safe_deepcopy(rrt_params)
rrt_params_no_sample_noise.joint_limits[idx_u] = np.zeros((7,2))

rrt_params_no_subgoals = safe_deepcopy(rrt_params)
rrt_params_no_subgoals.subgoal_ts = [0.0, 1.0]


rrt_params_ablation = {
    "normal": rrt_params,
    "batch=1": rrt_params_no_batch,
    "lstsq": rrt_params_lstsq,
    "reachability regrasp": rrt_params_no_corner_distance,
    "connect from anywhere": rrt_params_no_connect_from_behind,
    "connect to anywhere": rrt_params_no_connect_to_front,
    "no trajectory noise": rrt_params_no_sample_noise,
    "no subgoals": rrt_params_no_subgoals,
}

records = {}

data_folder = "ablation_data/box_lift_ur5e"
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(
    data_folder, f"ablation_{ts}.json"
)

for trial_name, rrt_param_config in rrt_params_ablation.items():
    record = {
        "dt": [],
        "success": [],
        "mean_dt": 0,
        "mean_success": 0,
    }

    for seed in seeds:
        dt, success = run_trial(rrt_param_config, seed)
        record["t"].append(dt)
        record["success"].append(success)

        t_array = np.array(record["t"])
        success_array = np.array(record["success"])

        record["mean_t"] = t_array.mean()
        record["mean_success"] = success_array.mean()
        records["mean_success_t"] = t_array[success_array].mean()

        records[trial_name] = record

        with open(save_path, 'w') as fp:
            json.dump(records, fp, indent=4)

    
