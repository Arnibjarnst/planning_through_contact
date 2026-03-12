import time
import copy
from datetime import datetime
import json


from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory

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


def safe_deepcopy(obj):
    cls = obj.__class__
    new_obj = cls.__new__(cls)
    for k, v in vars(obj).items():
        try:
            setattr(new_obj, k, copy.deepcopy(v))
        except Exception:
            setattr(new_obj, k, v)
    return new_obj

seeds = list(range(10))

batch_sizes = [1,2,4,8,16]

records = {}

data_folder = "ablation_data/box_lift_ur5"
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(
    data_folder, f"ablation_batch_size_{ts}.json"
)

for batch_size in batch_sizes:
    rrt_params_bs = safe_deepcopy(rrt_params)
    rrt_params_bs.batch_size = batch_size

    record = {
        "dt": [],
        "success": [],
        "mean_dt": 0,
        "mean_success": 0,
    }

    for seed in seeds:
        dt, success = run_trial(rrt_params_bs, seed)
        record["dt"].append(dt)
        record["success"].append(success)

        record["mean_dt"] = sum(record["dt"]) / len(record["dt"])
        record["mean_success"] = sum(record["success"]) / len(record["success"])

        records[batch_size] = record

        with open(save_path, 'w') as fp:
            json.dump(records, fp, indent=4)

