import time
import copy
from datetime import datetime
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("update", type=str, default=None)
args = parser.parse_args()

from irs_rrt.irs_rrt_trajectory import IrsRrtTrajectory

from box_lift_setup import *

import numpy as np


with open(args.update) as fp:
    records = json.load(fp)
save_path = args.update

contact_sampler.flip_axis_prob = 0.0


for key in records:
    ts = np.array(records[key]["dt"])
    success = np.array(records[key]["success"])
    success_ts = ts[success]
    mean_success_t = success_ts.mean() if len(success_ts) > 0 else 0
    records[key]["mean_success_t"] = mean_success_t

with open(save_path, 'w') as fp:
    json.dump(records, fp, indent=4)

