import argparse
import numpy as np

from box_push_setup import *

parser = argparse.ArgumentParser()
parser.add_argument("traj_path", type=str)
parser.add_argument
args = parser.parse_args()

data = np.load(args.traj_path)

q = data["q_trj"]
    
q_vis.publish_trajectory(q, 0.1)

input("EXIT")

