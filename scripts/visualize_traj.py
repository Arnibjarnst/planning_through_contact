"""
Publish a saved trajectory via the task's q_vis.

The setup module is loaded automatically from the task_name stored in the
npz (falling back to the trajectory's parent directory name if absent).

Usage:
    python -m scripts.visualize_traj ptc_data/box_push_ur5e/traj_refined_*.npz
"""
import argparse
import importlib
import os

import numpy as np

from scripts.task_setup import deduce_setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_path", type=str)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument(
        "--setup_module", type=str, default=None,
        help="Override the auto-deduced setup module name (e.g. box_push_setup).",
    )
    args = parser.parse_args()

    data = np.load(args.traj_path)

    if args.setup_module is not None:
        setup = importlib.import_module(args.setup_module)
    else:
        if "task_name" in data.files:
            task_name = str(data["task_name"])
        else:
            task_name = os.path.basename(
                os.path.dirname(os.path.abspath(args.traj_path))
            )
        setup = deduce_setup(task_name)

    q = data["q_trj"]
    h = args.h
    if h is None:
        h = float(data["h"]) if "h" in data.files else 0.1

    setup.q_vis.publish_trajectory(q, h)
    input("EXIT")


if __name__ == "__main__":
    main()
