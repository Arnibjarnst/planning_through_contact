"""
Build task metadata from a setup module so it can be saved into the
trajectory npz alongside q_trj/u_trj. Once a trajectory has been written
by run_planner, every downstream tool (refine_rrt, collision_free_rrt,
prepare_for_isaaclab) reads the metadata directly from the npz and never
needs the setup module again.

Usage in run_planner:
    md = metadata_from_setup(setup)
    np.savez_compressed(path, q_trj=..., u_trj=..., **md)

Downstream scripts just modify the loaded npz dict directly, e.g.:
    data = np.load(in_path)
    out = {k: data[k] for k in data.files}
    out["q_trj"] = new_q
    out["u_trj"] = new_u
    np.savez_compressed(out_path, **out)
"""
import os

import numpy as np


def metadata_from_setup(setup):
    """Build a flat metadata dict from a task setup module.

    The returned dict can be unpacked directly into np.savez_compressed.
    """

    task_name = os.path.basename(setup.data_folder.rstrip("/"))

    arm_items = list(setup.rrt_params.arm_poses.items())
    arm_names = np.array([name for name, _ in arm_items])
    arm_poses = np.array([pose for _, pose in arm_items])

    md = {
        "task_name":   task_name,
        "arm_names":   arm_names,
        "arm_poses":   arm_poses,
        "eef_offset":  float(getattr(setup, "eef_offset", 0.0)),
        "object_dims": np.asarray(setup.object_dims),
        "object_mass": float(getattr(setup, "box_mass", 0.0)),
    }

    return md
