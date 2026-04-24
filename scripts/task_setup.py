"""
Bridge between a task_name string and its setup module / metadata.

- `metadata_from_setup(setup)` builds a flat dict that gets saved into the
  trajectory npz (task_name, arm poses, object dims, etc.) so downstream
  tools can read the metadata without re-importing the setup module.
- `deduce_setup(task_name)` does the reverse: given a task_name (typically
  read back from the npz), import and return the setup module from
  `examples/<task_name>/*_setup.py`.
"""
import glob
import importlib
import os
import sys

import numpy as np


def metadata_from_setup(setup, task_name: str = None):
    """Build a flat metadata dict from a task setup module.

    The returned dict can be unpacked directly into np.savez_compressed.
    `task_name` defaults to the basename of setup.data_folder.
    """
    if task_name is None:
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


def deduce_setup(task_name: str):
    """Load and return the setup module for the given task name.

    `<repo>/examples/<task_name>/` is expected to contain exactly one
    `*_setup.py` file. That file is imported and returned.
    """
    task_name = str(task_name)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    example_dir = os.path.join(repo_root, "examples", task_name)
    if not os.path.isdir(example_dir):
        raise ValueError(
            f"Could not deduce setup module: no directory "
            f"'examples/{task_name}' found."
        )

    setup_files = sorted(glob.glob(os.path.join(example_dir, "*_setup.py")))
    if not setup_files:
        raise ValueError(f"No *_setup.py file found in {example_dir}.")
    if len(setup_files) > 1:
        raise ValueError(
            f"Multiple *_setup.py files found in {example_dir}: "
            f"{[os.path.basename(p) for p in setup_files]}. "
            f"Cannot auto-deduce."
        )

    setup_path = setup_files[0]
    module_name = os.path.splitext(os.path.basename(setup_path))[0]

    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)

    return importlib.import_module(module_name)
