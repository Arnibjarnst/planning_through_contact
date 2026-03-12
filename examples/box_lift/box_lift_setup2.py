import os
import numpy as np

from qsim.model_paths import models_dir

from pydrake.math import RollPitchYaw

q_model_path = os.path.join(models_dir, "q_sys", "magic_finger_box.yml")

# Names
eef_name = "finger"
object_name = "box"

# initial conditions.
object_dims = [0.4, 0.6, 0.06]
q_u0 = np.array([1, 0, 0, 0, 0, 0, 0.03])

q_a0_local = np.array([0, 0.32, 0.0])
q_a0 = q_a0_local + q_u0[4:]


def get_obj_pose_from_t(t: float):
    t = max(min(t, 1), 0)
    rad = t * np.pi / 2
    quat = RollPitchYaw(rad, 0, 0).ToQuaternion()
    rotation_origin = np.array([0, -0.3, 0.0])
    rotation_origin_to_obj_center = q_u0[4:] - rotation_origin
    pos_diff = (quat.rotation() @ rotation_origin_to_obj_center) - rotation_origin_to_obj_center

    shift = np.array([0, 0.06, 0.0]) * np.sin(rad)

    curr_pos = q_u0[4:] + pos_diff + shift
    pose = np.concatenate([quat.wxyz(), curr_pos])
    return pose

# Goal conditions
goal_u = get_obj_pose_from_t(1.0)
goal_a = np.array([0, -0.33, 0.62])

# data collection.
data_folder = "ptc_data/box_lift"
