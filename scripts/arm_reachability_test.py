import numpy as np

import utils

from examples.box_lift.box_lift_setup import *


def random_quaternion():
    """
    Generate a random unit quaternion (w, x, y, z),
    uniformly distributed over SO(3).
    """
    u1, u2, u3 = np.random.rand(3)

    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),  # x
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),  # y
        np.sqrt(u1) * np.sin(2 * np.pi * u3),      # z
        np.sqrt(u1) * np.cos(2 * np.pi * u3),      # w
    ])

    quat = np.array([q[3], q[0], q[1], q[2]])

    quat /= np.linalg.norm(quat)

    # Return as (w, x, y, z)
    return np.array([q[3], q[0], q[1], q[2]])

N = 1000
ds = np.linspace(0, 1.0, 101)

success = np.zeros(len(ds))

import time

t1 = time.perf_counter()

for i, d in enumerate(ds):
    for _ in range(N):
        dir = np.random.rand(3)
        t = arm_l_pose[4:] + d * dir / np.linalg.norm(dir)
        
        quat = random_quaternion()

        pose = np.concatenate([quat, t])

        joints = utils.get_joints(pose, arm_l_pose)

        success[i] += joints is not None

t2 = time.perf_counter()

print("average IK time: ", 1000 * (t2-t1) / (N * len(ds)))

print("highest d with 95% success rate: ", ds[np.where(success > 0.95 * N)[0][-1]])
print("lowest d with 95% success rate: ", ds[np.where(success > 0.95 * N)[0][0]])





