import numpy as np
import time
import argparse

np.set_printoptions(precision=5, suppress=True)

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from pxr import Gf, UsdPhysics, Sdf

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere, FixedCuboid, VisualCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects.ground_plane import GroundPlane

import carb.input
import omni.kit.app


parser = argparse.ArgumentParser()
parser.add_argument("IK_file_path", type=str)
parser.add_argument("--simulate", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

data = np.load(args.IK_file_path)

obj_poses       = data["obj_poses"]
joints_l        = data["joints_l"]
joints_r        = data["joints_r"]
joints_target_l = data["joints_target_l"]
joints_target_r = data["joints_target_r"]
arm_l_pose    = data["arm_l_pose"]
arm_r_pose    = data["arm_r_pose"]
dt              = data["dt"]

N = len(joints_l)

physics_dt = 1.0 / 240.0

world = World(physics_dt=physics_dt)
# world.scene.add_default_ground_plane()

ground = GroundPlane(prim_path="/World/groundPlane", size=10, z_position=-1, color=np.array([0.5, 0.5, 0.5]))

normal_friction = PhysicsMaterial(
    prim_path="/World/PhysicsMaterials/normal_friction",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.0,
)

prim_path_box = "/World/envs/env_0/box"
box = DynamicCuboid(
    prim_path=prim_path_box,
    name="box",
    orientation=obj_poses[0,3:],
    position=obj_poses[0, :3],
    scale=np.array([0.4, 0.6, 0.06]),
    mass=1.0,
    physics_material=normal_friction,
) if args.simulate else VisualCuboid(
    prim_path=prim_path_box,
    name="box",
    orientation=obj_poses[0,3:],
    position=obj_poses[0, :3],
    scale=np.array([0.4, 0.6, 0.06]),
)

FixedCuboid(
    prim_path="/World/envs/env_0/ground_box",
    name="ground_box",
    position=np.array([0.0, 0.0, -0.5]),
    scale=np.array([0.4, 0.6, 1.0]),
    color=np.array([0.3, 0.3, 0.3])
) if args.simulate else VisualCuboid(
    prim_path="/World/envs/env_0/ground_box",
    name="ground_box",
    position=np.array([0.0, 0.0, -0.5]),
    scale=np.array([0.4, 0.6, 1.0]),
    color=np.array([0.3, 0.3, 0.3])
)


usd_path = "../BoxLift/robots/ur5_sphere_1.0_2.usd"

prim_path_l = "/World/envs/env_0/ur5_left"
prim_path_r = "/World/envs/env_0/ur5_right"
add_reference_to_stage(usd_path, prim_path_l)
add_reference_to_stage(usd_path, prim_path_r)

# Create SingleArticulation wrapper (automatically creates articulation controller)
arm_l = SingleArticulation(prim_path=prim_path_l, name="ur5_left")
arm_r = SingleArticulation(prim_path=prim_path_r, name="ur5_right")

def print_state():
    current_positions_l = arm_l.get_joint_positions()
    current_positions_r = arm_r.get_joint_positions()
    position, orientation = box.get_world_pose()


    print(f"Original joint positions left: {joints_l}")
    print(f"Original joint positions right: {joints_r}")

    print(f"Target joint positions left: {joints_target_l}")
    print(f"Target joint positions right: {joints_target_r}")

    print(f"Current joint positions left: {current_positions_l}")
    print(f"Current joint positions right: {current_positions_r}")

    print(f"Initial box position: {obj_poses[0, :3]}")
    print(f"Current box position:    {position}")

    print(f"Initial box orientation: {[1.0, 0.0, 0.0, 0.0]}")
    print(f"Current box orientation: {orientation}")




def initialize():
    # initialize the world
    world.reset()

    # Initialize the robot (initializes articulation controller internally)
    arm_l.initialize()
    arm_r.initialize()

    arm_l.set_joint_positions(joints_l[0])
    arm_l.set_world_pose(position=arm_l_pose[:3], orientation=arm_l_pose[3:])
    arm_r.set_joint_positions(joints_r[0])
    arm_r.set_world_pose(position=arm_r_pose[:3], orientation=arm_r_pose[3:])

    box.set_world_pose(orientation=obj_poses[0,3:], position=obj_poses[0, :3])

initialize()


input_iface = carb.input.acquire_input_interface()

def on_keyboard_event(event):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.R:
            print("Restarting simulation")
            initialize()

input_iface.subscribe_to_keyboard_events(None, on_keyboard_event)

while simulation_app.is_running():
    world.step(render=True)

    i = min(int(world.current_time // dt), N-1)

    if args.simulate:
        # Create and apply articulation action
        action_l = ArticulationAction(joint_positions=joints_target_l[i])
        action_r = ArticulationAction(joint_positions=joints_target_r[i])
        arm_l.apply_action(action_l)
        arm_r.apply_action(action_r)
    else:
        arm_l.set_joint_positions(joints_l[i])
        arm_r.set_joint_positions(joints_r[i])
        box.set_world_pose(orientation=obj_poses[i,3:], position=obj_poses[i, :3])

    time.sleep(physics_dt)
