# visualize_trajectory.py
import os
import pickle
import numpy as np
import time

from omni.isaac.kit import SimulationApp

# start Isaac Sim
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf
import omni.usd
import argparse

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCone
import omni.isaac.core.utils.prims as prims_utils

import omni.ui as ui

FORCE_SCALE = 5e-2

def visualize(obj_traj, eef_traj, forces, animation_duration = None):
    """
    Visualize an object + 2 spheres + a trajectory in Isaac Sim.

    object_sdf, sphere1_sdf, sphere2_sdf: paths to SDF files
    trajectory: list of [x,y,z] positions
    headless: if True, runs without GUI
    """

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    box = DynamicCuboid(
        name="box1",
        prim_path="/World/box1",
        scale=np.array([0.4, 0.6, 0.06]),  # 1m x 1m x 1m
        color=np.array([1,0,0])  # red
    )
    world.scene.add(box)

    eefs = []
    for i in range(2):
        eef = DynamicSphere(
            name=f"sphere{i}",
            prim_path=f"/World/sphere{i}",
            position=eef_traj[0,i],
            radius=0.03,
            color=np.array([0,0,1]),
        )
        eefs.append(eef)
        world.scene.add(eef)

    def clear_arrows():
        prims_utils.delete_prim("/World/force")

    def draw_arrows(start, vec, name_prefix="arrow"):
        """
        Draws an arrow from 'start' along vector 'vec' using a cylinder + cone.
        
        Parameters:
            start: array-like (3,) starting point
            vec: array-like (3,) vector
            color: Gf.Vec3f
            name_prefix: string prefix for prim names
        """
        length = np.linalg.norm(vec)
        if length == 0:
            return

        direction = np.array(vec) / length
        start = np.array(start)

        # ---------------------
        # Shaft (cylinder)
        # ---------------------
        shaft_length = length * 0.8
        shaft_radius = 0.01
        shaft_center = start + direction * (shaft_length / 2)

        # Create DynamicCuboid as shaft aligned along Z
        shaft = DynamicCuboid(
            name=f"{name_prefix}_shaft",
            prim_path=f"/World/force/{name_prefix}_shaft",
            position=shaft_center,
            scale=np.array([shaft_radius, shaft_radius, shaft_length]),
        )
        
        # ---------------------
        # Head (cone)
        # ---------------------
        head_length = length * 0.2
        head_radius = 0.02
        head_center = start + direction * (shaft_length + head_length / 2)
        
        head = DynamicCone(
            name=f"{name_prefix}_head",
            prim_path=f"/World/force/{name_prefix}_head",
            position=head_center,
            radius=head_radius,
            height=head_length,
        )

        # ---------------------
        # Rotation (align Z to vector)
        # ---------------------
        # Compute rotation quaternion to align local Z with vec
        z_axis = np.array([0,0,1])
        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        if s < 1e-6:
            quat = np.array([1,0,0,0])  # no rotation needed
        else:
            v_norm = v / s
            angle = np.arctan2(s, c)
            quat = np.array([np.cos(angle/2), *(v_norm*np.sin(angle/2))])
        
        shaft.set_world_pose(orientation=quat)
        head.set_world_pose(orientation=quat)

        # world.scene.add(head)
        # world.scene.add(shaft)

    def update_frame(t):
        obj_pose = obj_traj[t]
        box.set_world_pose(position=obj_pose[4:], orientation=obj_pose[:4])
        
        # Move spheres
        clear_arrows()
        for i, sphere in enumerate(eefs):
            sphere.set_world_pose(eef_traj[t, i])
            draw_arrows(eef_traj[t, i], FORCE_SCALE * forces[t][i], name_prefix=f"arrow_{i}_{t}")

    num_frames = len(obj_traj)
    playing = False
    last_time = time.time()
    frame_idx = 0

    window = ui.Window("Trajectory Control", width=400, height=150)
    with window.frame:
        with ui.VStack():
            # Play / Pause / Reset buttons
            with ui.HStack():
                play_button = ui.Button("Play")
                pause_button = ui.Button("Pause")
                reset_button = ui.Button("Reset")

            # Slider
            slider = ui.IntSlider(min=0, max=num_frames-1, step=1)
            ui.Label("Frame")

            # Animation duration input
            with ui.HStack():
                ui.Label("Total Animation Time (s):")
                duration_input = ui.FloatField()
                duration_input.model.set_value(5.0)  # default 5 seconds

    # Button callbacks
    def on_play():
        nonlocal playing, last_time
        playing = True
        last_time = time.time()

    def on_pause():
        nonlocal playing
        playing = False

    def on_reset():
        nonlocal frame_idx, playing, last_time
        frame_idx = 0
        slider.model.set_value(frame_idx)
        update_frame(frame_idx)
        playing = False
        last_time = time.time()

    # Slider callback
    def slider_changed(temp):
        nonlocal frame_idx
        frame_idx = slider.model.as_int
        update_frame(frame_idx)

    play_button.set_clicked_fn(on_play)
    pause_button.set_clicked_fn(on_pause)
    reset_button.set_clicked_fn(on_reset)
    slider.model.add_value_changed_fn(slider_changed)

    while simulation_app.is_running():
        current_time = time.time()
        elapsed = current_time - last_time
        animation_duration = duration_input.model.get_value_as_float()
        dt = animation_duration / num_frames

        if playing and dt is not None and elapsed >= dt:
            # Advance frame based on time
            last_time = current_time
            frame_idx += 1
            if frame_idx >= num_frames:
                frame_idx = num_frames - 1
                playing = False  # Stop at end

            print(num_frames, frame_idx)
            slider.model.set_value(frame_idx)
            update_frame(frame_idx)

        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_file")
    parser.add_argument(
        "--animate",
        type=float,
        default=None,
        help="Animate trajectory over T seconds (optional)"
    )

    args = parser.parse_args()

    with open(args.force_file, "rb") as f:
        traj_dict = pickle.load(f)

    obj_traj = traj_dict["obj_traj"]
    eef_traj = traj_dict["eef_traj"]
    forces = traj_dict["forces"]

    eef_traj = eef_traj.reshape((len(eef_traj), 2, 3))
    forces = forces.reshape((len(forces), 2, 3))

    print(obj_traj.shape, eef_traj.shape, forces.shape)

    visualize(obj_traj, eef_traj, forces, args.animate)
