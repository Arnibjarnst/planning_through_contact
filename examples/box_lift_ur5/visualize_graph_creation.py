import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.quasistatic_visualizer import (
    InternalVisualizationType,
)

parser = argparse.ArgumentParser()
parser.add_argument("tree_file_path", type=str)
parser.add_argument("--color_method", type=str, default="contact")
parser.add_argument("--t_aperture", action="store_true")
args = parser.parse_args()

with open(args.tree_file_path, "rb") as f:
    tree = pickle.load(f)

prob_rrt = IrsRrt.make_from_pickled_tree(
    tree, internal_vis=InternalVisualizationType.Cpp
)

idx_xyz = prob_rrt.q_sim.get_q_u_indices_into_q()[4:]

def get_aperture(node):
    if args.t_aperture:
        return np.exp(-node.distance_from_traj)
    else:
        return 1.0

def color_from_contact_mode(node):
    rng = np.random.default_rng(seed=node.contact_mode_id)
    rgb = rng.random(3)
    a = get_aperture(node)
    rgba = np.concatenate((rgb, [a]))
    return rgba

def color_from_static(node):
    a = get_aperture(node)
    if node.is_static:
        return np.array([0,1,0,a])
    else:
        return np.array([1,0,0,a])
    
def color_from_active(node):
    a = get_aperture(node)
    if node.is_active:
        return np.array([0,1,0,a])
    else:
        return np.array([1,0,0,a])

def color_from_node(node):
    if args.color_method == "contact":
        return color_from_contact_mode(node)
    elif args.color_method == "static":
        return color_from_static(node)
    elif args.color_method == "active":
        return color_from_active(node)
    else:
        print("INVALID COLOR METHOD")
        return None


init_graph_size = 100

nodes = [prob_rrt.get_node_from_id(i) for i in range(prob_rrt.size)]

def animate_nodes(nodes):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    plt.subplots_adjust(bottom=0.25)

    xyzs = np.array([n.q[idx_xyz] for n in nodes])
    colors = [color_from_node(node) for node in nodes]

    lines = []
    scatter = ax.scatter([], [], [], s=30)

    total_frames = len(nodes) - init_graph_size

    state = {
        "paused": False,
        "frame": 0,
        "updating_slider": False,
    }

    def clear_lines():
        for l in lines:
            l.remove()
        lines.clear()

    def draw_until(frame):
        clear_lines()

        node_id = init_graph_size + frame

        scatter._offsets3d = (
            xyzs[:node_id+1, 0],
            xyzs[:node_id+1, 1],
            xyzs[:node_id+1, 2],
        )
        scatter.set_color(colors[:node_id+1])

        for f in range(frame + 1):
            i = init_graph_size + f
            path = prob_rrt.trace_nodes_to_root_from(i)
            if len(path) > 1:
                parent_id = path[-2]
                parent_xyz = xyzs[parent_id]
                line, = ax.plot(
                    [parent_xyz[0], xyzs[i, 0]],
                    [parent_xyz[1], xyzs[i, 1]],
                    [parent_xyz[2], xyzs[i, 2]],
                    linewidth=1
                )
                line.set_color(colors[parent_id])
                lines.append(line)

    def update(fake_frame):
        state["frame"] = min(state["frame"] + 1, total_frames - 1)
        draw_until(state["frame"])

        state["updating_slider"] = True
        slider.set_val(state["frame"])
        state["updating_slider"] = False

        return [scatter] + lines



    # Axis limits
    ax.set_xlim(xyzs[:,0].min(), xyzs[:,0].max())
    ax.set_ylim(xyzs[:,1].min(), xyzs[:,1].max())
    ax.set_zlim(xyzs[:,2].min(), xyzs[:,2].max())

    global ani
    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=200,
        repeat=False,
        blit=False
    )

    # ---------------- Controls ---------------- #

    # Pause button
    ax_pause = plt.axes([0.1, 0.1, 0.1, 0.05])
    btn_pause = Button(ax_pause, "Pause")

    def toggle_pause(event):
        if state["paused"]:
            ani.event_source.start()
            btn_pause.label.set_text("Pause")
            state["paused"] = False
        else:
            ani.event_source.stop()
            btn_pause.label.set_text("Resume")
            state["paused"] = True


    btn_pause.on_clicked(toggle_pause)

    # Restart button
    ax_restart = plt.axes([0.22, 0.1, 0.1, 0.05])
    btn_restart = Button(ax_restart, "Restart")

    def restart(event):
        ani.event_source.stop()
        state["paused"] = True
        btn_pause.label.set_text("Resume")

        state["frame"] = 0
        slider.set_val(0)
        draw_until(0)

    btn_restart.on_clicked(restart)

    # Slider
    ax_slider = plt.axes([0.4, 0.1, 0.45, 0.05])
    slider = Slider(
        ax_slider,
        "Frame",
        0,
        total_frames - 1,
        valinit=0,
        valstep=1
    )

    def on_slider(val):
        if state["updating_slider"]:
            return

        ani.event_source.stop()
        state["paused"] = True
        btn_pause.label.set_text("Resume")

        frame = int(val)
        state["frame"] = frame
        draw_until(frame)


    slider.on_changed(on_slider)

    plt.show()


animate_nodes(nodes)
