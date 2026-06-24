# Planning Through Contact

Contact-rich manipulation planning using iterative smoothing of quasidynamic contact models. Plans trajectories for robots lifting, pushing, and rotating objects via contact, then exports them to IsaacLab for RL fine-tuning.

![](/media/planar_hand.gif) ![](/media/allegro_hand_ball.gif) ![](/media/allegro_hand_door.gif)

Based on [Global Planning for Contact-Rich Manipulation via Local Smoothing of Quasi-dynamic Contact Models](https://arxiv.org/abs/2206.10787).

---

## Pipeline overview

```
run_planner  →  refine_rrt  →  collision_free_rrt  →  prepare_for_isaaclab
    (RRT)         (iMPC, opt.)    (regrasp paths)        (RL export)
```

### 1. Run the RRT planner

Each task lives in `examples/<task_name>/`. The entry point is `run_planner.py`:

```bash
cd examples/box_lift_ur5e
python run_planner.py
```

This calls `scripts/run_planner.py`, which reads the task's setup module, runs `IrsRrtTrajectory.iterate()`, and saves a timestamped trajectory to `ptc_data/<task_name>/traj_<timestamp>.npz`.

### 2. Refine with iMPC (optional)

```bash
python scripts/refine_rrt.py ptc_data/<task_name>/traj_<timestamp>.npz
```

Runs `IrsMpcQuasistatic` on each contact segment to smooth the trajectory. Saves a `traj_refined_<timestamp>.npz` alongside the original.

### 3. Fill regrasp gaps

```bash
python scripts/collision_free_rrt.py ptc_data/<task_name>/traj_<timestamp>.npz
```

Plans collision-free paths between regrasp configurations and stitches them into the trajectory.

### 4. Export to IsaacLab

```bash
python scripts/prepare_for_isaaclab.py ptc_data/<task_name>/traj_refined_<timestamp>.npz
```

Upsamples to the RL timestep rate and converts poses to IsaacLab's `[x, y, z, qw, qx, qy, qz]` convention. Saves an `IK_<timestamp>.npz` to `RL_data/<task_name>/`.

### 5. Evaluate

```bash
python scripts/evaluate_trajectory.py ptc_data/<task_name>/traj_refined_<timestamp>.npz
```

Reports goal-position error, smoothness (max Δu, max Δ²u), and trajectory duration.

---

## Tasks

| Folder | Description |
|--------|-------------|
| `examples/box_lift_ur5e/` | Bimanual box lifting (two UR5e arms) — main task |
| `examples/box_push_ur5e/` | Single-arm box pushing (UR5e) |
| `examples/box_rotate_ur5e/` | Single-arm box rotation (UR5e) |
| `examples/box_lift_ur5/` | Earlier bimanual variant (UR5) |
| `examples/allegro_hand/` | Allegro dexterous hand examples |
| `examples/planar_hand/` | 2D planar hand (original TRO paper) |

Each task folder contains:
- `*_setup.py` — robot/object/RRT parameters, contact sampler
- `run_planner.py` — thin wrapper calling `scripts/run_planner.py`
- `collision_free_rrt.py` — regrasp path planner (if needed)

---

## Core library

| Module | Contents |
|--------|----------|
| `irs_rrt/` | RRT planner (`IrsRrtTrajectory`), reachable set, distance metrics |
| `irs_mpc2/` | iMPC trajectory optimizer (`IrsMpcQuasistatic`) |
| `scripts/` | Shared pipeline scripts (run, refine, export, evaluate, visualize) |
| `control/` | Hardware controllers for real-robot experiments |
| `dash_vis/` | Dash-based tree visualization |
