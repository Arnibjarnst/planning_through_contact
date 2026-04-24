#!/bin/bash
# Run the auto-improve loop with the correct conda environment and paths.
# Usage: ./run_auto_improve.sh [duration_minutes]
#
# Monitor:  tail -f auto_improve_log.json

set -e

DURATION=${1:-30}
REPO=/home/arni/planning_through_contact
TRAJ=${2:-$REPO/ptc_data/box_push_ur5e/traj_20260412_145454.npz}

cd "$REPO/examples/box_push_ur5e"

export LD_LIBRARY_PATH=/home/arni/drake-build/install/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO:/home/arni/quasistatic_simulator:${PYTHONPATH:-}

conda run -n planning --no-capture-output \
    python auto_improve.py "$TRAJ" \
        --duration_minutes "$DURATION" \
        --log_file auto_improve_log.json
