#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

# Path and runtime configuration.

# App paths.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALBERLA_ROOT="$(cd "$APP_DIR/../.." && pwd)"
BUILD_DIR="$WALBERLA_ROOT/build-cpu"
REAL_EXE="$BUILD_DIR/apps/FluidSim_cpu/FluidSim_cpu"
SHARED_DIR="$APP_DIR/../shared"
PARAMS="$SHARED_DIR/params/FluidSim_LZ.prm"

# Runtime knobs (launcher-owned, environment-overridable).
NP="${NP:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
OMP_PLACES="${OMP_PLACES:-cores}"
LOG_DIR="$APP_DIR/output/logs"
LOG_FILE="$LOG_DIR/run_sim.log"

# Launcher-owned FluidSim defaults for local execution.
FLAGS=(
    --parallelMode inner
    --minimalLogs 1000
    --thermalLogs 100
    --initPerturb 0
    --vtkinit 1
    --timesteps 1
    --checkpointEvery 0
    --vtkevery 0
    --vtkmeshonly true
)

# This launcher owns all app runtime flags.
if (( $# > 0 )); then
    echo "ERROR: run_sim_local.sh does not accept extra app arguments." >&2
    echo "Use environment knobs (for example NP, OMP_NUM_THREADS)." >&2
    exit 1
fi

# Input and executable validation.
if [[ ! -f "$PARAMS" ]]; then
    echo "ERROR: Parameter file not found: $PARAMS" >&2
    exit 1
fi

if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun not found in PATH." >&2
    exit 1
fi
if ! command -v stdbuf >/dev/null 2>&1; then
    echo "ERROR: stdbuf not found in PATH." >&2
    exit 1
fi

if [[ ! -x "$REAL_EXE" ]]; then
    echo "ERROR: FluidSim executable not found: $REAL_EXE" >&2
    echo "Checked: $REAL_EXE" >&2
    echo "Build it first with: ./build_local.sh" >&2
    exit 1
fi

# Runtime metadata and launch.
mkdir -p "$LOG_DIR"
cd "$APP_DIR"

export OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES

# Keep a common mpirun argument shape across local/CPU/GPU launchers.
LAUNCH_TARGET="$REAL_EXE"
MPI_EXTRA_ARGS=(--allow-run-as-root)
MPIRUN_ARGS=(
    "${MPI_EXTRA_ARGS[@]}"
    -np "$NP"
)

mpirun "${MPIRUN_ARGS[@]}" stdbuf -oL -eL "$LAUNCH_TARGET" "$PARAMS" "${FLAGS[@]}" > "$LOG_FILE" 2>&1
