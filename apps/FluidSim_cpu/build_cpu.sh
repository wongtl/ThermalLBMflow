#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

# Project paths.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$APP_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$SRC_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/venv-walberla-codegen"
BUILD_DIR="$PROJECT_ROOT/build-cpu"
INSTALL_CODEGEN_VENV_SCRIPT="$SRC_DIR/apps/shared/scripts/install_codegen_venv.sh"

# Build options.
TARGET="${TARGET:-FluidSim_cpu}"
BUILD_CPUS_PER_TASK="${BUILD_CPUS_PER_TASK:-8}"
RECONFIGURE="${RECONFIGURE:-0}"
BUILD_USE_SRUN="${BUILD_USE_SRUN:-1}"

# Module options (set empty to skip).
TOOLCHAIN_MODULE="${TOOLCHAIN_MODULE:-}"
PYTHON_MODULE="${PYTHON_MODULE:-}"
MPI_MODULE="${MPI_MODULE:-}"
CMAKE_MODULE="${CMAKE_MODULE:-}"

# Optional Slurm routing/resource defaults.
BUILD_CLUSTER="${BUILD_CLUSTER:-}"
BUILD_PARTITION="${BUILD_PARTITION:-}"
BUILD_TIME="${BUILD_TIME:-01:00:00}"
BUILD_MEM="${BUILD_MEM:-16G}"

# Required paths/files.
CORE_PATCH_SCRIPT="$SRC_DIR/apps/shared/scripts/apply_core_patches.sh"

[[ -d "$SRC_DIR" ]] || { echo "ERROR: Source directory not found: $SRC_DIR"; exit 1; }
[[ -f "$SRC_DIR/CMakeLists.txt" ]] || { echo "ERROR: Missing CMakeLists.txt in source dir: $SRC_DIR"; exit 1; }
[[ -f "$APP_DIR/build_cpu.sh" ]] || { echo "ERROR: Missing build script: $APP_DIR/build_cpu.sh"; exit 1; }
[[ -f "$INSTALL_CODEGEN_VENV_SCRIPT" ]] || { echo "ERROR: Missing codegen installer: $INSTALL_CODEGEN_VENV_SCRIPT"; exit 1; }
[[ -f "$CORE_PATCH_SCRIPT" ]] || { echo "ERROR: Missing core patch script: $CORE_PATCH_SCRIPT"; exit 1; }

# Always launch through srun when BUILD_USE_SRUN=1. Guard with FLUIDSIM_BUILD_INNER to avoid recursion.
if [[ "${BUILD_USE_SRUN}" == "1" && "${FLUIDSIM_BUILD_INNER:-0}" != "1" ]]; then
    # Prevent accidental reuse of stale allocation context from shell env.
    unset SLURM_JOB_ID SLURM_JOBID SLURM_STEP_ID SLURM_STEPID SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_JOB_NUM_NODES SLURM_NNODES

    SRUN_ARGS=(
        --time="$BUILD_TIME"
        --ntasks=1
        --cpus-per-task="$BUILD_CPUS_PER_TASK"
        --chdir="$PROJECT_ROOT"
    )
    [[ -n "$BUILD_CLUSTER" ]] && SRUN_ARGS+=(--clusters="$BUILD_CLUSTER")
    [[ -n "$BUILD_PARTITION" ]] && SRUN_ARGS+=(--partition="$BUILD_PARTITION")
    [[ -n "$BUILD_MEM" ]] && SRUN_ARGS+=(--mem="$BUILD_MEM")

    exec srun \
        "${SRUN_ARGS[@]}" \
        /usr/bin/env \
        FLUIDSIM_BUILD_INNER=1 \
        BUILD_USE_SRUN="$BUILD_USE_SRUN" \
        RECONFIGURE="$RECONFIGURE" \
        BUILD_CPUS_PER_TASK="$BUILD_CPUS_PER_TASK" \
        TARGET="$TARGET" \
        TOOLCHAIN_MODULE="$TOOLCHAIN_MODULE" \
        PYTHON_MODULE="$PYTHON_MODULE" \
        MPI_MODULE="$MPI_MODULE" \
        CMAKE_MODULE="$CMAKE_MODULE" \
        "$APP_DIR/build_cpu.sh"
fi

# Module environment bootstrap.
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi

# Module load (if available).
if command -v module >/dev/null 2>&1; then
    module purge
    [[ -n "$TOOLCHAIN_MODULE" ]] && module load "$TOOLCHAIN_MODULE"
    [[ -n "$PYTHON_MODULE" ]] && module load "$PYTHON_MODULE"
    [[ -n "$MPI_MODULE" ]] && module load "$MPI_MODULE"
    [[ -n "$CMAKE_MODULE" ]] && module load "$CMAKE_MODULE"
fi

# Tool checks and environment diagnostics.
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 was not found in PATH." >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake was not found in PATH." >&2
    exit 1
fi

PYTHON_BIN="$(command -v python3)"
echo "Using python: $PYTHON_BIN"
echo "Using cmake: $(command -v cmake)"
cmake --version | head -n 1

# Venv policy (CPU): require the canonical pinned codegen environment.
# Bootstrap once with:
#   apps/shared/scripts/install_codegen_venv.sh
if [[ ! -x "$VENV/bin/python" || ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: Required codegen venv not found at $VENV" >&2
    echo "Run once: $INSTALL_CODEGEN_VENV_SCRIPT" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

if ! python - <<'PY'
import numpy, sympy, jinja2, lbmpy, pystencils, pystencilssfg, sweepgen
PY
then
    echo "ERROR: Missing codegen dependencies in $VENV." >&2
    echo "Run once: $INSTALL_CODEGEN_VENV_SCRIPT" >&2
    exit 1
fi

# MPI compiler checks.
if ! command -v mpicc >/dev/null 2>&1 || ! command -v mpicxx >/dev/null 2>&1; then
    echo "ERROR: mpicc/mpicxx not found after module load." >&2
    echo "Set TOOLCHAIN_MODULE/MPI_MODULE explicitly." >&2
    exit 1
fi

export CC="mpicc"
export CXX="mpicxx"

# Apply shared core patches.
echo "Applying core patches..."
bash "$CORE_PATCH_SCRIPT" --walberla-root "$SRC_DIR"

# Configure and build.
if [[ "$RECONFIGURE" == "1" ]]; then
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

if [[ "$RECONFIGURE" == "1" || ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DPython_EXECUTABLE="$VENV/bin/python"
        -DWALBERLA_CODEGEN_PYTHON="$VENV/bin/python"
        -DWALBERLA_BUILD_WITH_MPI=ON
        -DWALBERLA_BUILD_WITH_OPENMP=ON
        -DWALBERLA_BUILD_WITH_OPENMESH=ON
        -DWALBERLA_BUILD_WITH_CODEGEN=ON
        -DWALBERLA_ENABLE_SWEEPGEN=ON
        -DWALBERLA_SWEEPGEN_MANAGED_VENV=OFF
        -DWALBERLA_BUILD_TESTS=OFF
    )
    cmake -S "$SRC_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
fi

cmake --build "$BUILD_DIR" --target "$TARGET" --parallel "$BUILD_CPUS_PER_TASK"

echo "Build OK: $BUILD_DIR/apps/FluidSim_cpu/FluidSim_cpu"
