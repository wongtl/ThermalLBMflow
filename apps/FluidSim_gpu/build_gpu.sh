#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

# Project paths.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$APP_DIR/../../.." && pwd)"
SRC_DIR="$PROJECT_ROOT/walberla"
VENV="$PROJECT_ROOT/venv-walberla-codegen"
BUILD_DIR="$PROJECT_ROOT/build-gpu"

# Build options.
TARGET="${TARGET:-FluidSim_gpu}"
BUILD_CPUS_PER_TASK="${BUILD_CPUS_PER_TASK:-16}"
RECONFIGURE="${RECONFIGURE:-0}"
BUILD_USE_SRUN="${BUILD_USE_SRUN:-1}"

# Module options (set empty to skip).
TOOLCHAIN_MODULE="${TOOLCHAIN_MODULE:-foss/2024a}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
MPI_MODULE="${MPI_MODULE:-}"
GPU_MODULE="${GPU_MODULE:-CUDA/12.9.0}"
CMAKE_MODULE="${CMAKE_MODULE:-CMake/3.29.3-GCCcore-13.3.0}"

# CUDA build options.
# Set CUDA_ARCHITECTURES to match your GPU(s): V100=70, L40S=89, H100=90.
# Example single-arch build: CUDA_ARCHITECTURES=89 ./build_gpu.sh
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-70;89;90}"

# Slurm allocation defaults.
BUILD_CLUSTER="${BUILD_CLUSTER:-htc}"
BUILD_PARTITION="${BUILD_PARTITION:-interactive}"
BUILD_TIME="${BUILD_TIME:-01:00:00}"
BUILD_MEM="${BUILD_MEM:-16G}"

# Required paths/files.
CORE_PATCH_SCRIPT="$SRC_DIR/apps/shared/scripts/apply_core_patches.sh"

[[ -d "$SRC_DIR" ]] || { echo "ERROR: Source directory not found: $SRC_DIR"; exit 1; }
[[ -f "$SRC_DIR/CMakeLists.txt" ]] || { echo "ERROR: Missing CMakeLists.txt in source dir: $SRC_DIR"; exit 1; }
[[ -f "$APP_DIR/build_gpu.sh" ]] || { echo "ERROR: Missing build script: $APP_DIR/build_gpu.sh"; exit 1; }
[[ -f "$CORE_PATCH_SCRIPT" ]] || { echo "ERROR: Missing core patch script: $CORE_PATCH_SCRIPT"; exit 1; }

# Always launch through srun when BUILD_USE_SRUN=1. Guard with FLUIDSIM_BUILD_INNER to avoid recursion.
if [[ "${BUILD_USE_SRUN}" == "1" && "${FLUIDSIM_BUILD_INNER:-0}" != "1" ]]; then
    # Prevent accidental reuse of stale allocation context from shell env.
    unset SLURM_JOB_ID SLURM_JOBID SLURM_STEP_ID SLURM_STEPID SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_JOB_NUM_NODES SLURM_NNODES

    exec srun \
        --clusters="$BUILD_CLUSTER" \
        --partition="$BUILD_PARTITION" \
        --time="$BUILD_TIME" \
        --ntasks=1 \
        --cpus-per-task="$BUILD_CPUS_PER_TASK" \
        --mem="$BUILD_MEM" \
        --chdir="$PROJECT_ROOT" \
        /usr/bin/env \
        FLUIDSIM_BUILD_INNER=1 \
        BUILD_USE_SRUN="$BUILD_USE_SRUN" \
        RECONFIGURE="$RECONFIGURE" \
        BUILD_CPUS_PER_TASK="$BUILD_CPUS_PER_TASK" \
        TARGET="$TARGET" \
        TOOLCHAIN_MODULE="$TOOLCHAIN_MODULE" \
        PYTHON_MODULE="$PYTHON_MODULE" \
        MPI_MODULE="$MPI_MODULE" \
        GPU_MODULE="$GPU_MODULE" \
        CMAKE_MODULE="$CMAKE_MODULE" \
        CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
        "$APP_DIR/build_gpu.sh"
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
    [[ -n "$GPU_MODULE" ]] && module load "$GPU_MODULE"
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
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc was not found in PATH." >&2
    echo "Load a CUDA toolkit module (set GPU_MODULE) and retry." >&2
    exit 1
fi

PYTHON_BIN="$(command -v python3)"
echo "Using python: $PYTHON_BIN"
echo "Using cmake: $(command -v cmake)"
cmake --version | head -n 1
echo "GPU_BACKEND: cuda (fixed)"
echo "CUDA_ARCHITECTURES: $CUDA_ARCHITECTURES"

# Venv policy (GPU): must already exist and be pre-populated.
# Reason: avoid pip installs inside GPU build jobs/restricted cluster environments.
# Bootstrap once with:
#   apps/shared/scripts/install_codegen_venv.sh
if [[ ! -x "$VENV/bin/python" || ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: Required codegen venv not found at $VENV" >&2
    echo "Create it once outside this build job and install sweepgen deps." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

if ! python - <<'PY'
import numpy, sympy, jinja2, lbmpy, pystencils, pystencilssfg, sweepgen
PY
then
    echo "ERROR: Missing codegen dependencies in $VENV." >&2
    echo "Install required packages in that venv and retry." >&2
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
        -DWALBERLA_BUILD_WITH_CUDA=ON
        -DWALBERLA_BUILD_WITH_HIP=OFF
    )
    if [[ -n "$CUDA_ARCHITECTURES" ]]; then
        CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES")
    fi
    cmake -S "$SRC_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
fi

cmake --build "$BUILD_DIR" --target "$TARGET" --parallel "$BUILD_CPUS_PER_TASK"

echo "Build OK: $BUILD_DIR/apps/FluidSim_gpu/FluidSim_gpu"
