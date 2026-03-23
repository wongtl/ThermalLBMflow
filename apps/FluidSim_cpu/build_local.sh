#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

# Project paths.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALBERLA_ROOT="$(cd "$APP_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$APP_DIR/../../.." && pwd)"
BUILD_DIR="$WALBERLA_ROOT/build-cpu"
REAL_EXE="$BUILD_DIR/apps/FluidSim_cpu/FluidSim_cpu"
VENV="${VENV:-$PROJECT_ROOT/venv-walberla-codegen}"

# Local build knobs.
BUILD_JOBS="${BUILD_JOBS:-1}"
RECONFIGURE="${RECONFIGURE:-0}"

if (( $# > 0 )); then
    echo "ERROR: build_local.sh does not accept positional arguments." >&2
    echo "Use environment knobs (for example BUILD_JOBS, RECONFIGURE, VENV)." >&2
    exit 1
fi

if [[ ! -f "$WALBERLA_ROOT/CMakeLists.txt" ]]; then
    echo "ERROR: Missing CMakeLists.txt in source dir: $WALBERLA_ROOT" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found in PATH." >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake not found in PATH." >&2
    exit 1
fi

# Prefer the shared codegen venv Python when available.
CODEGEN_PY="$(command -v python3)"
if [[ -x "$VENV/bin/python" ]]; then
    CODEGEN_PY="$VENV/bin/python"
fi

if [[ "$RECONFIGURE" == "1" ]]; then
    rm -rf "$BUILD_DIR"
fi

cmake -S "$WALBERLA_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DWALBERLA_BUILD_WITH_LTO=OFF \
    -DWALBERLA_BUILD_WITH_MPI=ON \
    -DWALBERLA_BUILD_WITH_OPENMP=ON \
    -DWALBERLA_BUILD_WITH_OPENMESH=ON \
    -DWALBERLA_BUILD_WITH_CODEGEN=ON \
    -DWALBERLA_ENABLE_SWEEPGEN=ON \
    -DWALBERLA_SWEEPGEN_MANAGED_VENV=OFF \
    -DWALBERLA_BUILD_TESTS=OFF \
    -DWALBERLA_BUILD_TUTORIALS=OFF \
    -DPython_EXECUTABLE="$CODEGEN_PY" \
    -DPystencilsSfg_PYTHON_INTERPRETER="$CODEGEN_PY" \
    -DWALBERLA_CODEGEN_PYTHON="$CODEGEN_PY"

cmake --build "$BUILD_DIR" --target FluidSim_cpu --parallel "$BUILD_JOBS" 2>&1 | sed '/^ninja: no work to do\.?$/d'

if [[ ! -x "$REAL_EXE" ]]; then
    echo "ERROR: FluidSim executable not found after build." >&2
    echo "Checked: $REAL_EXE" >&2
    exit 1
fi

echo "Build OK: $REAL_EXE"
