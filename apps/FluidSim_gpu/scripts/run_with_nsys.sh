#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys not found in PATH." >&2
    exit 1
fi
if [[ -z "${FLUIDSIM_REAL_EXE:-}" ]]; then
    echo "ERROR: FLUIDSIM_REAL_EXE must point to the real FluidSim_gpu executable." >&2
    exit 1
fi
if [[ -z "${FLUIDSIM_PROFILE_BASE_DIR:-}" ]]; then
    echo "ERROR: FLUIDSIM_PROFILE_BASE_DIR must be set." >&2
    exit 1
fi

RANK="${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}"
NSYS_REPORT_BASE="${FLUIDSIM_PROFILE_BASE_DIR}/${SLURM_JOB_ID:-manual}/nsys-report-r${RANK}"

mkdir -p "$(dirname "$NSYS_REPORT_BASE")"

CMD=(
    nsys
    profile
    --output "$NSYS_REPORT_BASE"
    --force-overwrite true
    --trace cuda,nvtx
    --sample none
    --stats false
    --cuda-event-trace=false
)

exec "${CMD[@]}" "$FLUIDSIM_REAL_EXE" "$@"
