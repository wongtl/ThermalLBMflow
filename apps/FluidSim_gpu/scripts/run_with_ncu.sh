#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: ncu not found in PATH." >&2
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
SIZE="${OMPI_COMM_WORLD_SIZE:-${SLURM_NTASKS:-1}}"

if [[ "$SIZE" -ne 1 ]]; then
    echo "ERROR: run_with_ncu.sh supports single-rank only (MPI world size must be 1, got $SIZE)." >&2
    exit 1
fi

NCU_REPORT_BASE="${FLUIDSIM_PROFILE_BASE_DIR}/${SLURM_JOB_ID:-manual}/ncu-report-r${RANK}"

mkdir -p "$(dirname "$NCU_REPORT_BASE")"

NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-50}"
if ! [[ "$NCU_LAUNCH_COUNT" =~ ^[0-9]+$ ]] || [[ "$NCU_LAUNCH_COUNT" -lt 1 ]]; then
    echo "ERROR: NCU_LAUNCH_COUNT must be a positive integer (got '$NCU_LAUNCH_COUNT')." >&2
    exit 1
fi
NCU_SET="${NCU_SET:-basic}"

CMD=(
    ncu
    --target-processes all
    --kernel-name-base demangled
    --launch-skip 0
    --launch-count "$NCU_LAUNCH_COUNT"
    -o "$NCU_REPORT_BASE"
)
if [[ -n "$NCU_SET" ]]; then
    CMD+=(--set "$NCU_SET")
fi

# Keep Nsight Compute isolated from user Python site packages/modules to avoid
# end-of-run Python codec/import noise on some cluster environments.
exec env \
    -u PYTHONHOME \
    -u PYTHONPATH \
    -u PYTHONUSERBASE \
    -u PYTHONSTARTUP \
    -u PYTHONBREAKPOINT \
    -u PYTHONWARNINGS \
    PYTHONNOUSERSITE=1 \
    PYTHONUTF8=1 \
    LANG="${LANG:-C.UTF-8}" \
    LC_ALL="${LC_ALL:-C.UTF-8}" \
    "${CMD[@]}" "$FLUIDSIM_REAL_EXE" "$@"
