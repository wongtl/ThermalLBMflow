#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

# Submit profiling jobs:
# - Nsight Systems timeline (nsys)
# - Nsight Compute kernel analysis (ncu)
# Use --pmode=nsys|ncu|both (required).
# ncu depends on nsys only when pmode=both.

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NSYS_WRAPPER="$APP_DIR/scripts/run_with_nsys.sh"
NCU_WRAPPER="$APP_DIR/scripts/run_with_ncu.sh"

if [[ ! -f "$APP_DIR/run_sim_gpu.sbatch" ]]; then
    echo "ERROR: sbatch script not found: $APP_DIR/run_sim_gpu.sbatch" >&2
    exit 1
fi

PMODE=""
SBATCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pmode=*)
            PMODE="${1#*=}"
            shift
            ;;
        --pmode)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --pmode requires a value (nsys|ncu|both)." >&2
                exit 1
            fi
            PMODE="$2"
            shift 2
            ;;
        *)
            SBATCH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$PMODE" ]]; then
    echo "ERROR: missing required --pmode (nsys|ncu|both)." >&2
    exit 1
fi

PMODE="$(echo "$PMODE" | tr '[:upper:]' '[:lower:]')"
case "$PMODE" in
    nsys)
        RUN_NSYS=1
        RUN_NCU=0
        ;;
    ncu)
        RUN_NSYS=0
        RUN_NCU=1
        ;;
    both)
        RUN_NSYS=1
        RUN_NCU=1
        ;;
    *)
        echo "ERROR: --pmode must be one of: nsys, ncu, both (got '$PMODE')." >&2
        exit 1
        ;;
esac

if [[ "$RUN_NSYS" == "1" && ! -x "$NSYS_WRAPPER" ]]; then
    echo "ERROR: Nsight Systems wrapper not found or not executable: $NSYS_WRAPPER" >&2
    exit 1
fi
if [[ "$RUN_NCU" == "1" && ! -x "$NCU_WRAPPER" ]]; then
    echo "ERROR: Nsight Compute wrapper not found or not executable: $NCU_WRAPPER" >&2
    exit 1
fi

VTKEVERY="${VTKEVERY:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
MPI_TRANSPORT_MODE="${MPI_TRANSPORT_MODE:-ob1_stable}"

SANITIZED_ENV=(
    env
    -u PYTHONHOME
    -u PYTHONPATH
    -u PYTHONUSERBASE
    -u PYTHONSTARTUP
    -u PYTHONBREAKPOINT
    -u PYTHONWARNINGS
    PYTHONNOUSERSITE=1
    PYTHONUTF8=1
    LANG=C.UTF-8
    LC_ALL=C.UTF-8
)

JOB_IDS=()

if [[ "$RUN_NSYS" == "1" ]]; then
    echo "Submitting Nsight Systems job..."
    echo "MPI_TRANSPORT_MODE: $MPI_TRANSPORT_MODE"
    NSYS_SUBMIT_ARGS=(--parsable "${SBATCH_ARGS[@]}" "$APP_DIR/run_sim_gpu.sbatch")
    NSYS_JOBINFO="$(
        "${SANITIZED_ENV[@]}" \
        FLUIDSIM_PROFILE_MODE="nsys" \
        VTKEVERY="$VTKEVERY" \
        CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
        MPI_TRANSPORT_MODE="$MPI_TRANSPORT_MODE" \
        sbatch "${NSYS_SUBMIT_ARGS[@]}"
    )"
    NSYS_JOBID="${NSYS_JOBINFO%%;*}"
    echo "NSYS job id: $NSYS_JOBID"
    JOB_IDS+=("$NSYS_JOBID")
fi

if [[ "$RUN_NCU" == "1" ]]; then
    NCU_SUBMIT_ARGS=(--parsable)
    if [[ "$RUN_NSYS" == "1" ]]; then
        echo "Submitting Nsight Compute job (afterok:${NSYS_JOBID})..."
        NCU_SUBMIT_ARGS+=(--dependency="afterok:${NSYS_JOBID}")
    else
        echo "Submitting Nsight Compute job..."
    fi
    NCU_SUBMIT_ARGS+=("${SBATCH_ARGS[@]}" "$APP_DIR/run_sim_gpu.sbatch")

    NCU_JOBINFO="$(
        "${SANITIZED_ENV[@]}" \
        FLUIDSIM_PROFILE_MODE="ncu" \
        VTKEVERY="$VTKEVERY" \
        CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
        MPI_TRANSPORT_MODE="$MPI_TRANSPORT_MODE" \
        sbatch "${NCU_SUBMIT_ARGS[@]}"
    )"
    NCU_JOBID="${NCU_JOBINFO%%;*}"
    echo "NCU job id: $NCU_JOBID"
    JOB_IDS+=("$NCU_JOBID")
fi

echo "Track jobs with:"
echo "  squeue -u \$USER"
echo "Inspect status with:"
JOB_ID_CSV="$(IFS=,; echo "${JOB_IDS[*]}")"
echo "  sacct -j ${JOB_ID_CSV} --format=JobID,State,ExitCode,Elapsed"
