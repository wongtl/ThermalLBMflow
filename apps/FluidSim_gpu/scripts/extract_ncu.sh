#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

SRC_BASE="${SRC_BASE:-$APP_DIR/output/profiling}"
DST_BASE="${DST_BASE:-$APP_DIR/../storage/profiling}"

TOOLCHAIN_MODULE="${TOOLCHAIN_MODULE:-}"
GPU_MODULE="${GPU_MODULE:-}"

USE_SRUN="${USE_SRUN:-1}"
BUILD_CLUSTER="${BUILD_CLUSTER:-}"
BUILD_PARTITION="${BUILD_PARTITION:-}"
BUILD_TIME="${BUILD_TIME:-01:00:00}"
BUILD_MEM="${BUILD_MEM:-8G}"
BUILD_CPUS_PER_TASK="${BUILD_CPUS_PER_TASK:-4}"

JOBID_FILTER=""

usage() {
    cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --local            Force local execution (disable srun compute dispatch).
  --jobid <id>       Extract only one profiling job directory.
  -h, --help         Show this help.

Environment overrides:
  SRC_BASE             Source profiling dir (default: $APP_DIR/output/profiling)
  DST_BASE             Destination dir (default: $APP_DIR/../storage/profiling)
  TOOLCHAIN_MODULE     Optional toolchain module to load before extraction
  GPU_MODULE           Optional CUDA/Nsight module to load before extraction
  USE_SRUN             Dispatch to srun when not already in Slurm allocation (default: 1)
  BUILD_CLUSTER        Optional srun cluster
  BUILD_PARTITION      Optional srun partition
  BUILD_TIME           srun walltime (default: 01:00:00)
  BUILD_MEM            srun memory request (default: 8G)
  BUILD_CPUS_PER_TASK  srun cpus-per-task (default: 4)
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            USE_SRUN=0
            shift
            ;;
        --jobid)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --jobid requires a value." >&2
                exit 1
            fi
            JOBID_FILTER="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

run_extract() {
    if ! command -v ncu >/dev/null 2>&1; then
        if command -v module >/dev/null 2>&1 || [[ -f /etc/profile.d/modules.sh ]]; then
            if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
                # shellcheck disable=SC1091
                source /etc/profile.d/modules.sh
            fi
            if command -v module >/dev/null 2>&1; then
                module purge
                [[ -n "$TOOLCHAIN_MODULE" ]] && module load "$TOOLCHAIN_MODULE"
                [[ -n "$GPU_MODULE" ]] && module load "$GPU_MODULE"
            fi
        fi
    fi

    if ! command -v ncu >/dev/null 2>&1; then
        echo "ERROR: ncu not found in PATH after module setup." >&2
        exit 1
    fi

    mkdir -p "$DST_BASE"

    run_ncu_import() {
        local mode="$1"
        local report_file="$2"
        local err_file="$3"

        local page_args=()
        if [[ "$mode" == "raw" ]]; then
            page_args=(--page raw)
        fi

        if env \
            -u PYTHONHOME \
            -u PYTHONPATH \
            -u PYTHONUSERBASE \
            -u PYTHONSTARTUP \
            -u PYTHONBREAKPOINT \
            -u PYTHONWARNINGS \
            PYTHONNOUSERSITE=1 \
            PYTHONUTF8=1 \
            LANG=C.UTF-8 \
            LC_ALL=C.UTF-8 \
            ncu --import "$ncu_rep" "${page_args[@]}" --csv > "$report_file" 2> "$err_file"; then
            echo "Generated: $report_file"
            if [[ -s "$err_file" ]]; then
                echo "WARN: non-empty stderr: $err_file"
            fi
        else
            echo "WARN: failed NCU import mode '$mode' for $ncu_rep (see $err_file)" >&2
        fi
    }

    local found=0
    for JDIR in "$SRC_BASE"/*; do
        [[ -d "$JDIR" ]] || continue

        local jobid
        jobid="$(basename "$JDIR")"

        if [[ -n "$JOBID_FILTER" && "$jobid" != "$JOBID_FILTER" ]]; then
            continue
        fi

        local ncu_rep
        ncu_rep="$JDIR/ncu-report-r0.ncu-rep"
        [[ -f "$ncu_rep" ]] || continue

        found=1

        local job_dst
        job_dst="$DST_BASE/$jobid"
        mkdir -p "$job_dst"
        if find "$job_dst" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
            find "$job_dst" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
            echo "Cleared existing output directory: $job_dst"
        fi

        echo "[NCU] $ncu_rep -> $job_dst"

        run_ncu_import "default" "$job_dst/ncu_default.csv" "$job_dst/ncu_default.err"
        run_ncu_import "raw"     "$job_dst/ncu_raw.csv"     "$job_dst/ncu_raw.err"
    done

    if [[ "$found" -eq 0 ]]; then
        if [[ -n "$JOBID_FILTER" ]]; then
            echo "No NCU report found for jobid '$JOBID_FILTER' under $SRC_BASE"
        else
            echo "No NCU reports found under $SRC_BASE"
        fi
        exit 0
    fi

    echo "NCU extraction completed."
}

if [[ "$USE_SRUN" == "1" ]] && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if command -v srun >/dev/null 2>&1; then
        SRUN_CMD=(
            srun
            --time "$BUILD_TIME"
            --ntasks 1
            --cpus-per-task "$BUILD_CPUS_PER_TASK"
            --job-name extract_ncu
        )
        [[ -n "$BUILD_CLUSTER" ]] && SRUN_CMD+=(--clusters "$BUILD_CLUSTER")
        [[ -n "$BUILD_PARTITION" ]] && SRUN_CMD+=(--partition "$BUILD_PARTITION")
        [[ -n "$BUILD_MEM" ]] && SRUN_CMD+=(--mem "$BUILD_MEM")
        SRUN_CMD+=(
            /usr/bin/env
            USE_SRUN=0
            SRC_BASE="$SRC_BASE"
            DST_BASE="$DST_BASE"
            TOOLCHAIN_MODULE="$TOOLCHAIN_MODULE"
            GPU_MODULE="$GPU_MODULE"
            BUILD_CLUSTER="$BUILD_CLUSTER"
            BUILD_PARTITION="$BUILD_PARTITION"
            BUILD_TIME="$BUILD_TIME"
            BUILD_MEM="$BUILD_MEM"
            BUILD_CPUS_PER_TASK="$BUILD_CPUS_PER_TASK"
            "$SELF_PATH"
        )
        if [[ -n "$JOBID_FILTER" ]]; then
            SRUN_CMD+=(--jobid "$JOBID_FILTER")
        fi
        echo "Dispatching NCU extraction via srun: time=$BUILD_TIME cpus=$BUILD_CPUS_PER_TASK mem=${BUILD_MEM:-site-default}"
        [[ -n "$BUILD_CLUSTER" ]] && echo "  cluster=$BUILD_CLUSTER"
        [[ -n "$BUILD_PARTITION" ]] && echo "  partition=$BUILD_PARTITION"
        exec "${SRUN_CMD[@]}"
    else
        echo "WARN: USE_SRUN=1 but srun not found; running locally." >&2
    fi
fi

run_extract
