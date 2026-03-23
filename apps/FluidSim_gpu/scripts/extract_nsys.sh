#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

SRC_BASE="${SRC_BASE:-$APP_DIR/output/profiling}"
DST_BASE="${DST_BASE:-$APP_DIR/../storage/profiling}"

TOOLCHAIN_MODULE="${TOOLCHAIN_MODULE:-foss/2024a}"
GPU_MODULE="${GPU_MODULE:-CUDA/12.9.0}"

USE_SRUN="${USE_SRUN:-1}"
BUILD_CLUSTER="${BUILD_CLUSTER:-htc}"
BUILD_PARTITION="${BUILD_PARTITION:-interactive}"
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
  TOOLCHAIN_MODULE     Module to load before extraction (default: foss/2024a)
  GPU_MODULE           CUDA module to load before extraction (default: CUDA/12.9.0)
  USE_SRUN             Dispatch to srun when not already in Slurm allocation (default: 1)
  BUILD_CLUSTER        srun cluster (default: htc)
  BUILD_PARTITION      srun partition (default: interactive)
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
    if ! command -v nsys >/dev/null 2>&1; then
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

    if ! command -v nsys >/dev/null 2>&1; then
        echo "ERROR: nsys not found in PATH after module setup." >&2
        exit 1
    fi

    mkdir -p "$DST_BASE"

    run_nsys_report() {
        local report="$1"
        local report_file="$2"
        local err_file="$3"

        if nsys stats --report "$report" --format csv "$nsys_rep" > "$report_file" 2> "$err_file"; then
            echo "Generated: $report_file"
            if [[ -s "$err_file" ]]; then
                echo "WARN: non-empty stderr: $err_file"
            fi
        else
            echo "WARN: failed report '$report' for $nsys_rep (see $err_file)" >&2
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

        local nsys_rep
        nsys_rep="$JDIR/nsys-report-r0.nsys-rep"
        [[ -f "$nsys_rep" ]] || continue

        found=1

        local job_dst
        job_dst="$DST_BASE/$jobid"
        mkdir -p "$job_dst"
        if find "$job_dst" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
            find "$job_dst" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
            echo "Cleared existing output directory: $job_dst"
        fi

        echo "[NSYS] $nsys_rep -> $job_dst"

        # Remove stale sqlite sidecar so stats regenerates from report cleanly.
        rm -f "${nsys_rep%.nsys-rep}.sqlite"

        run_nsys_report "cuda_gpu_kern_sum"     "$job_dst/nsys_cuda_gpu_kern_sum.csv"     "$job_dst/nsys_cuda_gpu_kern_sum.err"
        run_nsys_report "cuda_api_sum"          "$job_dst/nsys_cuda_api_sum.csv"          "$job_dst/nsys_cuda_api_sum.err"
        run_nsys_report "cuda_gpu_mem_time_sum" "$job_dst/nsys_cuda_gpu_mem_time_sum.csv" "$job_dst/nsys_cuda_gpu_mem_time_sum.err"
        run_nsys_report "nvtx_sum"              "$job_dst/nsys_nvtx_sum.csv"              "$job_dst/nsys_nvtx_sum.err"
        run_nsys_report "cuda_kern_exec_sum"    "$job_dst/nsys_cuda_kern_exec_sum.csv"    "$job_dst/nsys_cuda_kern_exec_sum.err"
        run_nsys_report "nvtx_gpu_proj_sum"     "$job_dst/nsys_nvtx_gpu_proj_sum.csv"     "$job_dst/nsys_nvtx_gpu_proj_sum.err"
        run_nsys_report "cuda_api_gpu_sum"      "$job_dst/nsys_cuda_api_gpu_sum.csv"      "$job_dst/nsys_cuda_api_gpu_sum.err"
    done

    if [[ "$found" -eq 0 ]]; then
        if [[ -n "$JOBID_FILTER" ]]; then
            echo "No NSYS report found for jobid '$JOBID_FILTER' under $SRC_BASE"
        else
            echo "No NSYS reports found under $SRC_BASE"
        fi
        exit 0
    fi

    echo "NSYS extraction completed."
}

if [[ "$USE_SRUN" == "1" ]] && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if command -v srun >/dev/null 2>&1; then
        SRUN_CMD=(
            srun
            --clusters "$BUILD_CLUSTER"
            --partition "$BUILD_PARTITION"
            --time "$BUILD_TIME"
            --ntasks 1
            --cpus-per-task "$BUILD_CPUS_PER_TASK"
            --mem "$BUILD_MEM"
            --job-name extract_nsys
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

        echo "Dispatching NSYS extraction via srun: cluster=$BUILD_CLUSTER partition=$BUILD_PARTITION time=$BUILD_TIME cpus=$BUILD_CPUS_PER_TASK mem=$BUILD_MEM"
        exec "${SRUN_CMD[@]}"
    else
        echo "WARN: USE_SRUN=1 but srun not found; running locally." >&2
    fi
fi

run_extract
