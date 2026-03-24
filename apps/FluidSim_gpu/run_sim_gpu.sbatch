#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
#SBATCH --job-name=fs_gpu
## Example submit (set partition/account/GPU options for your site):
## sbatch --partition=<partition> --gres=gpu:<N> run_sim_gpu.sbatch
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Submit context checks.

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run_sim_gpu.sbatch must be submitted with sbatch." >&2
    exit 1
fi

# Slurm executes a spool copy of this script; derive paths from submit directory.
# Contract: submit from walberla/apps/FluidSim_gpu.
if ! APP_DIR="$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && pwd)"; then
    echo "ERROR: Could not resolve submit directory from SLURM_SUBMIT_DIR/PWD." >&2
    exit 1
fi
if [[ ! -f "$APP_DIR/run_sim_gpu.sbatch" ]]; then
    echo "ERROR: submit from walberla/apps/FluidSim_gpu (current submit dir: $APP_DIR)." >&2
    exit 1
fi

# Path and runtime configuration.

# App paths.
REAL_EXE="$APP_DIR/../../../build-gpu/apps/FluidSim_gpu/FluidSim_gpu"
SHARED_DIR="$APP_DIR/../shared"
PARAMS="$SHARED_DIR/params/FluidSim.prm"

# Module/loading knobs.
TOOLCHAIN_MODULE="${TOOLCHAIN_MODULE:-}"
GPU_MODULE="${GPU_MODULE:-}"
MPI_MODULE="${MPI_MODULE:-}"

# Runtime knobs (launcher-owned, environment-overridable).
TIMESTEPS="${TIMESTEPS:-5000001}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000000}"
VTKEVERY="${VTKEVERY:-5000000}"
VTKINIT="${VTKINIT:-0}"
VTKMESHONLY="${VTKMESHONLY:-false}"
FLUIDSIM_PROFILE_MODE="${FLUIDSIM_PROFILE_MODE:-none}"
MPI_TRANSPORT_MODE="${MPI_TRANSPORT_MODE:-ob1_stable}"

# Paths/state.
SCRATCH_BASE="${SCRATCH:-${TMPDIR:-/tmp/$USER}}"
RUNROOT="${SCRATCH_BASE}/fluidsim_gpu_${SLURM_JOB_ID}"
RUN_OUTPUT_DIR="${RUNROOT}/output"
RUN_STARTED=0

OUTDIR="$APP_DIR/output/${SLURM_JOB_ID}"
SLURM_LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SLURM_LOG_BASENAME="${SLURM_JOB_NAME:-fs_gpu}-${SLURM_JOB_ID}"
SLURM_STDOUT_PATH="${SLURM_LOG_DIR}/${SLURM_LOG_BASENAME}.out"
SLURM_STDERR_PATH="${SLURM_LOG_DIR}/${SLURM_LOG_BASENAME}.err"
PROFILE_BASE_DIR="$APP_DIR/output/profiling"
EXE="$REAL_EXE"

# Profile launcher selection.
case "$FLUIDSIM_PROFILE_MODE" in
    none)
        unset FLUIDSIM_REAL_EXE FLUIDSIM_PROFILE_BASE_DIR 2>/dev/null || true
        ;;
    nsys)
        EXE="$APP_DIR/scripts/run_with_nsys.sh"
        export FLUIDSIM_REAL_EXE="$REAL_EXE"
        export FLUIDSIM_PROFILE_BASE_DIR="$PROFILE_BASE_DIR"
        ;;
    ncu)
        EXE="$APP_DIR/scripts/run_with_ncu.sh"
        export FLUIDSIM_REAL_EXE="$REAL_EXE"
        export FLUIDSIM_PROFILE_BASE_DIR="$PROFILE_BASE_DIR"
        ;;
    *)
        echo "ERROR: Invalid FLUIDSIM_PROFILE_MODE='${FLUIDSIM_PROFILE_MODE}'." >&2
        echo "Valid values: none, nsys, ncu." >&2
        exit 1
        ;;
esac
export FLUIDSIM_PROFILE_MODE

case "$MPI_TRANSPORT_MODE" in
    ob1_stable | ob1_smcuda)
        ;;
    ob1_auto_btl_test)
        echo "ERROR: MPI_TRANSPORT_MODE=ob1_auto_btl_test was removed in phase-2 (no measurable gain over ob1_stable)." >&2
        echo "Use MPI_TRANSPORT_MODE=ob1_smcuda." >&2
        exit 1
        ;;
    *)
        echo "ERROR: Invalid MPI_TRANSPORT_MODE='${MPI_TRANSPORT_MODE}'." >&2
        echo "Valid values: ob1_stable, ob1_smcuda." >&2
        exit 1
        ;;
esac

# Naming policy: direct Slurm outputs and directly processed Slurm-derived values
# keep the `SLURM_` prefix (for example SLURM_NODE_COUNT, SLURM_CPUS_ON_NODE_NORM).
SLURM_NODE_COUNT="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}"

# Slurm resource preflight.
if ! [[ "${SLURM_NTASKS:-}" =~ ^[0-9]+$ ]] || (( SLURM_NTASKS < 1 )); then
    echo "ERROR: Invalid SLURM_NTASKS='${SLURM_NTASKS:-unset}'." >&2
    echo "Hint: submit with explicit task/GPU mapping (e.g. --ntasks-per-gpu=1 --gres=gpu:<N>)." >&2
    exit 1
fi

if ! [[ "$SLURM_NODE_COUNT" =~ ^[0-9]+$ ]] || (( SLURM_NODE_COUNT < 1 )); then
    echo "ERROR: Could not determine a valid node count from Slurm (SLURM_JOB_NUM_NODES='${SLURM_JOB_NUM_NODES:-unset}', SLURM_NNODES='${SLURM_NNODES:-unset}')." >&2
    exit 1
fi
if (( SLURM_NODE_COUNT != 1 )); then
    echo "ERROR: run_sim_gpu.sbatch supports single-node runs only (got nodes=${SLURM_NODE_COUNT})." >&2
    echo "Hint: submit with --nodes=1 and keep all tasks/GPUs on that node." >&2
    exit 1
fi

if [[ "$FLUIDSIM_PROFILE_MODE" == "ncu" ]] && (( SLURM_NTASKS != 1 )); then
    echo "ERROR: FLUIDSIM_PROFILE_MODE=ncu requires single-rank launch (SLURM_NTASKS must be 1, got ${SLURM_NTASKS})." >&2
    echo "Hint: submit with one GPU/rank, e.g. --gres=gpu:1 --ntasks-per-gpu=1." >&2
    exit 1
fi

case "$VTKMESHONLY" in
    0|1|true|false|TRUE|FALSE|True|False)
        ;;
    *)
        echo "ERROR: VTKMESHONLY must be a boolean value (0/1/false/true) (got '$VTKMESHONLY')." >&2
        exit 1
        ;;
esac

SLURM_CPUS_PER_RANK=0
if [[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[0-9]+$ ]] && (( SLURM_CPUS_PER_TASK >= 1 )); then
    SLURM_CPUS_PER_RANK="${SLURM_CPUS_PER_TASK}"
elif [[ "${SLURM_CPUS_PER_GPU:-}" =~ ^[0-9]+$ ]] && (( SLURM_CPUS_PER_GPU >= 1 )); then
    SLURM_CPUS_PER_RANK="${SLURM_CPUS_PER_GPU}"
else
    echo "ERROR: Could not determine a numeric per-rank CPU count from Slurm." >&2
    echo "       SLURM_CPUS_PER_TASK='${SLURM_CPUS_PER_TASK:-unset}', SLURM_CPUS_PER_GPU='${SLURM_CPUS_PER_GPU:-unset}'." >&2
    echo "Hint: submit with --cpus-per-gpu=<N> (or --cpus-per-task=<N>)." >&2
    exit 1
fi

normalize_slurm_gpu_list() {
    local raw="$1"
    local token
    local out=""
    local i
    local -a _gpu_tokens
    raw="${raw//[[:space:]]/}"
    raw="${raw//;/,}"
    IFS=',' read -r -a _gpu_tokens <<< "$raw"
    for token in "${_gpu_tokens[@]}"; do
        [[ -z "$token" ]] && continue

        # Common Slurm formats include optional node/gres prefixes.
        if [[ "$token" == *:* ]]; then
            token="${token##*:}"
        fi
        token="${token#gpu[}"
        token="${token#GPU[}"
        token="${token#[}"
        token="${token%]}"

        [[ -z "$token" || "$token" == "(null)" || "$token" == "N/A" ]] && continue

        # Expand numeric ranges (for example "0-3") into "0,1,2,3".
        if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            for (( i=${BASH_REMATCH[1]}; i<=${BASH_REMATCH[2]}; ++i )); do
                if [[ -z "$out" ]]; then
                    out="$i"
                else
                    out="${out},${i}"
                fi
            done
            continue
        fi

        if [[ -z "$out" ]]; then
            out="$token"
        else
            out="${out},${token}"
        fi
    done
    printf '%s' "$out"
}

SLURM_VISIBLE_GPUS_RAW=""
SLURM_VISIBLE_GPUS_SOURCE=""
if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    SLURM_VISIBLE_GPUS_RAW="${SLURM_STEP_GPUS}"
    SLURM_VISIBLE_GPUS_SOURCE="SLURM_STEP_GPUS"
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    SLURM_VISIBLE_GPUS_RAW="${SLURM_JOB_GPUS}"
    SLURM_VISIBLE_GPUS_SOURCE="SLURM_JOB_GPUS"
fi

# Normalize CUDA visibility from Slurm GPU lists when CUDA_VISIBLE_DEVICES is missing.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "NoDevFiles" ]]; then
    if [[ -n "$SLURM_VISIBLE_GPUS_RAW" ]]; then
        _derived_cuda_visible_devices="$(normalize_slurm_gpu_list "$SLURM_VISIBLE_GPUS_RAW")"
        if [[ -n "$_derived_cuda_visible_devices" ]]; then
            export CUDA_VISIBLE_DEVICES="$_derived_cuda_visible_devices"
            echo "INFO: CUDA_VISIBLE_DEVICES was unset; derived from ${SLURM_VISIBLE_GPUS_SOURCE}='${SLURM_VISIBLE_GPUS_RAW}' -> '${CUDA_VISIBLE_DEVICES}'."
        fi
    fi
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "NoDevFiles" ]]; then
    if [[ -z "${SLURM_GPUS_ON_NODE:-}" && -z "${SLURM_STEP_GPUS:-}" && -z "${SLURM_JOB_GPUS:-}" ]]; then
        echo "WARNING: GPU visibility metadata missing (CUDA_VISIBLE_DEVICES, SLURM_STEP_GPUS, SLURM_JOB_GPUS, SLURM_GPUS_ON_NODE)." >&2
    fi
fi

GPU_COUNT=0
CUDA_VISIBLE_GPU_COUNT=0
HAS_CUDA_VISIBLE_GPU_COUNT=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    IFS=',' read -r -a _visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    for _gpu_id in "${_visible_gpus[@]}"; do
        [[ -n "$_gpu_id" ]] && (( CUDA_VISIBLE_GPU_COUNT += 1 ))
    done
    HAS_CUDA_VISIBLE_GPU_COUNT=1
    GPU_COUNT="$CUDA_VISIBLE_GPU_COUNT"
fi

if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
    if (( HAS_CUDA_VISIBLE_GPU_COUNT == 1 )) && (( CUDA_VISIBLE_GPU_COUNT != SLURM_GPUS_ON_NODE )); then
        echo "ERROR: GPU allocation/visibility mismatch detected." >&2
        echo "       SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE} but CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' (count=${CUDA_VISIBLE_GPU_COUNT})." >&2
        echo "       Check Slurm GPU binding/cgroup configuration before rerunning." >&2
        exit 1
    fi
    GPU_COUNT="${SLURM_GPUS_ON_NODE}"
fi

if (( GPU_COUNT > 0 )); then
    if (( SLURM_NTASKS > GPU_COUNT )); then
        echo "ERROR: SLURM_NTASKS=$SLURM_NTASKS exceeds allocated/visible GPU count=$GPU_COUNT." >&2
        echo "Hint: submit with matching resources, e.g. --ntasks-per-gpu=1 --gres=gpu:$SLURM_NTASKS" >&2
        exit 1
    fi
    if (( SLURM_NTASKS != GPU_COUNT )); then
        echo "ERROR: one-rank-per-GPU policy requires SLURM_NTASKS == GPU_COUNT (SLURM_NTASKS=$SLURM_NTASKS, GPUs=$GPU_COUNT)." >&2
        exit 1
    fi
else
    if (( SLURM_NTASKS > 1 )); then
        echo "ERROR: Unable to validate one-rank-per-GPU for multi-rank launch." >&2
        echo "       GPU count is unavailable (SLURM_GPUS_ON_NODE, CUDA_VISIBLE_DEVICES, SLURM_STEP_GPUS, SLURM_JOB_GPUS)." >&2
        echo "       Refusing to run with SLURM_NTASKS=$SLURM_NTASKS without detectable GPU visibility." >&2
        exit 1
    fi
    echo "WARNING: Unable to validate SLURM_NTASKS against GPUs (SLURM_GPUS_ON_NODE/CUDA_VISIBLE_DEVICES/SLURM_STEP_GPUS/SLURM_JOB_GPUS unavailable)."
fi

SLURM_CPUS_ON_NODE_NORM="${SLURM_CPUS_ON_NODE:-}"
SLURM_CPUS_ON_NODE_NORM="${SLURM_CPUS_ON_NODE_NORM%%(*}"
SLURM_CPUS_ON_NODE_NORM="${SLURM_CPUS_ON_NODE_NORM//[[:space:]]/}"
if [[ "$SLURM_CPUS_ON_NODE_NORM" =~ ^[0-9]+$ ]] && (( SLURM_CPUS_ON_NODE_NORM < SLURM_NTASKS * SLURM_CPUS_PER_RANK )); then
    echo "ERROR: Insufficient CPU allocation for launch mapping." >&2
    echo "       Required SLURM_NTASKS*SLURM_CPUS_PER_RANK=${SLURM_NTASKS}*${SLURM_CPUS_PER_RANK}=$((SLURM_NTASKS * SLURM_CPUS_PER_RANK)), but SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset} (normalized=${SLURM_CPUS_ON_NODE_NORM})." >&2
    echo "Hint: request more CPUs (e.g. --cpus-per-gpu / --cpus-per-task) or reduce task count." >&2
    exit 1
fi
if ! [[ "$SLURM_CPUS_ON_NODE_NORM" =~ ^[0-9]+$ ]]; then
    echo "WARNING: Could not parse SLURM_CPUS_ON_NODE='${SLURM_CPUS_ON_NODE:-unset}' for preflight mapping check." >&2
    if (( SLURM_NTASKS > 1 )); then
        echo "         Multi-rank launch continues without this guard; mapping errors may surface later from mpirun." >&2
    fi
fi

# Runtime validation.
if ! [[ "$CHECKPOINT_EVERY" =~ ^[0-9]+$ ]]; then
    echo "ERROR: CHECKPOINT_EVERY must be a non-negative integer (got '$CHECKPOINT_EVERY')." >&2
    exit 1
fi
if ! [[ "$VTKEVERY" =~ ^[0-9]+$ ]]; then
    echo "ERROR: VTKEVERY must be a non-negative integer (got '$VTKEVERY')." >&2
    exit 1
fi
if ! [[ "$VTKINIT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: VTKINIT must be a non-negative integer (got '$VTKINIT')." >&2
    exit 1
fi
if ! [[ "$TIMESTEPS" =~ ^[0-9]+$ ]] || (( TIMESTEPS < 1 )); then
    echo "ERROR: TIMESTEPS must be a positive integer (got '$TIMESTEPS')." >&2
    exit 1
fi

# This launcher owns all app runtime flags.
if (( $# > 0 )); then
    echo "ERROR: run_sim_gpu.sbatch does not accept extra app arguments." >&2
    echo "Use environment knobs (for example TIMESTEPS, CHECKPOINT_EVERY, VTKEVERY, VTKINIT)." >&2
    exit 1
fi

# Input and executable validation.
if [[ ! -f "$PARAMS" ]]; then
    echo "ERROR: Parameter file not found: $PARAMS" >&2
    exit 1
fi
if [[ ! -x "$REAL_EXE" ]]; then
    echo "ERROR: GPU executable not found: $REAL_EXE" >&2
    echo "Build FluidSim_gpu first with build_gpu.sh." >&2
    exit 1
fi
if [[ ! -x "$EXE" ]]; then
    echo "ERROR: Launcher executable not found: $EXE" >&2
    exit 1
fi

# Environment setup.
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
    module purge
    [[ -n "$TOOLCHAIN_MODULE" ]] && module load "$TOOLCHAIN_MODULE"
    [[ -n "$GPU_MODULE" ]] && module load "$GPU_MODULE"
    [[ -n "$MPI_MODULE" ]] && module load "$MPI_MODULE"
fi

# Tool availability checks.
if ! command -v mpirun >/dev/null 2>&1; then
    echo "ERROR: mpirun was not found in PATH." >&2
    echo "Hint: load an MPI runtime module via MPI_MODULE." >&2
    exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync not found in PATH (required for output staging)." >&2
    exit 1
fi

has_ompi_component() {
    local framework="$1"
    local component="$2"
    if ! command -v ompi_info >/dev/null 2>&1; then
        return 1
    fi
    ompi_info --parsable --all 2>/dev/null | grep -qiE "mca:${framework}:${component}:"
}

has_ompi_param() {
    local framework="$1"
    local component="$2"
    local param="$3"
    if ! command -v ompi_info >/dev/null 2>&1; then
        return 1
    fi
    ompi_info --parsable --all 2>/dev/null | grep -qiE "mca:${framework}:${component}:param:${param}:"
}

configure_mpi_transport_mode() {
    if (( SLURM_NTASKS <= 1 )); then
        return
    fi

    case "$MPI_TRANSPORT_MODE" in
        ob1_stable)
            if [[ -n "${OMPI_MCA_pml:-}" && "${OMPI_MCA_pml}" != "ob1" ]]; then
                echo "Overriding OMPI_MCA_pml='${OMPI_MCA_pml}' -> 'ob1'"
            fi
            if [[ -n "${OMPI_MCA_btl:-}" && "${OMPI_MCA_btl}" != "self,vader,tcp" ]]; then
                echo "Overriding OMPI_MCA_btl='${OMPI_MCA_btl}' -> 'self,vader,tcp'"
            fi
            if [[ -n "${OMPI_MCA_btl_vader_single_copy_mechanism:-}" && "${OMPI_MCA_btl_vader_single_copy_mechanism}" != "none" ]]; then
                echo "Overriding OMPI_MCA_btl_vader_single_copy_mechanism='${OMPI_MCA_btl_vader_single_copy_mechanism}' -> 'none'"
            fi
            export OMPI_MCA_pml="ob1"
            export OMPI_MCA_btl="self,vader,tcp"
            export OMPI_MCA_btl_vader_single_copy_mechanism="none"
            unset OMPI_MCA_btl_smcuda_use_cuda_ipc 2>/dev/null || true
            ;;
        ob1_smcuda)
            if ! has_ompi_component "btl" "smcuda"; then
                echo "ERROR: MPI_TRANSPORT_MODE=ob1_smcuda requested, but OpenMPI component 'btl:smcuda' is unavailable." >&2
                echo "Hint: run 'ompi_info --parsable --all | grep -i \"mca:btl:smcuda\"' to verify your MPI build." >&2
                exit 1
            fi
            if [[ -n "${OMPI_MCA_pml:-}" && "${OMPI_MCA_pml}" != "ob1" ]]; then
                echo "Overriding OMPI_MCA_pml='${OMPI_MCA_pml}' -> 'ob1'"
            fi
            if [[ -n "${OMPI_MCA_btl:-}" && "${OMPI_MCA_btl}" != "self,smcuda,tcp" ]]; then
                echo "Overriding OMPI_MCA_btl='${OMPI_MCA_btl}' -> 'self,smcuda,tcp'"
            fi
            export OMPI_MCA_pml="ob1"
            export OMPI_MCA_btl="self,smcuda,tcp"
            if has_ompi_param "btl" "smcuda" "btl_smcuda_use_cuda_ipc"; then
                export OMPI_MCA_btl_smcuda_use_cuda_ipc=1
            else
                unset OMPI_MCA_btl_smcuda_use_cuda_ipc 2>/dev/null || true
            fi
            unset OMPI_MCA_btl_vader_single_copy_mechanism 2>/dev/null || true
            ;;
    esac
}

# Output staging helpers.

stage_run_output() {
    if [[ -d "$RUN_OUTPUT_DIR" ]]; then
        echo "Staging run output: $RUN_OUTPUT_DIR -> $OUTDIR"
        rsync -a "$RUN_OUTPUT_DIR"/ "$OUTDIR"/
        return $?
    fi
    return 1
}

stage_slurm_logs() {
    local moved_any=0

    if [[ -f "$SLURM_STDOUT_PATH" ]]; then
        mv -f "$SLURM_STDOUT_PATH" "$OUTDIR/slurm-${SLURM_JOB_ID}.out"
        moved_any=1
    fi
    if [[ -f "$SLURM_STDERR_PATH" ]]; then
        mv -f "$SLURM_STDERR_PATH" "$OUTDIR/slurm-${SLURM_JOB_ID}.err"
        moved_any=1
    fi

    if (( moved_any == 1 )); then
        echo "Moved Slurm stdout/stderr to $OUTDIR/slurm-${SLURM_JOB_ID}.out/.err"
    fi
}

on_exit() {
    local rc=$?
    local stage_rc=0

    trap - EXIT
    set +e

    mkdir -p "$OUTDIR"
    stage_slurm_logs

    if (( RUN_STARTED == 1 )); then
        if [[ -d "$RUN_OUTPUT_DIR" ]]; then
            stage_run_output
            stage_rc=$?
        else
            if (( rc != 0 )); then
                echo "Staging skipped: run failed before output creation ($RUN_OUTPUT_DIR)."
                stage_rc=0
            else
                echo "ERROR: No run output directory found at $RUN_OUTPUT_DIR"
                stage_rc=1
            fi
        fi
    fi

    set -e
    if (( rc == 0 )) && (( stage_rc != 0 )); then
        rc=2
    fi
    exit "$rc"
}

# Runtime metadata and launch.

trap on_exit EXIT

mkdir -p "$RUNROOT" "$OUTDIR"

exec > >(tee -a "$OUTDIR/run_sim_gpu-${SLURM_JOB_ID}.log") 2>&1

CUDA_VERSION_INFO="${GPU_MODULE:-<none>}"
if command -v nvcc >/dev/null 2>&1; then
    _nvcc_release="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([^,]*\),.*/\1/p' | head -n1 || true)"
    if [[ -n "$_nvcc_release" ]]; then
        CUDA_VERSION_INFO="nvcc ${_nvcc_release}"
    fi
fi

MPI_VERSION_INFO="$(mpirun --version 2>/dev/null | head -n1 || true)"
if [[ -z "$MPI_VERSION_INFO" ]]; then
    MPI_VERSION_INFO="${MPI_MODULE:-mpirun}"
fi

GPU_MODEL_INFO="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
    _gpu_models="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "$_gpu_models" ]]; then
        GPU_MODEL_INFO="$(printf '%s\n' "$_gpu_models" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sort -u | awk 'BEGIN{ORS=""} {if(NR>1) printf(", "); printf("%s", $0)}')"
        [[ -z "$GPU_MODEL_INFO" ]] && GPU_MODEL_INFO="unknown"
    fi
fi

echo "JobID: $SLURM_JOB_ID"
echo "Node(s): ${SLURM_JOB_NODELIST:-unknown}"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "Binary: $REAL_EXE"
if [[ "$FLUIDSIM_PROFILE_MODE" != "none" ]]; then
    echo "Profile mode: $FLUIDSIM_PROFILE_MODE"
    echo "Launcher: $EXE"
fi
echo "Params: $PARAMS"
echo "Runroot: $RUNROOT"
echo "Outdir: $OUTDIR"
echo "CUDA: $CUDA_VERSION_INFO"
echo "MPI: $MPI_VERSION_INFO"
echo "GPU_COUNT: $GPU_COUNT"
echo "GPU_MODEL: $GPU_MODEL_INFO"
echo "MPI_TRANSPORT_MODE: $MPI_TRANSPORT_MODE"
echo "TIMESTEPS: $TIMESTEPS"
echo "CHECKPOINT_EVERY: $CHECKPOINT_EVERY"
echo "VTKEVERY: $VTKEVERY"
echo "VTKINIT: $VTKINIT"
echo "SLURM_CPUS_PER_RANK: $SLURM_CPUS_PER_RANK"

cd "$RUNROOT"

# GPU run keeps host-side orchestration single-threaded.
OMP_NUM_THREADS=1
export OMP_NUM_THREADS

# Multi-rank transport mode policy + preflight.
configure_mpi_transport_mode
if (( SLURM_NTASKS > 1 )); then
    echo "Resolved OMPI_MCA_pml: ${OMPI_MCA_pml:-<unset>}"
    echo "Resolved OMPI_MCA_btl: ${OMPI_MCA_btl:-<unset>}"
    echo "Resolved OMPI_MCA_btl_vader_single_copy_mechanism: ${OMPI_MCA_btl_vader_single_copy_mechanism:-<unset>}"
    echo "Resolved OMPI_MCA_btl_smcuda_use_cuda_ipc: ${OMPI_MCA_btl_smcuda_use_cuda_ipc:-<unset>}"
    # Multi-rank runs require CUDA-aware MPI; app startup aborts if unavailable.
fi

RUN_STARTED=1

FLAGS=(
    --minimalLogs 1000
    --thermalLogs 100
    --initPerturb 0
    --vtkinit "$VTKINIT"
    --timesteps "$TIMESTEPS"
    --checkpointEvery "$CHECKPOINT_EVERY"
    --vtkevery "$VTKEVERY"
    --vtkmeshonly "$VTKMESHONLY"
)

# Keep a common mpirun argument shape across CPU/GPU scripts.
LAUNCH_TARGET="$EXE"
MPI_EXTRA_ARGS=()
MPIRUN_ARGS=(
    "${MPI_EXTRA_ARGS[@]}"
    -n "$SLURM_NTASKS"
)

if (( SLURM_NTASKS == 1 )); then
    # Some nodes expose a fragmented 2-CPU cpuset for single-rank jobs.
    # OpenMPI core binding can fail with "bound to CPUs in more than one package".
    # Single-rank GPU runs do not need explicit core binding.
    MPIRUN_ARGS+=(--bind-to none)
else
    MPIRUN_ARGS+=(--bind-to core --map-by "slot:PE=${SLURM_CPUS_PER_RANK}" --rank-by slot)
fi

mpirun "${MPIRUN_ARGS[@]}" "$LAUNCH_TARGET" "$PARAMS" "${FLAGS[@]}"
