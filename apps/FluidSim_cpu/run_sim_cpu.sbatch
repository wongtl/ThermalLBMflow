#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
#SBATCH --job-name=fs_cpu
## Example submit (set partition/account/options for your site):
## sbatch --partition=<partition> run_sim_cpu.sbatch
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Submit context checks.

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run_sim_cpu.sbatch must be submitted with sbatch." >&2
    exit 1
fi

# Slurm executes a spool copy of this script; derive paths from submit directory.
# Contract: submit from walberla/apps/FluidSim_cpu.
if ! APP_DIR="$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && pwd)"; then
    echo "ERROR: Could not resolve submit directory from SLURM_SUBMIT_DIR/PWD." >&2
    exit 1
fi
if [[ ! -f "$APP_DIR/run_sim_cpu.sbatch" ]]; then
    echo "ERROR: submit from walberla/apps/FluidSim_cpu (current submit dir: $APP_DIR)." >&2
    exit 1
fi

# Path and runtime configuration.

# App paths.
REAL_EXE="$APP_DIR/../../../build-cpu/apps/FluidSim_cpu/FluidSim_cpu"
SHARED_DIR="$APP_DIR/../shared"
PARAMS="$SHARED_DIR/params/FluidSim.prm"

# Module/loading knobs.
TOOLCHAIN_MODULE="${TOOLCHAIN_MODULE:-}"
CPU_MODULE="${CPU_MODULE:-}"
MPI_MODULE="${MPI_MODULE:-}"

# Runtime knobs (launcher-owned, environment-overridable).
PARALLEL_MODE="${PARALLEL_MODE:-inner}"
TIMESTEPS="${TIMESTEPS:-1001}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
VTKEVERY="${VTKEVERY:-1000}"
VTKINIT="${VTKINIT:-0}"
VTKMESHONLY="${VTKMESHONLY:-false}"

# Paths/state.
SCRATCH_BASE="${SCRATCH:-${TMPDIR:-/tmp/$USER}}"
RUNROOT="${SCRATCH_BASE}/fluidsim_cpu_${SLURM_JOB_ID}"
RUN_OUTPUT_DIR="${RUNROOT}/output"
RUN_STARTED=0

OUTDIR="$APP_DIR/output/${SLURM_JOB_ID}"
SLURM_LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SLURM_LOG_BASENAME="${SLURM_JOB_NAME:-fs_cpu}-${SLURM_JOB_ID}"
SLURM_STDOUT_PATH="${SLURM_LOG_DIR}/${SLURM_LOG_BASENAME}.out"
SLURM_STDERR_PATH="${SLURM_LOG_DIR}/${SLURM_LOG_BASENAME}.err"

# Naming policy: direct Slurm outputs and directly processed Slurm-derived values
# keep the `SLURM_` prefix (for example SLURM_NODE_COUNT, SLURM_CPUS_PER_RANK).
SLURM_NODE_COUNT="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}"

# Slurm resource preflight.
if ! [[ "${SLURM_NTASKS:-}" =~ ^[0-9]+$ ]] || (( SLURM_NTASKS < 1 )); then
    echo "ERROR: Invalid SLURM_NTASKS='${SLURM_NTASKS:-unset}'." >&2
    echo "Hint: submit with explicit task mapping (for example --ntasks=1 --cpus-per-task=<N>)." >&2
    exit 1
fi

if ! [[ "$SLURM_NODE_COUNT" =~ ^[0-9]+$ ]] || (( SLURM_NODE_COUNT < 1 )); then
    echo "ERROR: Could not determine a valid node count from Slurm (SLURM_JOB_NUM_NODES='${SLURM_JOB_NUM_NODES:-unset}', SLURM_NNODES='${SLURM_NNODES:-unset}')." >&2
    exit 1
fi
if (( SLURM_NODE_COUNT != 1 )); then
    echo "ERROR: run_sim_cpu.sbatch supports single-node runs only (got nodes=${SLURM_NODE_COUNT})." >&2
    echo "Hint: submit with --nodes=1 and keep all tasks on that node." >&2
    exit 1
fi

SLURM_CPUS_PER_RANK=0
if [[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[0-9]+$ ]] && (( SLURM_CPUS_PER_TASK >= 1 )); then
    SLURM_CPUS_PER_RANK="${SLURM_CPUS_PER_TASK}"
else
    echo "ERROR: Could not determine a numeric per-rank CPU count from Slurm." >&2
    echo "       SLURM_CPUS_PER_TASK='${SLURM_CPUS_PER_TASK:-unset}'." >&2
    echo "Hint: submit with --cpus-per-task=<N>." >&2
    exit 1
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
case "$VTKMESHONLY" in
    0|1|true|false|TRUE|FALSE|True|False)
        ;;
    *)
        echo "ERROR: VTKMESHONLY must be a boolean value (0/1/false/true) (got '$VTKMESHONLY')." >&2
        exit 1
        ;;
esac
if ! [[ "$TIMESTEPS" =~ ^[0-9]+$ ]] || (( TIMESTEPS < 1 )); then
    echo "ERROR: TIMESTEPS must be a positive integer (got '$TIMESTEPS')." >&2
    exit 1
fi

# This launcher owns all app runtime flags.
if (( $# > 0 )); then
    echo "ERROR: run_sim_cpu.sbatch does not accept extra app arguments." >&2
    echo "Use environment knobs (for example PARALLEL_MODE, TIMESTEPS, CHECKPOINT_EVERY, VTKEVERY, VTKINIT)." >&2
    exit 1
fi

# Input and executable validation.
if [[ ! -f "$PARAMS" ]]; then
    echo "ERROR: Parameter file not found: $PARAMS" >&2
    exit 1
fi
if [[ ! -x "$REAL_EXE" ]]; then
    echo "ERROR: FluidSim_cpu executable not found: $REAL_EXE" >&2
    echo "Build FluidSim_cpu first with build_cpu.sh." >&2
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
    [[ -n "$CPU_MODULE" ]] && module load "$CPU_MODULE"
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

exec > >(tee -a "$OUTDIR/run_sim_cpu-${SLURM_JOB_ID}.log") 2>&1

MPI_VERSION_INFO="$(mpirun --version 2>/dev/null | head -n1 || true)"
if [[ -z "$MPI_VERSION_INFO" ]]; then
    MPI_VERSION_INFO="${MPI_MODULE:-mpirun}"
fi

CPU_COUNT_INFO="${SLURM_CPUS_ON_NODE:-unknown}"
CPU_MODEL_INFO="unknown"
if command -v lscpu >/dev/null 2>&1; then
    _cpu_model="$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -n1 || true)"
    if [[ -n "$_cpu_model" ]]; then
        CPU_MODEL_INFO="$_cpu_model"
    fi
fi

echo "JobID: $SLURM_JOB_ID"
echo "Node(s): ${SLURM_JOB_NODELIST:-unknown}"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "Binary: $REAL_EXE"
echo "Params: $PARAMS"
echo "Runroot: $RUNROOT"
echo "Outdir: $OUTDIR"
echo "MPI: $MPI_VERSION_INFO"
echo "CPU_COUNT: $CPU_COUNT_INFO"
echo "CPU_MODEL: $CPU_MODEL_INFO"
echo "PARALLEL_MODE: $PARALLEL_MODE"
echo "SLURM_CPUS_PER_RANK: $SLURM_CPUS_PER_RANK"

cd "$RUNROOT"

# CPU run uses OpenMP thread placement.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_RANK"
export OMP_PROC_BIND
export OMP_PLACES

RUN_STARTED=1

FLAGS=(
    --parallelMode "$PARALLEL_MODE"
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
LAUNCH_TARGET="$REAL_EXE"
MPI_EXTRA_ARGS=(--mca ras "^slurm" --mca plm "^slurm")
MPIRUN_ARGS=(
    "${MPI_EXTRA_ARGS[@]}"
    -n "$SLURM_NTASKS"
)

if (( SLURM_NTASKS == 1 )); then
    # Single-rank CPU runs do not need explicit core binding.
    MPIRUN_ARGS+=(--bind-to none)
else
    MPIRUN_ARGS+=(--bind-to core --map-by "slot:PE=${SLURM_CPUS_PER_RANK}" --rank-by slot)
fi

mpirun "${MPIRUN_ARGS[@]}" "$LAUNCH_TARGET" "$PARAMS" "${FLAGS[@]}"
