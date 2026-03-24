<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# FluidSim_gpu

GPU build/runtime path for FluidSim. In the current workflow this is the
cluster/manual path; local users typically only need
`apps/FluidSim_cpu/run_sim_local.sh`.

## Scope

- GPU build/runtime only.
- Single-level runtime only (`AMR.refinementLevels = 0`, so `levels == 1`).
- Single-node submission model in `run_sim_gpu.sbatch`.
- Cross-app intentional differences are documented in `../shared/README.md`.

## Runtime Contract

- MPI launcher is fixed to `mpirun`.
- Host orchestration is fixed to serial (single-threaded host control path).
- `OMP_NUM_THREADS=1`.
- Single-node is enforced in `run_sim_gpu.sbatch` (`--nodes=1`).
- One-rank-per-GPU policy is enforced in `run_sim_gpu.sbatch`
  (`SLURM_NTASKS == GPU_COUNT`, where `GPU_COUNT` is derived from `SLURM_GPUS_ON_NODE` or `CUDA_VISIBLE_DEVICES`).
  For multi-rank jobs, if GPU count is not detectable, the launcher aborts.
- Multi-rank OpenMPI transport is pinned for stability:
  - `OMPI_MCA_pml=ob1`
  - `OMPI_MCA_btl=self,vader,tcp`
  - `OMPI_MCA_btl_vader_single_copy_mechanism=none`
  via `MPI_TRANSPORT_MODE=ob1_stable` (default).
- Runtime enforces explicit local-rank GPU device selection.
- MPI rank binding policy: single-rank uses `--bind-to none`; multi-rank uses `--bind-to core --map-by slot:PE=<cpus-per-rank>`.

## Supported Knobs (GPU sbatch launcher)

- `CHECKPOINT_EVERY`
- `VTKEVERY`
- `TIMESTEPS`
- `MPI_TRANSPORT_MODE`
- `MPI_MODULE`
- `TOOLCHAIN_MODULE`
- `GPU_MODULE`

Parameter file is fixed to `$APP_DIR/../shared/params/FluidSim.prm` in
`run_sim_gpu.sbatch` (not a supported override knob).
Simulation length is launcher-owned via `TIMESTEPS` (passed as `--timesteps`).
This launcher owns app flags; positional app CLI args are rejected.
Path variables are launcher-managed and not supported override knobs in this launcher
(`APP_DIR`, `EXE`, `OUTDIR`).
VTK output base directory is fixed to `output`.

## Not In Default Path

- Profiling is wrapper-driven: `profile_loop_gpu.sh` sets `FLUIDSIM_PROFILE_MODE` for `run_sim_gpu.sbatch`.
- Use profiling wrappers:
  - `profile_loop_gpu.sh`
  - `scripts/run_with_nsys.sh`
  - `scripts/run_with_ncu.sh`
- Nsight reports are written directly to `apps/FluidSim_gpu/output/profiling/<jobid>/...`
  by the wrappers.

## MPI Transport Modes

`run_sim_gpu.sbatch` supports a launcher-owned transport selector:

- `MPI_TRANSPORT_MODE=ob1_stable` (default):
  - `OMPI_MCA_pml=ob1`
  - `OMPI_MCA_btl=self,vader,tcp`
  - `OMPI_MCA_btl_vader_single_copy_mechanism=none`
- `MPI_TRANSPORT_MODE=ob1_smcuda` (optional alternative):
  - `OMPI_MCA_pml=ob1`
  - `OMPI_MCA_btl=self,smcuda,tcp`
  - preflight requires `btl:smcuda` to exist (`ompi_info` check)

Mode policy is applied only for multi-rank runs (`SLURM_NTASKS > 1`).
Preflight is fail-fast: requested unavailable components abort launch (no silent fallback).
Resolved transport mode and MCA values are printed into the job log header.

## Build + Run (Cluster / Manual)

Build on your cluster or prepared HPC shell:

```bash
./build_gpu.sh
```

Supported `build_gpu.sh` environment overrides:
- Build resources: `BUILD_CLUSTER`, `BUILD_PARTITION`, `BUILD_TIME`, `BUILD_MEM`, `BUILD_CPUS_PER_TASK`
- Build behavior: `TARGET`, `RECONFIGURE`
- `BUILD_USE_SRUN=1` uses the cluster `srun` wrapper (default); `BUILD_USE_SRUN=0`
  builds directly in the current shell if you have already prepared the environment
- Tool/modules: `TOOLCHAIN_MODULE`, `PYTHON_MODULE`, `MPI_MODULE`, `GPU_MODULE`, `CMAKE_MODULE`
- CUDA arch selection: `CUDA_ARCHITECTURES`
  - Choose archs matching your GPU(s): V100=`70`, L40S=`89`, H100=`90`
  - Example: `CUDA_ARCHITECTURES=89 ./build_gpu.sh`

Run a batch job:

```bash
cd walberla/apps/FluidSim_gpu
sbatch --nodes=1 --gres=gpu:2 --ntasks-per-gpu=1 --cpus-per-gpu=2 run_sim_gpu.sbatch

# Example with optional site-specific routing flags:
# sbatch --clusters=<cluster> --partition=<partition> \
#   --nodes=1 --gres=gpu:2 --ntasks-per-gpu=1 --cpus-per-gpu=2 \
#   run_sim_gpu.sbatch
```

Path resolution is automatic for `build_gpu.sh` and `profile_loop_gpu.sh`
(derived from script location).
`run_sim_gpu.sbatch` derives paths from `SLURM_SUBMIT_DIR`; submit from `walberla/apps/FluidSim_gpu`.

For the simple local/public workflow, use the CPU local launcher instead:

```bash
cd apps/FluidSim_cpu
./run_sim_local.sh
```

## Restart / Checkpoints

- Checkpoints are written under `output/checkpoint/` with prefix `fluidsim_state_*` (inside the job output tree).
- Restart archive root is fixed to `<repo_root>/walberla/apps/storage`.
- To keep a checkpoint set for later restart, copy it to `<repo_root>/walberla/apps/storage/<case>/checkpoint/`.
- The expected files are `<repo_root>/walberla/apps/storage/<case>/checkpoint/fluidsim_state_*`.
- In `../shared/params/FluidSim.prm`, set `MeshGeometry.checkpointFolder "<case>"`.
- Restart then resolves to `<repo_root>/walberla/apps/storage/<case>/checkpoint/fluidsim_state_*`.
- If a PLY needs OpenMesh-compatible face-property ordering, the solver may create a cached `geometry_compat/*_openmesh_compat.ply`; this cache is safe to delete.

## Notes

- Slurm stdout/stderr are written by Slurm to `%x-%j.out` and `%x-%j.err` (submit directory by default), and the launcher moves them to `OUTDIR/slurm-<jobid>.out` and `OUTDIR/slurm-<jobid>.err` at job exit.
- The app tee log is written to `OUTDIR/run_sim_gpu-<jobid>.log` (default `OUTDIR`: `apps/FluidSim_gpu/output/<jobid>`), and simulation files are written under `OUTDIR/output`.
- Keep job artifacts under `apps/FluidSim_gpu/output/` or external paths.
- For multi-rank runs (`SLURM_NTASKS > 1` / MPI world size > 1), startup aborts if CUDA-aware MPI is unavailable.
- If your cluster requires a specific GPU SKU, override `--gres` (for example `--gres=gpu:h100:2`).
- `MINIMAL` logger field `E` is the domain integral of `theta` (not a generic total energy diagnostic).
- `MINIMAL` logger field `dE` is the change in that theta integral since the previous minimal-log tick.
- `MINIMAL` logger fields `M`/`dM` are mass integral and its delta; `rhoMean` is `M / fluidVolume`.
