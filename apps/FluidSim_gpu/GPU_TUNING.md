<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# FluidSim_gpu Runtime and Profiling Defaults

## Runtime profile (fixed in code)

`GpuReductions.cu` now uses a single built-in launch profile:

- `thetaFullTPB=256`
- `thetaMixedTPB=256`
- `minimalFullTPB=256`
- `minimalMixedTPB=256`
- `nuTPB=256`
- `gridCapX=65535`

All `FLUIDSIM_GPU_*` launch env tuning knobs were removed from the default path.
The launch config above is compile-time fixed and logged once as `profile=production_default`.

## Runtime policy (sbatch)

`run_sim_gpu.sbatch` provides the standard GPU batch-launch path:

- Launcher: `mpirun` (fixed)
- Host orchestration: serial (fixed)
- Host threading: `OMP_NUM_THREADS=1` (fixed)
- single-node submission is enforced
- one-rank-per-GPU policy is enforced (`SLURM_NTASKS == GPU_COUNT`)
  where `GPU_COUNT` comes from `SLURM_GPUS_ON_NODE` or `CUDA_VISIBLE_DEVICES`
- multi-rank jobs abort if GPU count is not detectable
- multi-rank transport default is `MPI_TRANSPORT_MODE=ob1_stable`:
  - `OMPI_MCA_pml=ob1`
  - `OMPI_MCA_btl=self,vader,tcp`
  - `OMPI_MCA_btl_vader_single_copy_mechanism=none`
- MPI rank binding policy: single-rank uses `--bind-to none`; multi-rank uses `--bind-to core --map-by slot:PE=<cpus-per-rank>`.

Supported runtime knobs:

- `CHECKPOINT_EVERY`
- `VTKEVERY`
- `TIMESTEPS`
- `MPI_TRANSPORT_MODE`
- `MPI_MODULE`
- `TOOLCHAIN_MODULE`
- `GPU_MODULE`

Path resolution is automatic for `build_gpu.sh` and `profile_loop_gpu.sh` (derived from script location).
`run_sim_gpu.sbatch` derives paths from `SLURM_SUBMIT_DIR`; submit from `ThermalLBMflow/apps/FluidSim_gpu`.

## High-impact runtime knobs

- `theta_update` (in `../shared/params/FluidSim.prm`): update period for global `theta_ref` reduction.
  Larger values reduce reduction + MPI allreduce overhead, but also change physical coupling cadence.
- `TIMESTEPS`: total simulation timesteps from launcher env (passed as `--timesteps`).
- `VTKEVERY`: VTK output cadence from launcher env; higher values reduce I/O overhead.
- `CHECKPOINT_EVERY`: checkpoint cadence from launcher env; higher values reduce checkpoint I/O overhead.

Parameter file is fixed to `$APP_DIR/../shared/params/FluidSim.prm` in `run_sim_gpu.sbatch` (not a supported override knob).
The launcher owns app runtime flags; positional app CLI args are rejected.
Path variables are launcher-managed and not supported override knobs in this launcher
(`APP_DIR`, `EXE`, `OUTDIR`).
VTK output base directory is fixed to `output`.

## Profiling wrappers (separate from the default run path)

Profiling is wrapper-driven: `profile_loop_gpu.sh` sets `FLUIDSIM_PROFILE_MODE` for `run_sim_gpu.sbatch`.
`profile_loop_gpu.sh` also passes through `MPI_TRANSPORT_MODE` to submitted jobs.
Use wrappers:

- `scripts/run_with_nsys.sh`
- `scripts/run_with_ncu.sh`
- `profile_loop_gpu.sh` (submits `nsys` then dependent `ncu`)
- Nsight reports are written directly to `apps/FluidSim_gpu/output/profiling/<jobid>/...`
  by the wrappers.

## MPI transport modes

`MPI_TRANSPORT_MODE` values for multi-rank runs:

- `ob1_stable` (default): pinned `ob1 + self,vader,tcp`
- `ob1_smcuda` (optional alternative): `pml=ob1`, force `btl=self,smcuda,tcp` (with `tcp` fallback)
  - fail-fast preflight if `btl:smcuda` is unavailable

No silent fallback is performed: if a requested mode is unavailable, launch aborts with an actionable message.

Fixed Nsight defaults in wrappers:

- Nsight Systems:
  - `--trace cuda,nvtx`
  - `--sample none`
  - `--stats false`
  - `--cuda-event-trace=false`
  - report base is fixed to `apps/FluidSim_gpu/output/profiling/<jobid>/nsys-report-r<rank>`
- Nsight Compute:
  - default section set is unset (more portable across different clusters)
  - optional override: set `NCU_SET=<set_name>` (for example `NCU_SET=basic` or `NCU_SET=detailed`)
  - profiles all kernels (no kernel-name filter)
  - `--launch-skip 0`
  - `--launch-count 50`
  - single-rank only (`MPI world size == 1`) is enforced in `run_with_ncu.sh`
  - report base is fixed to `apps/FluidSim_gpu/output/profiling/<jobid>/ncu-report-r<rank>`
