<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# FluidSim_cpu

CPU build of ThermalLBMflow.

## Recommended Workflows

- Local users: use `build_local.sh` followed by `run_sim_local.sh`.
- Cluster/manual users: use `build_cpu.sh` and `run_sim_cpu.sbatch`.

## Build (Local)

```bash
./build_local.sh
```

`build_local.sh` is the intended local build path. It configures
`build-local` next to the `walberla` checkout, builds `FluidSim_cpu`, and
requires the shared codegen venv provisioned by
`apps/shared/scripts/install_codegen_venv.sh`.

Environment overrides:

| Variable | Purpose | Default |
|---|---|---|
| `BUILD_JOBS` | Local build parallelism | `1` |
| `RECONFIGURE` | Remove `build-local` before configuring | `0` |
| `VENV` | Codegen venv location to prefer | `<repo_root>/venv-walberla-codegen` |

## Build (Cluster / Manual)

```bash
./build_cpu.sh
```

`build_cpu.sh` environment overrides:

| Variable | Purpose | Default |
|---|---|---|
| `BUILD_USE_SRUN` | Use the cluster `srun` wrapper (`1`) or build directly in the current shell (`0`) | `1` |
| `BUILD_CLUSTER` | Slurm cluster for the build job when `BUILD_USE_SRUN=1` | `arc` |
| `BUILD_PARTITION` | Slurm partition when `BUILD_USE_SRUN=1` | `interactive` |
| `BUILD_TIME` | Slurm time limit when `BUILD_USE_SRUN=1` | `01:00:00` |
| `BUILD_MEM` | Slurm memory when `BUILD_USE_SRUN=1` | `16G` |
| `BUILD_CPUS_PER_TASK` | Build parallelism | `8` |
| `TOOLCHAIN_MODULE` | Compiler module to load | `foss/2024a` |
| `MPI_MODULE` | MPI module to load | (empty) |
| `PYTHON_MODULE` | Python module to load | `Python/3.12.3-GCCcore-13.3.0` |
| `CMAKE_MODULE` | CMake module to load | (empty) |
| `TARGET` | CMake build target | `FluidSim_cpu` |
| `RECONFIGURE` | Force CMake reconfigure | `0` |

For a normal local workflow you usually do not need this script. Use
`build_local.sh` instead.

## Run (Local)

```bash
./run_sim_local.sh
```

This is the intended local run entrypoint. It runs a single-rank,
single-timestep mesh-only job using `../shared/params/FluidSim.prm`. It is
useful for verifying the geometry setup before moving to longer runs or cluster
jobs.

This launcher is run-only. Build first with `./build_local.sh`.

Environment overrides: `NP`, `OMP_NUM_THREADS`, `OMP_PROC_BIND`, `OMP_PLACES`.

## Run (Slurm)

```bash
sbatch --nodes=1 --ntasks=1 --cpus-per-task=48 run_sim_cpu.sbatch
```

Launcher-owned knobs (set as environment variables before `sbatch`):

| Variable | Purpose | Default |
|---|---|---|
| `TIMESTEPS` | Number of LBM timesteps | `1` |
| `CHECKPOINT_EVERY` | Checkpoint write cadence | `0` (off) |
| `VTKEVERY` | VTK write cadence | `0` (off) |
| `TOOLCHAIN_MODULE` | Compiler module | (none) |
| `MPI_MODULE` | MPI module | (none) |
| `CPU_MODULE` | CPU-specific module | (none) |

The parameter file path is fixed inside `run_sim_cpu.sbatch`; inspect the
launcher script before submission if you want to change it.

## Parallel Modes

The `--parallelMode` flag controls how OpenMP parallelism is applied:

| Mode | Description |
|---|---|
| `outer` | Block-level parallelism: each block is processed by one thread. Best when number of blocks >> number of threads. |
| `inner` | Kernel-level parallelism: OpenMP directives inside generated sweeps. Best for single-block or few-block configurations. |
| `serial` | No OpenMP. Single-threaded execution. |

Default is `outer`. The launcher sets this via the `--parallelMode` flag.

## Output

- Local launcher log: `output/logs/run_sim.log`
- Local simulation output: `output/vtk/` and `output/checkpoint/`
- Cluster job log: `output/<jobid>/run_sim_cpu-<jobid>.log`
- Cluster Slurm logs: `output/<jobid>/slurm-<jobid>.out` and `output/<jobid>/slurm-<jobid>.err`
- Cluster simulation output: `output/<jobid>/vtk/` and `output/<jobid>/checkpoint/`
