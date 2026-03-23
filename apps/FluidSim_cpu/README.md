<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# FluidSim_cpu

CPU build of ThermalLBMflow.

## Build

```bash
./build_cpu.sh
```

`build_cpu.sh` environment overrides:

| Variable | Purpose | Default |
|---|---|---|
| `BUILD_CLUSTER` | Slurm cluster for build job | (none — local build if unset) |
| `BUILD_PARTITION` | Slurm partition | (none) |
| `BUILD_TIME` | Slurm time limit | `02:00:00` |
| `BUILD_MEM` | Slurm memory | `8G` |
| `BUILD_CPUS_PER_TASK` | Build parallelism | `8` |
| `TOOLCHAIN_MODULE` | Compiler module to load | (none) |
| `MPI_MODULE` | MPI module to load | (none) |
| `PYTHON_MODULE` | Python module to load | (none) |
| `CMAKE_MODULE` | CMake module to load | (none) |
| `TARGET` | CMake build target | `FluidSim_cpu` |
| `RECONFIGURE` | Force CMake reconfigure | `0` |

## Run (Local)

```bash
./run_sim_local.sh
```

This runs a single-rank, single-timestep mesh-only job using
`../shared/params/FluidSim.prm`. Useful for verifying the geometry setup
before submitting production runs.

Environment overrides: `NP`, `OMP_NUM_THREADS`, `BUILD_JOBS`.

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

## Parallel Modes

The `--parallelMode` flag controls how OpenMP parallelism is applied:

| Mode | Description |
|---|---|
| `outer` | Block-level parallelism: each block is processed by one thread. Best when number of blocks >> number of threads. |
| `inner` | Kernel-level parallelism: OpenMP directives inside generated sweeps. Best for single-block or few-block configurations. |
| `serial` | No OpenMP. Single-threaded execution. |

Default is `outer`. The launcher sets this via the `--parallelMode` flag.

## Output

- VTK files: `output/<jobid>/output/vtk/`
- Checkpoints: `output/<jobid>/output/checkpoint/`
- Log file (local runs): `output/logs/run_sim.log`
