<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Building ThermalLBMflow

This document captures the canonical cold-build workflow for the public app
tree. FluidSim builds directly from the in-tree waLBerla checkout in this
repository: no extra submodule step is required for the app workflow, and
app-target prerequisites such as OpenMesh/code-generation support are enforced
by CMake. Start from a fresh clone, set up the code-generation environment
once, then choose the build/run path that matches where you want to execute.

## 1. Clone the repository

```bash
git clone <repo-url>
cd ThermalLBMflow
```

## 2. Install the code-generation virtual environment

```bash
./apps/shared/scripts/install_codegen_venv.sh
```

This creates the shared `venv-walberla-codegen` environment used by the app
builds and code generation.

## 3. Build

Choose one build path:

### Local CPU build

```bash
cd apps/FluidSim_cpu
./build_local.sh
```

### Cluster / manual CPU build

```bash
cd apps/FluidSim_cpu
./build_cpu.sh
```

### Cluster / manual GPU build

```bash
cd apps/FluidSim_gpu
CUDA_ARCHITECTURES=90 ./build_gpu.sh
```

Use a `CUDA_ARCHITECTURES` value matching your GPU:
- `70` = V100
- `89` = L40S
- `90` = H100

## 4. Run

Choose the matching launcher:

### Local CPU run

```bash
cd apps/FluidSim_cpu
./run_sim_local.sh
```

### Cluster CPU run

```bash
cd apps/FluidSim_cpu
sbatch --nodes=1 --ntasks=1 --cpus-per-task=48 run_sim_cpu.sbatch
```

### Cluster GPU run

```bash
cd apps/FluidSim_gpu
sbatch --nodes=1 --gres=gpu:2 --ntasks-per-gpu=1 --cpus-per-gpu=2 run_sim_gpu.sbatch
```

## Notes

- `run_sim_local.sh` is run-only. Build first with `build_local.sh`.
- `build_cpu.sh`, `build_gpu.sh`, and the `sbatch` launchers are the
  cluster/manual workflow.
- `run_sim_local.sh` uses `apps/shared/params/FluidSim.prm`.
- `run_sim_cpu.sbatch` and `run_sim_gpu.sbatch` use
  `apps/shared/params/FluidSim.prm`.
