<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# FluidSim Shared Notes

This directory contains assets shared by `FluidSim_cpu` and `FluidSim_gpu`:

- `geometry/`
- `params/`
- `helpers/`
- `scripts/`
- `patches/`

## Intentional CPU vs GPU Differences

The items below are intentional architecture/runtime differences and should not be treated as accidental drift.

### Profiling mode (GPU only)

- `FluidSim_gpu` uses `FLUIDSIM_PROFILE_MODE` with values `none|nsys|ncu`.
- `run_sim_gpu.sbatch` selects wrappers:
  - `scripts/run_with_nsys.sh`
  - `scripts/run_with_ncu.sh`
- `FluidSim_cpu` has no equivalent profiling wrapper contract in the default launcher.

### CUDA architecture selection (GPU only)

- `FluidSim_gpu` build expects CUDA architecture configuration via `CUDA_ARCHITECTURES` / `CMAKE_CUDA_ARCHITECTURES`.
- This is required for correct/efficient CUDA code generation and is not applicable to `FluidSim_cpu`.

### Multi-rank GPU policy

- `FluidSim_gpu` enforces one-rank-per-GPU policy in launcher/runtime.
- For multi-rank GPU runs, CUDA-aware MPI support is required and enforced at startup.
- `FluidSim_cpu` does not have the one-rank-per-GPU concept.
