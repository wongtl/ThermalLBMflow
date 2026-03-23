<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# ThermalLBMflow

Thermal lattice Boltzmann flow solver built on
[waLBerla](https://walberla.net/).

ThermalLBMflow solves coupled momentum and energy transport using the D3Q19
lattice Boltzmann method with TRT collision and Boussinesq buoyancy forcing.
It targets Rayleigh-Bénard and mixed-convection problems in complex 3-D
geometries defined by colored PLY surface meshes.

## Features

- D3Q19 TRT collision with Guo-style Boussinesq forcing
- Passive scalar (temperature) transport with configurable boundary conditions:
  Dirichlet, adiabatic, heat-flux, inlet/outlet, and pressure boundaries
- Color-coded PLY mesh workflow: paint boundary regions in a mesh editor,
  assign physics per color in the parameter file
- Checkpoint/restart with full field serialization
- VTK output (ParaView-compatible) for velocity, density, temperature, and
  Nusselt-number fields
- CPU build with OpenMP parallelism (outer or inner modes)
- GPU build with CUDA (one MPI rank per GPU, multi-GPU via MPI)
- Code-generated LBM kernels via [lbmpy](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/)
  and [pystencils](https://pycodegen.pages.i10git.cs.fau.de/pystencils/)

## Repository Layout

```
apps/
├── FluidSim_cpu/          CPU application, build scripts, Slurm launcher
├── FluidSim_gpu/          GPU application, build scripts, Slurm launcher, profiling wrappers
├── shared/
│   ├── geometry/          PLY mesh files
│   ├── params/            Parameter file (FluidSim.prm)
│   ├── helpers/           Shared C++ headers
│   ├── scripts/           Codegen venv installer, patch scripts
│   └── patches/           Upstream patches (if needed)
├── LICENSE                GPL-3.0-or-later
└── DEPENDENCIES.md        Third-party dependency summary
```

## Prerequisites

- C++17 compiler (GCC >= 10 or Clang >= 14)
- CMake >= 3.24
- MPI-3 implementation (e.g., OpenMPI >= 4.1)
- Python >= 3.10 with pip (for code generation)
- **GPU build only:** CUDA Toolkit >= 12.0

## Quick Start (CPU)

```bash
# 1. Set up the code-generation virtual environment once
./apps/shared/scripts/install_codegen_venv.sh

# 2. Run locally (builds FluidSim_cpu as needed, then launches a local smoke test)
cd apps/FluidSim_cpu
./run_sim_local.sh
```

For local CPU use, `run_sim_local.sh` is the intended entrypoint. It rebuilds
`FluidSim_cpu` when needed, runs locally, writes the launcher log to
`apps/FluidSim_cpu/output/logs/run_sim.log`, and writes simulation output under
`apps/FluidSim_cpu/output/`.

The repository ships a public example parameter file in
`shared/params/FluidSim.prm`. The local launcher currently uses the parameter
file configured in `run_sim_local.sh`.

For cluster/manual workflows, use the dedicated build and launcher scripts:

```bash
cd apps/FluidSim_cpu
./build_cpu.sh

sbatch --nodes=1 --ntasks=1 --cpus-per-task=48 run_sim_cpu.sbatch
```

See [FluidSim_cpu/README.md](FluidSim_cpu/README.md) for CPU-specific details.

## Quick Start (GPU)

```bash
# 1. Set up the code-generation virtual environment (if not already done)
./apps/shared/scripts/install_codegen_venv.sh

# 2. Build (cluster/manual path)
cd apps/FluidSim_gpu
CUDA_ARCHITECTURES=90 ./build_gpu.sh    # 90 = H100; use 89 for L40S, 70 for V100

# 3. Run (cluster/manual path)
sbatch --nodes=1 --gres=gpu:2 --ntasks-per-gpu=1 --cpus-per-gpu=2 run_sim_gpu.sbatch
```

`build_gpu.sh` and `run_sim_gpu.sbatch` are cluster/manual tools. They are not
needed for the simple local CPU workflow above.

See [FluidSim_gpu/README.md](FluidSim_gpu/README.md) for GPU-specific details,
including MPI transport modes, profiling, and cluster execution.

## Parameter File

All simulation physics and geometry are configured in
[shared/params/FluidSim.prm](shared/params/FluidSim.prm).
Key blocks:

| Block | Purpose |
|---|---|
| `MeshGeometry` | PLY mesh regions, roles, scaling, restart folder |
| `Physical` | Fluid properties, domain size, buoyancy parameters |
| `Resolution` | Lattice resolution and block decomposition |
| `Numerics` | Target lattice viscosity |
| `DomainSetup` | Periodicity |
| `Parameters` | Theta reference, update frequency |
| `ColorBC` | Per-color boundary condition definitions |

## Command-Line Flags

Flags are passed by the launcher scripts; the executable does not accept
bare positional arguments.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--timesteps` | uint | required | Number of LBM timesteps |
| `--minimalLogs` | uint | 0 | Minimal logger cadence (0 = off) |
| `--thermalLogs` | uint | 0 | Thermal logger cadence (0 = off) |
| `--initPerturb` | float | 0.0 | Initial temperature perturbation amplitude |
| `--checkpointEvery` | uint | 0 | Checkpoint write cadence (0 = off) |
| `--vtkevery` | uint | 0 | VTK write cadence (0 = off) |
| `--vtkinit` | bool | false | Write VTK at timestep 0 |
| `--vtkmeshonly` | bool | false | Write mesh VTK and exit (no simulation) |
| `--parallelMode` | string | outer | CPU only: `outer`, `inner`, or `serial` |

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
