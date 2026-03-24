<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# ThermalLBMflow

Thermal lattice Boltzmann flow solver built on
[waLBerla](https://walberla.net/).

ThermalLBMflow solves coupled momentum and energy transport using the D3Q19
lattice Boltzmann method with TRT collision and Boussinesq buoyancy forcing.
It targets Rayleigh-Bénard and mixed-convection problems in complex 3-D
geometries defined by colored PLY surface meshes.

See [BUILDING.md](BUILDING.md) for the canonical cold-build workflow. The app
builds directly from the in-tree waLBerla checkout in this repository, no extra
submodule step is required for the public app workflow, and relevant app-target
prerequisites are enforced by CMake.

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
├── BUILDING.md            Canonical cold-build guide
├── FluidSim_cpu/          CPU application, local + cluster build scripts, launchers
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

# 2. Build locally
cd apps/FluidSim_cpu
./build_local.sh

# 3. Run locally
./run_sim_local.sh
```

For local CPU use, `build_local.sh` and `run_sim_local.sh` are the intended
pair. The launcher writes the log to
`apps/FluidSim_cpu/output/logs/run_sim.log`, and writes simulation output under
`apps/FluidSim_cpu/output/`.

The repository ships a public example parameter file in
`shared/params/FluidSim.prm`, and `run_sim_local.sh` now uses that public
example by default.

For cluster/manual workflows, use the dedicated build and launcher scripts:

```bash
cd apps/FluidSim_cpu
./build_cpu.sh

sbatch --nodes=1 --ntasks=1 --cpus-per-task=48 run_sim_cpu.sbatch
```

See [FluidSim_cpu/README.md](FluidSim_cpu/README.md) for CPU-specific details.
The cluster/manual `sbatch` launchers use `shared/params/FluidSim.prm`.

## Cluster/Manual (CPU/GPU)

- CPU: see [FluidSim_cpu/README.md](FluidSim_cpu/README.md) for `build_cpu.sh`,
  `run_sim_cpu.sbatch`, parallel modes, and CPU output layout.
- GPU: see [FluidSim_gpu/README.md](FluidSim_gpu/README.md) for `build_gpu.sh`,
  `run_sim_gpu.sbatch`, MPI transport modes, profiling, and GPU execution.

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
