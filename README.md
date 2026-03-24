<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# ThermalLBMflow

ThermalLBMflow is a thermal lattice Boltzmann flow solver for buoyancy-driven
and mixed-convection simulations in complex 3-D geometries.

It is built on top of the in-tree
[waLBerla](https://walberla.net/) checkout contained in this repository and
provides CPU and GPU applications for Rayleigh-Benard and related thermal-flow
problems defined by colored PLY surface meshes.

## Start Here

- App overview and workflow split: [apps/README.md](apps/README.md)
- Canonical cold-build guide: [apps/BUILDING.md](apps/BUILDING.md)
- CPU app details: [apps/FluidSim_cpu/README.md](apps/FluidSim_cpu/README.md)
- GPU app details: [apps/FluidSim_gpu/README.md](apps/FluidSim_gpu/README.md)

## What This Repository Contains

This repository includes:

- the ThermalLBMflow application under `apps/`
- the waLBerla framework source used to build it
- code-generation tooling and shared helper scripts
- public example geometry and parameter files for local and cluster workflows

ThermalLBMflow builds directly from the waLBerla checkout already present in
this repository. No extra submodule step is required for the app workflow.
Relevant app-target prerequisites are enforced by CMake.

## Quick Start

```bash
# 1. Set up the code-generation environment once
./apps/shared/scripts/install_codegen_venv.sh

# 2. Build the local CPU app
cd apps/FluidSim_cpu
./build_local.sh

# 3. Run the local example
./run_sim_local.sh
```

For cluster or manual workflows, use the dedicated CPU/GPU build scripts and
`sbatch` launchers documented in [apps/BUILDING.md](apps/BUILDING.md).

## Repository Layout

```text
apps/
  BUILDING.md            Public build and run guide
  README.md              App-level overview
  FluidSim_cpu/          CPU application and launchers
  FluidSim_gpu/          GPU application and launchers
  shared/                Parameter files, geometry, helpers, scripts
src/                     waLBerla core framework sources
python/                  waLBerla Python components
```

## Upstream waLBerla Context

ThermalLBMflow builds on waLBerla, a high-performance block-structured
multiphysics framework. This repository contains an application-focused fork
that keeps the upstream waLBerla source tree in place so the app can build from
one checkout.

If you are looking for the general-purpose upstream framework, documentation, or
its broader feature set, see:

- waLBerla project site: <https://walberla.net/>
- waLBerla documentation: <https://walberla.net/doxygen/index.html>
- upstream source project: <https://i10git.cs.fau.de/walberla/walberla>

## Contributing

Contributions to this repository go through GitHub. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the project-specific contribution
workflow.

## License

ThermalLBMflow is distributed under GPL-3.0-or-later. See [COPYING.txt](COPYING.txt)
and [apps/LICENSE](apps/LICENSE).
