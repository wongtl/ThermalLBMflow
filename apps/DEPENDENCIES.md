<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# ThermalLBMflow — Third-Party Dependencies

Versions below describe the exact tested setup provisioned by
`apps/shared/scripts/install_codegen_venv.sh` for the FluidSim build and
code-generation scripts.

## Core runtime/build dependencies

| Dependency | Version policy | License | Usage |
|---|---|---|---|
| [waLBerla](https://walberla.net/) | current waLBerla checkout (tested with the FluidSim app sources in this repository) | GPL-3.0-or-later | Core framework: block-structured grids, MPI communication, field storage, VTK I/O |
| [OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/) | fetched by CMake | BSD-3-Clause | PLY geometry loading (waLBerla dependency) |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | GPU build only, tested with `>= 12.0` | NVIDIA EULA | Device kernels, runtime support, cuFFT (optional) |
| [OpenMPI](https://www.open-mpi.org/) | any MPI-3 implementation; ARC/HTC runs currently use OpenMPI `>= 4.1` | BSD-3-Clause | MPI communication |

## Python build/codegen dependencies

| Dependency | Version policy | License | Usage |
|---|---|---|---|
| `walberla-sweepgen` (`sweepgen/`) | editable install from the current waLBerla checkout, tested: `7.2+61.gfa0eaab29` | GPL-3.0-or-later | waLBerla-integrated code-generation orchestration |
| [pystencils](https://pycodegen.pages.i10git.cs.fau.de/pystencils/) | pinned public Git commit `a13c5b1c71c924b0d58c1a343ac9780015786edd`, version `2.0.dev0+170.ga13c5b1` | GPL-3.0-or-later | Stencil kernel generation |
| [lbmpy](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/) | pinned public Git commit `fada99477fbd5de9d0f583402269595d77a2b40b`, version `1.4+2.gfada994` | GPL-3.0-or-later | LBM collision operators and boundary conditions |
| `pystencilssfg` | pinned public Git commit `c635a7666bb97fe6c15490011fa03d48a93e2879`, version `0.1a4+59.gc635a76` | GPL-3.0-or-later | Source-file generation backend used by `sweepgen` |
| `numpy` | pinned: `2.4.2` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Numerical dependency used by the pycodegen stack |
| `sympy` | pinned: `1.14.0` | BSD | Symbolic algebra for code generation |
| `Jinja2` | pinned: `3.1.6` | BSD-3-Clause | Template rendering used by codegen tooling |
| `py-cpuinfo` | pinned: `9.0.0` | MIT | Host capability helper required by `sweepgen` |

The canonical app-owned installer path is:

- `pystencils` from pinned public Git commit `a13c5b1c71c924b0d58c1a343ac9780015786edd`
- `lbmpy` from pinned public Git commit `fada99477fbd5de9d0f583402269595d77a2b40b`
- `pystencilssfg` from pinned public Git commit `c635a7666bb97fe6c15490011fa03d48a93e2879`
- pinned `numpy`, `sympy`, `Jinja2`, and `py-cpuinfo` from PyPI
- editable in-repo `sweepgen`

The open-source dependencies listed above are GPL-3.0-compatible.
Optional CUDA Toolkit use is governed separately by NVIDIA's EULA and redistribution terms.
