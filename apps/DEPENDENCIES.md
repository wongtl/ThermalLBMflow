<!-- SPDX-FileCopyrightText: 2026 David Wong, University of Oxford -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# ThermalLBMflow — Third-Party Dependencies

Versions below describe the current tested setup used by the FluidSim build and
code-generation scripts. The pycodegen packages are installed from upstream
development branches and/or an editable local checkout, so treat those versions
as tested snapshots rather than strict release pins.

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
| `walberla-sweepgen` (`sweepgen/`) | editable install from the current waLBerla checkout | GPL-3.0-or-later | waLBerla-integrated code-generation orchestration |
| [pystencils](https://pycodegen.pages.i10git.cs.fau.de/pystencils/) | `v2.0-dev` branch, tested: `2.0.dev0+170.ga13c5b1` | GPL-3.0-or-later | Stencil kernel generation |
| [lbmpy](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/) | upstream `master`, tested: `1.4+2.gfada994` | GPL-3.0-or-later | LBM collision operators and boundary conditions |
| `pystencilssfg` | tested: `0.1a4+59.gc635a76` | GPL-3.0-or-later | Source-file generation backend used by `sweepgen` |
| `numpy` | tested: `2.4.2` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Numerical dependency used by the pycodegen stack |
| `sympy` | tested: `1.14.0` | BSD | Symbolic algebra for code generation |
| `Jinja2` | tested: `3.1.6` | BSD-3-Clause | Template rendering used by codegen tooling |
| `py-cpuinfo` | `>= 9.0`, tested: `9.0.0` | MIT | Host capability helper required by `sweepgen` |

The open-source dependencies listed above are GPL-3.0-compatible.
Optional CUDA Toolkit use is governed separately by NVIDIA's EULA and redistribution terms.
