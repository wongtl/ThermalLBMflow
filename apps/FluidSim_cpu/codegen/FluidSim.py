# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
"""FluidSim CPU code-generation entry point."""

import pystencils as ps
import sympy as sp
from lbmpy import (
    ForceModel,
    LBMConfig,
    LBStencil,
    Method,
    Stencil,
)
from lbmpy import relaxation_rate_from_lattice_viscosity
from lbmpy.boundaries import NoSlip
from pystencilssfg import SourceFileGenerator
from sweepgen import Sweep, get_build_config
from sweepgen.boundaries import GenericBoundary
from sweepgen.prefabs import LbmBulk


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def generate_stream_collide_sweep(sfg, lbm_bulk, cfg, name, *, sparse):
    sweep = Sweep(name, lbm_bulk.update_rule, cfg)
    sweep.sparse = sparse
    sweep.swap_fields(lbm_bulk.pdfs, lbm_bulk._pdfs_tmp)
    sfg.generate(sweep)


# -----------------------------------------------------------------------------
# Code generation
# -----------------------------------------------------------------------------

with SourceFileGenerator(keep_unknown_argv=True) as sfg:
    sfg.namespace("fluidsim::gen")

    # Build target and code-generation configs.
    build_cfg = get_build_config(sfg)
    build_cfg.target = ps.Target.CPU

    omp_cfg = build_cfg.get_pystencils_config()
    omp_cfg.cpu.openmp.enable = True

    serial_cfg = build_cfg.get_pystencils_config()
    serial_cfg.cpu.openmp.enable = False

    # Keep dtype selection aligned with serial_cfg used by common kernels.
    dtype = serial_cfg.default_dtype

    # Physical symbols and fields.
    stencil = LBStencil(Stencil.D3Q19)
    nu = sp.Symbol("nu")
    theta = ps.fields(f"theta: {dtype}[3D]", layout="fzyx")
    G = sp.Symbol("G")
    theta_ref = sp.Symbol("theta_ref")

    # LBM model.
    lbm_config = LBMConfig(
        stencil=stencil,
        method=Method.TRT,
        compressible=False,
        relaxation_rate=relaxation_rate_from_lattice_viscosity(nu),
        force_model=ForceModel.GUO,
        force=(
            sp.Float(0.0),
            G * (theta.center - theta_ref),
            sp.Float(0.0),
        ),
    )
    lbm_bulk = LbmBulk(sfg, "LBM", lbm_config)

    # LBM sweeps (CPU generates OpenMP and serial variants).
    with sfg.namespace("LBM"):
        generate_stream_collide_sweep(
            sfg,
            lbm_bulk,
            omp_cfg,
            "StreamCollideDenseOmp",
            sparse=False,
        )
        generate_stream_collide_sweep(
            sfg,
            lbm_bulk,
            serial_cfg,
            "StreamCollideDenseSerial",
            sparse=False,
        )
        generate_stream_collide_sweep(
            sfg,
            lbm_bulk,
            omp_cfg,
            "StreamCollideOmp",
            sparse=True,
        )
        generate_stream_collide_sweep(
            sfg,
            lbm_bulk,
            serial_cfg,
            "StreamCollideSerial",
            sparse=True,
        )
        sfg.generate(Sweep("InitPdfs", lbm_bulk._init_rule))

    # Macro-field initialization.
    rho, u = lbm_bulk.rho, lbm_bulk.u
    sfg.generate(
        Sweep(
            "InitializeMacroFields",
            [
                ps.Assignment(rho(), 1),
                ps.Assignment(u(0), 0),
                ps.Assignment(u(1), 0),
                ps.Assignment(u(2), 0),
            ],
        )
    )

    # Boundary kernels.
    # Keep NoSlip single-threaded. OpenMP in this tiny irregular boundary kernel
    # adds per-call team overhead and makes parallel mode behavior less deterministic.
    old_openmp_enabled = build_cfg.openmp_enabled
    build_cfg.openmp_enabled = False
    try:
        no_slip = GenericBoundary(
            NoSlip(name="NoSlip"),
            lbm_bulk.lb_method,
            lbm_bulk.pdfs,
        )
        sfg.generate(no_slip)
    finally:
        build_cfg.openmp_enabled = old_openmp_enabled
