# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
"""FluidSim CPU theta-update code-generation entry point."""

import pystencils as ps
import sympy as sp
from lbmpy import LBStencil, Stencil
from pystencilssfg import SourceFileGenerator
from sweepgen import Sweep, get_build_config


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def generate_sweep(sfg, name, assignments, cfg, *, sparse=False):
    sweep = Sweep(name, assignments, cfg)
    sweep.sparse = sparse
    sfg.generate(sweep)


# -----------------------------------------------------------------------------
# Code generation
# -----------------------------------------------------------------------------

with SourceFileGenerator(keep_unknown_argv=True) as sfg:
    sfg.namespace("fluidsim::gen")

    # Build and backend configuration.
    build_cfg = get_build_config(sfg)
    build_cfg.target = ps.Target.CPU

    omp_cfg = build_cfg.get_pystencils_config()
    omp_cfg.cpu.openmp.enable = True

    serial_cfg = build_cfg.get_pystencils_config()
    serial_cfg.cpu.openmp.enable = False

    dtype = serial_cfg.default_dtype

    # Stencil metadata and boundary constants.
    stencil = LBStencil(Stencil.D3Q19)
    stencil_dirs = [tuple(int(v) for v in d) for d in stencil.stencil_entries]
    dir_to_idx = {d: i for i, d in enumerate(stencil_dirs)}

    BC_INLET = sp.Integer(4)
    BC_OUTLET = sp.Integer(5)
    BC_PRESSURE = sp.Integer(6)

    THERMAL_DIRICHLET = sp.Integer(1)
    THERMAL_ADIABATIC = sp.Integer(2)
    THERMAL_HEATLOAD = sp.Integer(3)

    CELL_FLUID = sp.Integer(0)
    CELL_SOLID = sp.Integer(1)

    FACE_DIRS = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]

    # Distribution-function helpers.
    def pdf_eq(cx, cy, cz, rho_val, ux_val, uy_val, uz_val):
        drho = rho_val - sp.Integer(1)
        u2 = ux_val * ux_val + uy_val * uy_val + uz_val * uz_val
        abs_sum = abs(cx) + abs(cy) + abs(cz)
        cu = sp.Integer(cx) * ux_val + sp.Integer(cy) * uy_val + sp.Integer(cz) * uz_val

        if abs_sum == 0:
            return sp.Rational(1, 3) * drho - sp.Rational(1, 3) * u2

        if abs_sum == 1:
            return (
                sp.Rational(1, 18) * drho
                + sp.Rational(1, 6) * cu
                + sp.Rational(1, 3) * cu * cu
                - sp.Rational(1, 6) * u2
            )

        if abs_sum == 2:
            active_sum_sq = (
                sp.Integer(cx * cx) * ux_val * ux_val
                + sp.Integer(cy * cy) * uy_val * uy_val
                + sp.Integer(cz * cz) * uz_val * uz_val
            )
            signed_cross = (
                sp.Integer(cx * cy) * ux_val * uy_val
                + sp.Integer(cx * cz) * ux_val * uz_val
                + sp.Integer(cy * cz) * uy_val * uz_val
            )
            return (
                sp.Rational(1, 36) * drho
                + sp.Rational(1, 12) * cu
                + sp.Rational(1, 12) * active_sum_sq
                + sp.Rational(1, 4) * signed_cross
            )

        raise ValueError("Invalid D3Q19 direction.")

    # Fields and symbols.
    theta, theta_tmp = ps.fields(f"theta, theta_tmp: {dtype}[3D]", layout="fzyx")
    u = ps.fields(f"u(3): {dtype}[3D]", layout="fzyx")
    rho = ps.fields(f"rho: {dtype}[3D]", layout="fzyx")
    pdf = ps.fields(f"pdf({stencil.Q}): {dtype}[3D]", layout="fzyx")
    flow_rho = ps.fields(f"flow_rho: {dtype}[3D]", layout="fzyx")
    flow_theta = ps.fields(f"flow_theta: {dtype}[3D]", layout="fzyx")
    flow_velocity = ps.fields(f"flow_velocity(3): {dtype}[3D]", layout="fzyx")
    thermal_value = ps.fields(f"thermal_value: {dtype}[3D]", layout="fzyx")

    bc_id = ps.fields("bc_id: uint16[3D]", layout="fzyx")
    thermal_type = ps.fields("thermal_type: uint8[3D]", layout="fzyx")
    cell_type = ps.fields("cell_type: uint8[3D]", layout="fzyx")

    alpha = sp.Symbol("alpha")
    a_lat = sp.Symbol("a_lat")
    theta_ref = sp.Symbol("theta_ref")
    heatload_scale = sp.Symbol("heatload_scale")

    # Theta advection-diffusion discretization.
    ux = u.center_vector[0]
    uy = u.center_vector[1]
    uz = u.center_vector[2]

    uxp = (ux + sp.Abs(ux)) / 2
    uxm = (ux - sp.Abs(ux)) / 2
    uyp = (uy + sp.Abs(uy)) / 2
    uym = (uy - sp.Abs(uy)) / 2
    uzp = (uz + sp.Abs(uz)) / 2
    uzm = (uz - sp.Abs(uz)) / 2

    adv_x = uxp * (theta.center - theta[-1, 0, 0]) + uxm * (theta[1, 0, 0] - theta.center)
    adv_y = uyp * (theta.center - theta[0, -1, 0]) + uym * (theta[0, 1, 0] - theta.center)
    adv_z = uzp * (theta.center - theta[0, 0, -1]) + uzm * (theta[0, 0, 1] - theta.center)

    lap = (
        theta[1, 0, 0]
        + theta[-1, 0, 0]
        + theta[0, 1, 0]
        + theta[0, -1, 0]
        + theta[0, 0, 1]
        + theta[0, 0, -1]
        - sp.Integer(6) * theta.center
    )

    adv = adv_x + adv_y + adv_z
    theta_next = theta.center + (-adv + alpha * lap)
    theta_assignment = [ps.Assignment(theta_tmp.center, theta_next)]

    # Theta update sweeps (CPU generates OpenMP and serial variants).
    generate_sweep(
        sfg,
        "ThetaUpdateDenseOmp",
        theta_assignment,
        omp_cfg,
        sparse=False,
    )
    generate_sweep(
        sfg,
        "ThetaUpdateDenseSerial",
        theta_assignment,
        serial_cfg,
        sparse=False,
    )
    generate_sweep(
        sfg,
        "ThetaUpdateOmp",
        theta_assignment,
        omp_cfg,
        sparse=True,
    )
    generate_sweep(
        sfg,
        "ThetaUpdateSerial",
        theta_assignment,
        serial_cfg,
        sparse=True,
    )

    # Open-boundary reconstruction sweeps.
    def make_open_boundary_reconstruct_assignments(target_bc, mode):
        rho_target_sym = sp.Symbol("rho_target")
        ux_target_sym = sp.Symbol("ux_target")
        uy_target_sym = sp.Symbol("uy_target")
        uz_target_sym = sp.Symbol("uz_target")

        if mode == "velocity":
            ux_target_expr = flow_velocity.center_vector[0]
            uy_target_no_force_expr = flow_velocity.center_vector[1]
            uz_target_expr = flow_velocity.center_vector[2]
        else:
            ux_raw = u.center_vector[0]
            uy_raw = u.center_vector[1]
            uz_raw = u.center_vector[2]
            nx = flow_velocity.center_vector[0]
            ny = flow_velocity.center_vector[1]
            nz = flow_velocity.center_vector[2]
            dot_un = ux_raw * nx + uy_raw * ny + uz_raw * nz
            norm_sq = nx * nx + ny * ny + nz * nz + sp.Float(1.0e-30)

            if mode == "pressure_in":
                corr = sp.Piecewise((dot_un / norm_sq, dot_un > sp.Integer(0)), (sp.Integer(0), True))
            elif mode == "pressure_out":
                corr = sp.Piecewise((dot_un / norm_sq, dot_un < sp.Integer(0)), (sp.Integer(0), True))
            else:
                corr = sp.Integer(0)

            ux_target_expr = ux_raw - corr * nx
            uy_target_no_force_expr = uy_raw - corr * ny
            uz_target_expr = uz_raw - corr * nz

        uy_target_expr = uy_target_no_force_expr - sp.Rational(1, 2) * a_lat * (theta.center - theta_ref)

        assignments = [
            ps.Assignment(rho_target_sym, flow_rho.center),
            ps.Assignment(ux_target_sym, ux_target_expr),
            ps.Assignment(uy_target_sym, uy_target_expr),
            ps.Assignment(uz_target_sym, uz_target_expr),
        ]

        for direction in stencil_dirs:
            cx, cy, cz = direction
            if cx == 0 and cy == 0 and cz == 0:
                continue

            incoming_direction = (-cx, -cy, -cz)
            q_in = dir_to_idx[incoming_direction]

            feq_target = pdf_eq(
                incoming_direction[0],
                incoming_direction[1],
                incoming_direction[2],
                rho_target_sym,
                ux_target_sym,
                uy_target_sym,
                uz_target_sym,
            )

            lhs = pdf[cx, cy, cz](q_in)
            rhs = sp.Piecewise(
                (feq_target, sp.Eq(bc_id[cx, cy, cz], target_bc)),
                (lhs, True),
            )
            assignments.append(ps.Assignment(lhs, rhs))

        return assignments

    open_boundary_serial_cfg = serial_cfg.copy()
    open_boundary_serial_cfg.skip_independence_check = True
    open_boundary_serial_cfg.allow_double_writes = True

    generate_sweep(
        sfg,
        "OpenBoundaryReconstructInletSerial",
        make_open_boundary_reconstruct_assignments(BC_INLET, "velocity"),
        open_boundary_serial_cfg,
        sparse=True,
    )
    generate_sweep(
        sfg,
        "OpenBoundaryReconstructOutletSerial",
        make_open_boundary_reconstruct_assignments(BC_OUTLET, "velocity"),
        open_boundary_serial_cfg,
        sparse=True,
    )
    generate_sweep(
        sfg,
        "OpenBoundaryReconstructPressureInSerial",
        make_open_boundary_reconstruct_assignments(BC_PRESSURE, "pressure_in"),
        open_boundary_serial_cfg,
        sparse=True,
    )
    generate_sweep(
        sfg,
        "OpenBoundaryReconstructPressureOutSerial",
        make_open_boundary_reconstruct_assignments(BC_PRESSURE, "pressure_out"),
        open_boundary_serial_cfg,
        sparse=True,
    )

    # Clamp sweeps for Dirichlet open boundaries.
    def make_clamp_open_boundary_assignments(target_bc):
        neighbor_conditions = []
        for dx, dy, dz in FACE_DIRS:
            neighbor_conditions.append(
                sp.And(
                    sp.Eq(bc_id[dx, dy, dz], target_bc),
                    sp.Eq(thermal_type[dx, dy, dz], THERMAL_DIRICHLET),
                )
            )

        has_dirichlet_neighbor = sp.Or(*neighbor_conditions)
        return [
            ps.Assignment(
                theta_tmp.center,
                sp.Piecewise(
                    (flow_theta.center, has_dirichlet_neighbor),
                    (theta_tmp.center, True),
                ),
            )
        ]

    generate_sweep(
        sfg,
        "ClampOpenBoundaryThetaTmpInletSerial",
        make_clamp_open_boundary_assignments(BC_INLET),
        serial_cfg,
        sparse=True,
    )
    generate_sweep(
        sfg,
        "ClampOpenBoundaryThetaTmpOutletSerial",
        make_clamp_open_boundary_assignments(BC_OUTLET),
        serial_cfg,
        sparse=True,
    )
    generate_sweep(
        sfg,
        "ClampOpenBoundaryThetaTmpPressureSerial",
        make_clamp_open_boundary_assignments(BC_PRESSURE),
        serial_cfg,
        sparse=True,
    )

    # Thermal boundary sweep.
    theta_sum = sp.Integer(0)
    theta_count = sp.Integer(0)
    for dx, dy, dz in FACE_DIRS:
        is_fluid_neighbor = sp.Eq(cell_type[dx, dy, dz], CELL_FLUID)
        theta_sum += sp.Piecewise((theta_tmp[dx, dy, dz], is_fluid_neighbor), (sp.Integer(0), True))
        theta_count += sp.Piecewise((sp.Integer(1), is_fluid_neighbor), (sp.Integer(0), True))

    theta_avg = theta_sum / (theta_count + sp.Float(1.0e-30))
    is_solid = sp.Eq(cell_type.center, CELL_SOLID)
    is_dirichlet = sp.Eq(thermal_type.center, THERMAL_DIRICHLET)
    is_adiabatic = sp.Eq(thermal_type.center, THERMAL_ADIABATIC)
    is_heatload = sp.Eq(thermal_type.center, THERMAL_HEATLOAD)

    thermal_theta_next = sp.Piecewise(
        (
            sp.Integer(2) * thermal_value.center - theta_avg,
            sp.And(is_solid, is_dirichlet),
        ),
        (
            theta_avg,
            sp.And(is_solid, is_adiabatic),
        ),
        (
            theta_avg + thermal_value.center * heatload_scale,
            sp.And(is_solid, is_heatload),
        ),
        (theta_tmp.center, True),
    )

    generate_sweep(
        sfg,
        "ApplyThermalBoundaryDenseSerial",
        [ps.Assignment(theta_tmp.center, thermal_theta_next)],
        serial_cfg,
        sparse=False,
    )
