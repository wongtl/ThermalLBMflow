# ThermalLBMflow Documentation Outline

This document captures the proposed structure for the main technical
documentation of ThermalLBMflow. It is intended as a writing scaffold for a
current, repo-aligned document based on the public open-source codebase rather
than the earlier pre-GPU, AMR-oriented draft.

## Proposed Title

**ThermalLBMflow: A 3D CPU/GPU Thermal Lattice Boltzmann Solver for
Buoyancy-Driven Flow in Colored PLY Geometries**

## Abstract

Summarize the solver as it exists now:

- D3Q19 TRT lattice Boltzmann momentum solver
- explicit cell-centered thermal update
- Boussinesq buoyancy coupling through `theta_ref`
- colored PLY geometry for thermal boundary-condition assignment
- shared CPU/GPU solver structure with backend-specific execution details
- single-level structured block-forest runtime
- checkpoint/restart, diagnostics, and VTK output

The abstract should avoid references to AMR, subcycling, or multilevel runtime
behavior, because those are not part of the current public solver path.

## Table of Contents

### 1. Introduction

#### 1.1 Motivation and target problem class
Introduce the class of buoyancy-driven and mixed-convection flows that the
solver targets, and explain why complex embedded geometries motivate a
mesh-color-driven boundary workflow.

#### 1.2 Governing equations and modeling assumptions
State the incompressible-flow and thermal-transport assumptions, the
Boussinesq approximation, and the role of the dimensionless temperature field
used by the solver.

#### 1.3 Nondimensional groups and physical-to-lattice mapping
Present the Rayleigh, Prandtl, and related quantities actually used in the
implementation, and explain how physical inputs are converted into lattice
parameters.

#### 1.4 Numerical method overview
Give the high-level method pairing: TRT-based D3Q19 momentum update, explicit
thermal update, and buoyancy coupling through the evolving reference
temperature.

#### 1.5 Geometry and boundary-condition representation
Explain the colored PLY workflow, embedded solid treatment, no-slip momentum
boundaries, and color-mapped thermal/open-boundary region assignment.

#### 1.6 Current scope and limitations
State the current public solver scope clearly: single-level runtime, CPU and
GPU backends, one-rank-per-GPU policy on the GPU path, and the absence of the
older AMR/subcycling runtime from this codebase.

### 2. Configuration, Inputs, and Outputs

#### 2.1 User-facing workflows
Describe the intended ways users interact with the project: local CPU workflow,
cluster/manual CPU workflow, and cluster/manual GPU workflow.

#### 2.2 Parameter-file structure
Document the top-level blocks in `apps/shared/params/FluidSim.prm` and explain
which blocks control physics, numerics, geometry, output, and runtime cadence.

#### 2.3 Runtime flags and launcher-owned controls
Separate executable CLI flags from launcher-owned environment variables so the
control surface is clear.

#### 2.4 Boundary-condition schema from mesh colors
Explain how `ColorBC.Region` definitions map RGB colors to region names and
physics, including the supported boundary categories and validation rules.

#### 2.5 Diagnostics, checkpoints, and VTK output
Document what the solver writes during execution: minimal logs, thermal logs,
checkpoint directories, and ParaView-compatible VTK output.

#### 2.6 Restart contract and metadata validation
Explain what restart data is stored, what metadata is validated, and which
consistency checks are enforced before a restart is accepted.

### 3. Solver Workflow and Backend Execution

#### 3.1 Shared software architecture
Introduce the common structure used by both backends: setup/runtime split,
generated kernels, shared runtime control flow, and backend-specific callback
binding.

#### 3.2 CPU/GPU execution model at a glance
Provide a compact comparison table covering memory model, parallel model,
communication, reductions, synchronization points, and backend-specific
constraints.

#### 3.3 Startup sequence: shared pipeline
Describe the common startup path: parsing inputs, deriving physical/numerical
constants, reading geometry, building the structured domain, allocating
fields, mapping boundary regions, and initializing state.

#### 3.4 Startup sequence: backend-specific divergences
Capture where CPU and GPU setup differ in meaningful ways, such as OpenMP
handling, device mirrors, CUDA reductions, and host-device synchronization
requirements.

#### 3.5 Single-level timestep sequence
Present the current timestep order as a single shared sequence rather than
separate CPU and GPU narratives.

#### 3.6 Phase-by-phase execution details
Walk through the major runtime phases in detail:

- `theta_ref` update
- momentum stream-collide
- communication
- velocity and open-boundary application
- thermal update and thermal boundary handling
- diagnostics, checkpointing, and VTK output

Backend differences should be described inline only where they change how a
phase is executed.

#### 3.7 Diagnostics and cadence-controlled output
Explain how logging, Nusselt evaluation, checkpoint writing, and VTK output are
scheduled and how those cadences interact with the main loop.

#### 3.8 Validation and fail-fast behavior
Document the solver's current validation philosophy: strict parsing, geometry
consistency checks, restart metadata checks, and early aborts on malformed or
physically inconsistent input.

### 4. Performance-Oriented Implementation

#### 4.1 Shared performance design choices
Document the optimizations common to both backends, such as code generation,
dense plus sparse traversal, geometry preprocessing, and cadence-controlled
diagnostics and I/O.

#### 4.2 CPU backend optimizations
Describe CPU-specific choices including OpenMP `parallelMode`, outer versus
inner parallelism, and how irregular kernels are treated.

#### 4.3 GPU backend optimizations
Describe GPU-specific choices including device mirrors, custom CUDA
reductions, selective host synchronization, and the one-rank-per-GPU policy.

#### 4.4 Communication, diagnostics, and I/O cost control
Discuss how halo exchange, reductions, checkpoint writes, VTK output, and
profiling hooks are managed so they do not dominate runtime cost.

#### 4.5 Practical tuning knobs and tradeoffs
Collect the operational tuning advice that matters most in practice: CPU
threading mode selection, GPU architecture selection, rank/GPU mapping,
transport mode choice, and output-cadence tradeoffs.

### 5. Current Limitations and Extension Paths

#### 5.1 Current solver limitations
State the important boundaries of the current implementation, including
single-level execution, GPU runtime assumptions, and any workflow limitations
that users should know upfront.

#### 5.2 Backend asymmetries
Document the intentional CPU/GPU differences that remain, such as CPU-only
`parallelMode` and GPU-specific reduction and synchronization logic.

#### 5.3 Extension paths
Outline promising future directions for the codebase and documentation, such as
broader local GPU workflows, deeper performance tuning guidance, or future
algorithmic extensions.

## Writing Notes

- Keep Sections 3 and 4 shared-first rather than duplicating full CPU and GPU
  narratives.
- Use one CPU/GPU comparison table early in Section 3 to orient the reader.
- Remove all legacy AMR/subcycling/multilevel discussion unless it is clearly
  marked as historical context outside the current public solver.
- Keep implementation details aligned with the open-source repository rather
  than earlier internal project states.
