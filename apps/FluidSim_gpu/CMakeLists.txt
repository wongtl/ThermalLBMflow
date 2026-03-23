# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
# Build prerequisites.
if( NOT WALBERLA_BUILD_WITH_CUDA )
    message(STATUS "FluidSim_gpu requires WALBERLA_BUILD_WITH_CUDA=ON - skipping")
    return()
endif()

if( NOT WALBERLA_BUILD_WITH_CODEGEN )
    message(STATUS "FluidSim_gpu requires WALBERLA_BUILD_WITH_CODEGEN=ON - skipping")
    return()
endif()

if( NOT WALBERLA_ENABLE_SWEEPGEN )
    message(STATUS "FluidSim_gpu requires WALBERLA_ENABLE_SWEEPGEN=ON - skipping")
    return()
endif()

# Source generation configuration.
set( _fluidsim_target FluidSim_gpu )
set( _fluidsim_codegen_suffixes hpp cu )
set( _fluidsim_codegen_scripts
    codegen/FluidSim.py
    codegen/ThetaUpdate.py
)

# CUDA architecture configuration.
# Default to mixed cluster support (V100 + L40S + H100), but keep this overridable
# from the CMake cache / command line.
if( NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" )
    set( CMAKE_CUDA_ARCHITECTURES "70;89;90" CACHE STRING "CUDA archs (V100=70, L40S=89, H100=90)" )
endif()

# Target definition.
add_executable( ${_fluidsim_target}
    FluidSim_gpu.cpp
    src/FluidSimSetup.cpp
    src/FluidSimRuntime.cpp
)

set_target_properties( ${_fluidsim_target} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# CUDA release tuning.
target_compile_options( ${_fluidsim_target} PRIVATE
    # We intentionally keep sm_70 (V100) in the build; silence nvcc deprecation noise.
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    # Suppress cudafe warning #611 from external walberla VTK headers.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=611>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xptxas=-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>
)

# Additional target sources.
target_sources( ${_fluidsim_target} PRIVATE
    src/GpuReductions.cu
)

# Source generation.
walberla_generate_sources(
    ${_fluidsim_target}
    SCRIPTS
        ${_fluidsim_codegen_scripts}
    FILE_EXTENSIONS ${_fluidsim_codegen_suffixes}
)

# Include paths.
target_include_directories( ${_fluidsim_target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compile definitions.
target_compile_definitions( ${_fluidsim_target} PRIVATE FLUIDSIM_GPU_BUILD=1 )

# Link dependencies.
target_link_libraries( ${_fluidsim_target}
    PRIVATE
    walberla::blockforest
    walberla::core
    walberla::domain_decomposition
    walberla::field
    walberla::geometry
    walberla::gpu
    walberla::lbm
    walberla::mesh
    walberla::stencil
    walberla::timeloop
    walberla::vtk
)

target_link_libraries( ${_fluidsim_target} PRIVATE CUDA::cudart )
