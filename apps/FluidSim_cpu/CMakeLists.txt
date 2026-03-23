# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
# Build prerequisites.
if( NOT WALBERLA_BUILD_WITH_CODEGEN )
    message(STATUS "FluidSim_cpu requires WALBERLA_BUILD_WITH_CODEGEN=ON - skipping")
    return()
endif()

if( NOT WALBERLA_ENABLE_SWEEPGEN )
    message(STATUS "FluidSim_cpu requires WALBERLA_ENABLE_SWEEPGEN=ON - skipping")
    return()
endif()

if( NOT WALBERLA_BUILD_WITH_OPENMESH )
    message(STATUS "FluidSim_cpu requires WALBERLA_BUILD_WITH_OPENMESH=ON - skipping")
    return()
endif()

# Source generation configuration.
set( _fluidsim_target FluidSim_cpu )
set( _fluidsim_codegen_suffixes hpp cpp )
set( _fluidsim_codegen_scripts
    codegen/FluidSim.py
    codegen/ThetaUpdate.py
)

# Target definition.
add_executable( ${_fluidsim_target}
    FluidSim_cpu.cpp
    src/FluidSimSetup.cpp
    src/FluidSimRuntime.cpp
)

# Source generation.
walberla_generate_sources(
    ${_fluidsim_target}
    SCRIPTS
        ${_fluidsim_codegen_scripts}
    FILE_EXTENSIONS ${_fluidsim_codegen_suffixes}
)

# Generated-source warning hygiene.
set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/_gen/FluidSim_cpu/gen/ThetaUpdate.cpp
    PROPERTIES COMPILE_OPTIONS "-Wno-parentheses"
)

# Include paths.
target_include_directories( ${_fluidsim_target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies.
target_link_libraries( ${_fluidsim_target}
    PRIVATE
    walberla::blockforest
    walberla::core
    walberla::domain_decomposition
    walberla::field
    walberla::geometry
    walberla::lbm
    walberla::mesh
    walberla::stencil
    walberla::timeloop
    walberla::vtk
)
