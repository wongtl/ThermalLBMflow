# SPDX-FileCopyrightText: 2017-2026 waLBerla contributors
# SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
# SPDX-License-Identifier: GPL-3.0-or-later
# Benchmarks
if ( WALBERLA_BUILD_BENCHMARKS )
    add_subdirectory ( benchmarks )
endif()


# Tools
if ( WALBERLA_BUILD_TOOLS )
    add_subdirectory ( tools )
endif()


# Tutorials
if ( WALBERLA_BUILD_TUTORIALS )
    add_subdirectory ( tutorials )
endif()

# Examples
if ( WALBERLA_BUILD_EXAMPLES )
    add_subdirectory (examples )
endif()

# Showcases
if ( WALBERLA_BUILD_SHOWCASES )
    add_subdirectory ( showcases )
endif()

# Python module
if ( WALBERLA_BUILD_WITH_PYTHON )
    add_subdirectory( pythonmodule )
endif()

# FluidSim CPU app
if ( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/FluidSim_cpu )
    add_subdirectory ( FluidSim_cpu )
endif()

# FluidSim GPU app
if ( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/FluidSim_gpu )
    add_subdirectory ( FluidSim_gpu )
endif()
