// SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "src/FluidSimRuntime.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace fluidsim::gpureduce
{

#if defined(FLUIDSIM_GPU_BUILD) && defined(WALBERLA_BUILD_WITH_CUDA)

struct GpuNuEntryPacked
{
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;
    int32_t slot = -1;
    walberla::real_t thermalValue = walberla::real_t(0);
    walberla::uint8_t fluidNeighborCount = walberla::uint8_t(0);
    walberla::uint8_t fluidNeighborDirIndices[6] = {};
};

class NuBoundaryCacheCuda
{
public:
    NuBoundaryCacheCuda() = default;
    ~NuBoundaryCacheCuda();

    NuBoundaryCacheCuda(const NuBoundaryCacheCuda&) = delete;
    NuBoundaryCacheCuda& operator=(const NuBoundaryCacheCuda&) = delete;
    NuBoundaryCacheCuda(NuBoundaryCacheCuda&&) = delete;
    NuBoundaryCacheCuda& operator=(NuBoundaryCacheCuda&&) = delete;

    const GpuNuEntryPacked* getOrCreateDeviceEntries(
        const std::vector<ThermalBoundaryCell>& entries,
        const std::unordered_map<walberla::uint16_t, size_t>& regionSlotById,
        size_t& outCount);

private:
    struct Entry
    {
        GpuNuEntryPacked* devicePtr = nullptr;
        size_t count = size_t(0);
    };

    std::unordered_map<const std::vector<ThermalBoundaryCell>*, Entry> cache_;
};

bool reduceThetaRefLocal(
    const std::vector<walberla::Block*>& fullFluidBlocks,
    const std::vector<walberla::Block*>& mixedBlocks,
    SparseCellIndexList& fluidCellIndexList,
    walberla::BlockDataID thetaRuntimeID,
    double thetaRefCellVolume,
    double& thetaSumLocal,
    double& volumeLocal);

bool reduceMinimalLocal(
    const std::vector<walberla::Block*>& fullFluidBlocks,
    const std::vector<walberla::Block*>& mixedBlocks,
    SparseCellIndexList& fluidCellIndexList,
    walberla::BlockDataID velocityRuntimeID,
    walberla::BlockDataID thetaRuntimeID,
    walberla::BlockDataID densityRuntimeID,
    double cellVolume,
    double& uMaxSqLocal,
    double& uySqVolumeLocal,
    double& uSqVolumeLocal,
    double& energyLocal,
    double& massLocal);

bool reduceNuLocal(
    const std::vector<ThermalBCBlockEntry>& thermalBCBlocks,
    walberla::BlockDataID thetaRuntimeID,
    const std::unordered_map<walberla::uint16_t, size_t>& regionSlotById,
    double invDx,
    double faceArea,
    NuBoundaryCacheCuda& cache,
    std::vector<double>& fluxAreaLocal,
    std::vector<double>& areaLocal);

#endif

} // namespace fluidsim::gpureduce
