// SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "src/FluidSimInternal.hpp"
#include "gen/FluidSim.hpp"
#include "walberla/experimental/sweep/SparseIndexList.hpp"

#include <functional>
#include <type_traits>
#include <utility>

namespace fluidsim
{

// Generated sparse index-list type used by runtime sweep/boundary containers.
using SparseCellIndexList = std::remove_cvref_t<
    decltype(std::declval<fluidsim::gen::LBM::StreamCollideSerial>().indexList())>;

struct CheckpointRegionInfo
{
    std::string name;
    GeometryRole role = GeometryRole::SolidObstacle;
    std::uintmax_t sourceFileBytes = std::uintmax_t(0);
    std::string sourceFileHash;
    real_t scale = real_t(1);
    walberla::Vector3<real_t> translateFraction = walberla::Vector3<real_t>(real_t(0));
};

struct NuRegionOutputInfo
{
    walberla::uint16_t regionId = walberla::uint16_t(0);
    std::string regionName;
    double lCharLatFine = 0.0;
    bool hasDeltaThetaOverride = false;
    double deltaThetaOverride = 0.0;
};

struct NuVtkFieldInfo
{
    walberla::uint16_t regionId = walberla::uint16_t(0);
    std::string regionName;
    walberla::BlockDataID valueFieldID;
    walberla::BlockDataID countFieldID;
};

struct FluidSimRuntimeBindings
{
    // Execution context.
    bool isRoot = false;
    CmdOptions cmd;

    // Timestepping controls.
    uint_t numTimesteps = uint_t(0);
    uint_t thetaUpdateEvery = uint_t(0);

    // Domain/block ownership.
    std::shared_ptr<walberla::StructuredBlockForest> blocks;
    std::shared_ptr<walberla::blockforest::BlockForest> blockForest;

    // Runtime phase callbacks.
    std::function<void()> updateThetaRef;

    // Host-state sync phases.
    // CPU: currently no-op. GPU: device->host sync for diagnostics/output/checkpoint.
    std::function<void()> syncRuntimeToHostTheta;
    std::function<void()> syncRuntimeToHostRhoVelTheta;
    std::function<void()> syncRuntimeToHostPdfThetaRhoVel;

    std::function<void()> applyOpenBoundary;

    // Communication phases (explicit start/wait GPU variant).
    std::function<void()> startCommunicatePdfTheta;
    std::function<void()> waitCommunicatePdfTheta;

    // Boundary/thermal and stream phases.
    std::function<void()> applyNoSlip;
    std::function<void()> thermalStep;
    std::function<void(walberla::IBlock*)> runStreamDense;
    std::function<void(walberla::IBlock*)> runStreamSparse;

    // Lifetime contract: these raw pointers alias setup-owned containers/state.
    // They are valid only for the immediate runFluidSimRuntime(binding) call made
    // inside runFluidSimSetupAndRuntime(), while setup locals are still alive.
    // Do not store/copy this binding for deferred use without refactoring ownership.
    std::vector<walberla::Block*>* fullFluidBlocks = nullptr;
    std::vector<walberla::Block*>* mixedBlocks = nullptr;
    std::vector<ThermalBCBlockEntry>* thermalBCBlocks = nullptr;

    // Sparse boundary/fluid index lists.
    SparseCellIndexList* fluidCellIndexList = nullptr;
    SparseCellIndexList* boundaryFluidIndexList = nullptr;

    // Runtime field IDs consumed by diagnostics/output.
    walberla::BlockDataID pdfID;
    walberla::BlockDataID densityID;
    walberla::BlockDataID velocityID;
    walberla::BlockDataID thetaID;

    // Runtime field IDs used by diagnostics/reductions.
    // CPU: same as main IDs. GPU: device field IDs used by GPU reductions.
    walberla::BlockDataID densityRuntimeID;
    walberla::BlockDataID velocityRuntimeID;
    walberla::BlockDataID thetaRuntimeID;
    walberla::BlockDataID cellTypeID;
    walberla::BlockDataID bcIdID;
    walberla::BlockDataID thermalTypeID;
    walberla::BlockDataID thermalValueID;
    walberla::BlockDataID regionIdID;

    real_t* currentThetaRef = nullptr;
    real_t thetaDirichletMax = real_t(0);
    real_t thetaDirichletMin = real_t(0);

    // Output/checkpoint descriptors.
    CheckpointPaths checkpointPaths;
    std::vector<CheckpointRegionInfo> checkpointRegions;
    std::vector<NuRegionOutputInfo> nuOutputRegions;
    std::vector<NuVtkFieldInfo> nuVtkFields;

    // Geometry/domain metadata for checkpoint and diagnostics.
    walberla::Vector3<double> domainSizePhys;
    walberla::Vector3<double> paddingSizePhys;
    walberla::Vector3<double> fullSizePhys;
    walberla::Vector3<uint_t> interiorFineCells;
    walberla::Vector3<uint_t> totalFineCells;
    walberla::Vector3<uint_t> cellsPerBlock;
    walberla::Vector3<uint_t> paddingFineCells;
    walberla::Vector3<bool> periodicFlags;
    bool pruneOpenEdge = false;

    uint_t vtkWriteFrequency = uint_t(0);
    double dtPhysFine = 0.0;
};

int runFluidSimRuntime(FluidSimRuntimeBindings& binding);

} // namespace fluidsim
