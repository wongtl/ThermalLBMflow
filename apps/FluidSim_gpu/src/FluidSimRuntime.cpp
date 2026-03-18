#include "timeloop/all.h"
#include "stencil/all.h"
#include "field/FileIO.h"
#include "vtk/VTKOutput.h"
#include "walberla/experimental/sweep/SparseIndexList.hpp"

#if defined(FLUIDSIM_GPU_BUILD) && defined(WALBERLA_BUILD_WITH_CUDA)
#include <cuda_runtime.h>
#include "gpu/NVTX.h"
#endif

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using walberla::real_t;
using walberla::uint_t;

#define FLUIDSIM_RUNTIME_ONLY
#include "src/FluidSimRuntime.hpp"
#include "src/GpuReductions.hpp"
#undef FLUIDSIM_RUNTIME_ONLY

#include "../../shared/helpers/VtkTimeMetadataHelpers.hpp"

namespace fluidsim
{

int runFluidSimRuntime(FluidSimRuntimeBindings& binding)
{
    // Validate required runtime bindings.
    WALBERLA_CHECK(bool(binding.updateThetaRef), "updateThetaRef callback must be set.");
    WALBERLA_CHECK(bool(binding.syncRuntimeToHostTheta));
    WALBERLA_CHECK(bool(binding.syncRuntimeToHostRhoVelTheta));
    WALBERLA_CHECK(bool(binding.syncRuntimeToHostPdfThetaRhoVel));
    WALBERLA_CHECK(bool(binding.applyOpenBoundary));
    WALBERLA_CHECK(
        bool(binding.startCommunicatePdfTheta) && bool(binding.waitCommunicatePdfTheta),
        "Both startCommunicatePdfTheta and waitCommunicatePdfTheta callbacks must be set.");
    WALBERLA_CHECK(bool(binding.applyNoSlip));
    WALBERLA_CHECK(bool(binding.thermalStep));
    WALBERLA_CHECK(bool(binding.runStreamDense));
    WALBERLA_CHECK(bool(binding.runStreamSparse));
    WALBERLA_CHECK_NOT_NULLPTR(binding.fullFluidBlocks);
    WALBERLA_CHECK_NOT_NULLPTR(binding.mixedBlocks);
    WALBERLA_CHECK_NOT_NULLPTR(binding.thermalBCBlocks);
    WALBERLA_CHECK_NOT_NULLPTR(binding.fluidCellIndexList);
    WALBERLA_CHECK_NOT_NULLPTR(binding.boundaryFluidIndexList);
    WALBERLA_CHECK_NOT_NULLPTR(binding.currentThetaRef);

    // Unpack immutable runtime inputs.
    const bool isRoot = binding.isRoot;
    auto& cmd = binding.cmd;

    const uint_t numTimesteps = binding.numTimesteps;
    const uint_t thetaUpdateEvery = binding.thetaUpdateEvery;

    // Unpack block storage and runtime callbacks.
    auto blocks = binding.blocks;
    auto blockForest = binding.blockForest;

    auto& updateThetaRef = binding.updateThetaRef;
    auto& syncRuntimeToHostTheta = binding.syncRuntimeToHostTheta;
    auto& syncRuntimeToHostRhoVelTheta = binding.syncRuntimeToHostRhoVelTheta;
    auto& syncRuntimeToHostPdfThetaRhoVel = binding.syncRuntimeToHostPdfThetaRhoVel;
    auto& applyOpenBoundary = binding.applyOpenBoundary;
    auto& startCommunicatePdfTheta = binding.startCommunicatePdfTheta;
    auto& waitCommunicatePdfTheta = binding.waitCommunicatePdfTheta;
    auto& applyNoSlip = binding.applyNoSlip;
    auto& thermalStep = binding.thermalStep;
    auto& runStreamDense = binding.runStreamDense;
    auto& runStreamSparse = binding.runStreamSparse;

    auto& fullFluidBlocks = *binding.fullFluidBlocks;
    auto& mixedBlocks = *binding.mixedBlocks;
    auto& thermalBCBlocks = *binding.thermalBCBlocks;

    auto& fluidCellIndexList = *binding.fluidCellIndexList;
    auto& boundaryFluidIndexList = *binding.boundaryFluidIndexList;

    const auto pdfID = binding.pdfID;
    const auto densityID = binding.densityID;
    const auto velocityID = binding.velocityID;
    const auto thetaID = binding.thetaID;
    const auto densityRuntimeID = binding.densityRuntimeID;
    const auto velocityRuntimeID = binding.velocityRuntimeID;
    const auto thetaRuntimeID = binding.thetaRuntimeID;
    const auto cellTypeID = binding.cellTypeID;
    const auto bcIdID = binding.bcIdID;
    const auto thermalTypeID = binding.thermalTypeID;
    const auto thermalValueID = binding.thermalValueID;
    const auto regionIdID = binding.regionIdID;

    auto& currentThetaRef = *binding.currentThetaRef;
    const auto nuFacePrimitive = [](double thetaWall, double thetaFluid, double invDxLocal) {
        return (thetaWall - thetaFluid) * (2.0 * invDxLocal);
    };
    const auto nuOutputLabelFromRegionName = [](const std::string& regionName) {
        std::string label = regionName;
        const std::string upperName = toUpper(regionName);
        constexpr const char* prefix = "DIRICHLET";
        constexpr size_t prefixLen = size_t(9);
        if (upperName.rfind(prefix, size_t(0)) == size_t(0))
        {
            label = regionName.substr(prefixLen);
            while (!label.empty() && (label.front() == '_' || label.front() == '-' || label.front() == ' '))
                label.erase(label.begin());
        }
        if (label.empty())
            label = regionName;
        return label;
    };

    const auto checkpointPaths = binding.checkpointPaths;
    const auto& geometryRegions = binding.checkpointRegions;
    const auto& nuOutputRegions = binding.nuOutputRegions;
    const auto& nuVtkFields = binding.nuVtkFields;
    std::unordered_map<walberla::uint16_t, size_t> nuOutputSlotByRegionId;
    nuOutputSlotByRegionId.reserve(nuOutputRegions.size());
    for (size_t idx = size_t(0); idx < nuOutputRegions.size(); ++idx)
        nuOutputSlotByRegionId.emplace(nuOutputRegions[idx].regionId, idx);

    const auto domainSizePhys = binding.domainSizePhys;
    const auto paddingSizePhys = binding.paddingSizePhys;
    const auto fullSizePhys = binding.fullSizePhys;
    const auto interiorFineCells = binding.interiorFineCells;
    const auto totalFineCells = binding.totalFineCells;
    const auto cellsPerBlock = binding.cellsPerBlock;
    const auto paddingFineCells = binding.paddingFineCells;
    const auto periodicFlags = binding.periodicFlags;

    constexpr const char* outputBaseDir = kOutputBaseDir;
    const double dtPhysFine = binding.dtPhysFine;

    WALBERLA_ROOT_SECTION()
    {
        ensureDirectory(std::filesystem::path(outputBaseDir), "create vtk output base directory");
    }
    WALBERLA_MPI_SECTION()
    {
        MPI_Barrier(walberla::mpi::MPIManager::instance()->comm());
    }

    using LbStencil = walberla::stencil::D3Q19;
    using PdfField = walberla::field::GhostLayerField<real_t, LbStencil::Q>;
    using ScalarField = walberla::field::GhostLayerField<real_t, 1>;
    using VecField = walberla::field::GhostLayerField<real_t, 3>;
    using CellTypeField = walberla::field::GhostLayerField<walberla::uint8_t, 1>;
    using BcField = walberla::field::GhostLayerField<walberla::uint16_t, 1>;
    using ThermalTypeField = walberla::field::GhostLayerField<walberla::uint8_t, 1>;
    using RegionIdField = walberla::field::GhostLayerField<walberla::uint16_t, 1>;

    const char* profileModeEnv = std::getenv("FLUIDSIM_PROFILE_MODE");
    std::string profileMode = profileModeEnv ? std::string(profileModeEnv) : std::string("none");
    std::transform(profileMode.begin(), profileMode.end(), profileMode.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    const bool nvtxTimestepPhases = (profileMode == "nsys" || profileMode == "ncu");
    const uint_t minimalLogCadence = cmd.minimalLogsEvery;
    const uint_t thermalLogCadence = cmd.thermalLogsEvery;
    const uint_t checkpointCadence = cmd.checkpointEvery;
    const uint_t vtkWriteFrequency = binding.vtkWriteFrequency;
    const bool vtkWriteAtStepZero = cmd.vtkInit;
    auto cadenceDue = [&](uint_t step, uint_t cadence, bool includeZero) -> bool {
        if (cadence == uint_t(0))
            return false;
        if (!includeZero && step == uint_t(0))
            return false;
        return (step % cadence) == uint_t(0);
    };

    // Core single-level timestep order.
    auto runStreamLevel = [&]() {
        for (auto* block : fullFluidBlocks)
            runStreamDense(block);
        for (auto* block : mixedBlocks)
            runStreamSparse(block);
    };
    auto runNamedPhase = [&](const char* phaseName, const uint32_t phaseColor, const auto& fn) {
        if (nvtxTimestepPhases)
        {
#if defined(FLUIDSIM_GPU_BUILD) && defined(WALBERLA_BUILD_WITH_CUDA)
            walberla::gpu::NvtxRange range(phaseName, phaseColor);
            fn();
            return;
#endif
        }
        fn();
    };
    auto runSingleLevelStep = [&]() {
        runNamedPhase("FluidSim/Stream", 0x4e79a7, [&]() { runStreamLevel(); });
        // GPU communication exposes start/wait phases explicitly for profiling clarity.
        runNamedPhase("FluidSim/CommStart", 0x59a14f, [&]() {
            startCommunicatePdfTheta();
        });
        runNamedPhase("FluidSim/CommWait", 0x76b7b2, [&]() {
            waitCommunicatePdfTheta();
        });
        // Keep validated semantics: Stream -> Communicate -> Boundary -> Thermal.
        // Start+wait are intentionally back-to-back in production to avoid overlap races.
        runNamedPhase("FluidSim/Boundary", 0xe15759, [&]() {
            applyNoSlip();
            applyOpenBoundary();
        });
        runNamedPhase("FluidSim/Thermal", 0xb07aa1, [&]() {
            thermalStep();
        });
    };

    // Pre-timestep refresh to keep initial boundary/theta state consistent.
    {
        if (thetaUpdateEvery > uint_t(0))
            updateThetaRef();
        applyOpenBoundary();
    }

    // GPU host-sync scope tracking for output/checkpoint hooks.
    enum class HostSyncScope : uint8_t
    {
        None = uint8_t(0),
        Theta = uint8_t(1),
        RhoVelTheta = uint8_t(2),
        Full = uint8_t(3)
    };
    uint_t hostSyncedStep = std::numeric_limits<uint_t>::max();
    HostSyncScope hostSyncedScope = HostSyncScope::None;
    auto ensureHostState = [&](uint_t step, HostSyncScope requestedScope) {
        if (hostSyncedStep != step)
        {
            hostSyncedStep = step;
            hostSyncedScope = HostSyncScope::None;
        }
        if (static_cast<uint8_t>(hostSyncedScope) >= static_cast<uint8_t>(requestedScope))
            return;

        if (requestedScope == HostSyncScope::Theta)
            syncRuntimeToHostTheta();
        else if (requestedScope == HostSyncScope::RhoVelTheta)
            syncRuntimeToHostRhoVelTheta();
        else
            syncRuntimeToHostPdfThetaRhoVel();

        hostSyncedScope = requestedScope;
    };

    // Checkpoint metadata/file helpers.
    auto writeCheckpointMetadataFields = [&](std::ostream& out) {
        out << "region_count=" << geometryRegions.size() << "\n";
        for (size_t regionIdx = size_t(0); regionIdx < geometryRegions.size(); ++regionIdx)
        {
            const auto& region = geometryRegions[regionIdx];
            out << "region_" << regionIdx
                << "=" << region.name
                << "|" << (region.role == GeometryRole::FluidContainer ? "FLUID_CONTAINER" : "SOLID_OBSTACLE")
                << "|" << region.sourceFileBytes
                << "|" << region.sourceFileHash
                << "|" << double(region.scale)
                << "|" << vec3ToCsv(region.translateFraction) << "\n";
        }
        out << "domain_size=" << vec3ToCsv(domainSizePhys) << "\n";
        out << "padding_size=" << vec3ToCsv(paddingSizePhys) << "\n";
        out << "full_size=" << vec3ToCsv(fullSizePhys) << "\n";
        out << "interior_fine_cells=" << vec3ToCsv(interiorFineCells) << "\n";
        out << "total_fine_cells=" << vec3ToCsv(totalFineCells) << "\n";
        out << "cells_per_block=" << vec3ToCsv(cellsPerBlock) << "\n";
        out << "padding_cells=" << vec3ToCsv(paddingFineCells) << "\n";
        out << "periodic=" << (periodicFlags[0] ? 1 : 0) << ","
            << (periodicFlags[1] ? 1 : 0) << ","
            << (periodicFlags[2] ? 1 : 0) << "\n";
    };

    auto cleanupCheckpointAuxDirs = [&](const std::filesystem::path& checkpointDir, const char* context) {
        const auto checkpointStagingDir =
            checkpointDir.parent_path() / (checkpointDir.filename().string() + "_new");
        const auto checkpointBackupDir =
            checkpointDir.parent_path() / (checkpointDir.filename().string() + "_old");

        WALBERLA_ROOT_SECTION()
        {
            auto cleanupDir = [&](const std::filesystem::path& dir) {
                std::error_code ec;
                std::filesystem::remove_all(dir, ec);
                if (ec)
                {
                    WALBERLA_LOG_WARNING("Failed to " << context << " '" << dir.string()
                                         << "': " << ec.message());
                }
            };
            cleanupDir(checkpointStagingDir);
            cleanupDir(checkpointBackupDir);
        }
        WALBERLA_MPI_SECTION()
        {
            MPI_Barrier(walberla::mpi::MPIManager::instance()->comm());
        }
    };

    auto writeCheckpointState = [&](uint_t checkpointStep,
                                    const walberla::BlockDataID pdfWriteID,
                                    const walberla::BlockDataID densityWriteID,
                                    const walberla::BlockDataID velocityWriteID,
                                    const walberla::BlockDataID thetaWriteID) {
        const auto checkpointDir = checkpointPaths.forestFile.parent_path();
        const auto checkpointStagingDir =
            checkpointDir.parent_path() / (checkpointDir.filename().string() + "_new");
        const auto checkpointBackupDir =
            checkpointDir.parent_path() / (checkpointDir.filename().string() + "_old");

        CheckpointPaths stagedCheckpointPaths = checkpointPaths;
        stagedCheckpointPaths.forestFile = checkpointStagingDir / checkpointPaths.forestFile.filename();
        stagedCheckpointPaths.pdfFile = checkpointStagingDir / checkpointPaths.pdfFile.filename();
        stagedCheckpointPaths.densityFile = checkpointStagingDir / checkpointPaths.densityFile.filename();
        stagedCheckpointPaths.velocityFile = checkpointStagingDir / checkpointPaths.velocityFile.filename();
        stagedCheckpointPaths.thetaFile = checkpointStagingDir / checkpointPaths.thetaFile.filename();
        stagedCheckpointPaths.metaFile = checkpointStagingDir / checkpointPaths.metaFile.filename();

        WALBERLA_ROOT_SECTION()
        {
            ensureDirectory(checkpointDir.parent_path(), "create checkpoint parent directory");

            std::error_code ec;
            std::filesystem::remove_all(checkpointStagingDir, ec);
            if (ec)
            {
                WALBERLA_ABORT("Failed to clear checkpoint staging directory: " << checkpointStagingDir.string()
                               << " (" << ec.message() << ")");
            }
            ensureDirectory(checkpointStagingDir, "create checkpoint staging directory");
        }
        WALBERLA_MPI_SECTION()
        {
            MPI_Barrier(walberla::mpi::MPIManager::instance()->comm());
        }

        walberla::field::writeToFile<PdfField>(stagedCheckpointPaths.pdfFile.string(), blocks->getBlockStorage(), pdfWriteID);
        walberla::field::writeToFile<ScalarField>(stagedCheckpointPaths.densityFile.string(), blocks->getBlockStorage(), densityWriteID);
        walberla::field::writeToFile<VecField>(stagedCheckpointPaths.velocityFile.string(), blocks->getBlockStorage(), velocityWriteID);
        walberla::field::writeToFile<ScalarField>(stagedCheckpointPaths.thetaFile.string(), blocks->getBlockStorage(), thetaWriteID);
        blockForest->saveToFile(stagedCheckpointPaths.forestFile.string());

        WALBERLA_MPI_SECTION()
        {
            MPI_Barrier(walberla::mpi::MPIManager::instance()->comm());
        }

        WALBERLA_ROOT_SECTION()
        {
            std::ofstream metaOut(stagedCheckpointPaths.metaFile);
            if (!metaOut)
                WALBERLA_ABORT("Failed to write checkpoint metadata file: " << stagedCheckpointPaths.metaFile.string());
            metaOut << std::setprecision(std::numeric_limits<double>::max_digits10);
            metaOut << "step=" << checkpointStep << "\n";
            metaOut << "checkpoint_format=field_io_v1\n";
            writeCheckpointMetadataFields(metaOut);
            metaOut.flush();
            if (!metaOut)
                WALBERLA_ABORT("Failed while writing checkpoint metadata file: " << stagedCheckpointPaths.metaFile.string());
        }

        WALBERLA_MPI_SECTION()
        {
            MPI_Barrier(walberla::mpi::MPIManager::instance()->comm());
        }

        WALBERLA_ROOT_SECTION()
        {
            std::error_code ec;
            std::filesystem::remove_all(checkpointBackupDir, ec);
            if (ec)
            {
                WALBERLA_LOG_WARNING("Failed to clear previous checkpoint backup directory: "
                                     << checkpointBackupDir.string() << " (" << ec.message() << ")");
            }

            ec.clear();
            const bool checkpointDirExists = std::filesystem::exists(checkpointDir, ec);
            if (ec)
            {
                WALBERLA_ABORT("Failed to query checkpoint directory state: "
                               << checkpointDir.string() << " (" << ec.message() << ")");
            }

            if (checkpointDirExists)
            {
                ec.clear();
                std::filesystem::rename(checkpointDir, checkpointBackupDir, ec);
                if (ec)
                {
                    WALBERLA_ABORT("Failed to rotate checkpoint directory to backup: "
                                   << checkpointDir.string() << " -> " << checkpointBackupDir.string()
                                   << " (" << ec.message() << ")");
                }
            }

            ec.clear();
            std::filesystem::rename(checkpointStagingDir, checkpointDir, ec);
            if (ec)
            {
                std::error_code rollbackEc;
                if (checkpointDirExists)
                    std::filesystem::rename(checkpointBackupDir, checkpointDir, rollbackEc);

                WALBERLA_ABORT("Failed to promote checkpoint staging directory: "
                               << checkpointStagingDir.string() << " -> " << checkpointDir.string()
                               << " (" << ec.message() << ")"
                               << (checkpointDirExists
                                       ? std::string("; rollback ") +
                                             (rollbackEc ? "failed: " + rollbackEc.message() : "succeeded")
                                       : std::string("; no rollback needed")));
            }
        }

        WALBERLA_MPI_SECTION()
        {
            MPI_Barrier(walberla::mpi::MPIManager::instance()->comm());
        }

        cleanupCheckpointAuxDirs(checkpointDir, "cleanup checkpoint auxiliary directory");

        if (isRoot)
            WALBERLA_LOG_INFO("CHECKPOINT t=" << checkpointStep);
    };

    // Main timeloop and diagnostics.
    walberla::SweepTimeloop loop(blocks->getBlockStorage(), numTimesteps);
    loop.addFuncBeforeTimeStep([&]() {
        const uint_t step = loop.getCurrentTimeStep();
        // Step 0 is handled by pre-loop refresh; in-loop cadence starts after step 0.
        if (cadenceDue(step, thetaUpdateEvery, /* includeZero */ false))
            runNamedPhase("FluidSim/ThetaRef", 0xf28e2b, [&]() { updateThetaRef(); });
        runSingleLevelStep();
    }, "TimestepStep");

    // Minimal diagnostics logger.
    if (minimalLogCadence > uint_t(0))
    {
        double updatesPerCoarseStepLocal = 0.0;
        double fluidVolumeLocal = 0.0;
        {
            for (auto* block : fullFluidBlocks)
            {
                const auto bb = blocks->getBlockCellBB(*block);
                const double fluidCellCount = double(bb.numCells());
                updatesPerCoarseStepLocal += fluidCellCount;
                fluidVolumeLocal += fluidCellCount;
            }
            for (auto* block : mixedBlocks)
            {
                const double fluidCellCount = double(fluidCellIndexList.getVector(*block).size());
                updatesPerCoarseStepLocal += fluidCellCount;
                fluidVolumeLocal += fluidCellCount;
            }
        }

        const double updatesPerCoarseStep = walberla::mpi::allReduce(updatesPerCoarseStepLocal, walberla::mpi::SUM);
        const double fluidVolume = walberla::mpi::allReduce(fluidVolumeLocal, walberla::mpi::SUM);

        loop.addFuncAfterTimeStep([&, updatesPerCoarseStep, fluidVolume,
                                      energyInit = false,
                                      massInit = false,
                                      lastTime = std::chrono::steady_clock::now(),
                                      lastStep = uint_t(0),
                                      lastEnergy = 0.0,
                                      lastMass = 0.0]() mutable {
            const uint_t step = loop.getCurrentTimeStep();
            if (!cadenceDue(step, minimalLogCadence, false))
                return;
            double uMaxSqLocal = 0.0;
            double uySqVolumeLocal = 0.0;
            double uSqVolumeLocal = 0.0;
            double energyLocal = 0.0;
            double massLocal = 0.0;
            bool minimalReduceOk = true;
            minimalReduceOk = gpureduce::reduceMinimalLocal(
                fullFluidBlocks,
                mixedBlocks,
                fluidCellIndexList,
                velocityRuntimeID,
                thetaRuntimeID,
                densityRuntimeID,
                1.0,
                uMaxSqLocal,
                uySqVolumeLocal,
                uSqVolumeLocal,
                energyLocal,
                massLocal);

            if (!minimalReduceOk)
            {
                WALBERLA_ABORT("Non-finite rho/velocity detected by GPU minimal diagnostics reduction at step "
                               << step << ". Aborting (fail-fast).");
            }

            const double uMaxSq = walberla::mpi::allReduce(uMaxSqLocal, walberla::mpi::MAX);
            const double uySqVolume = walberla::mpi::allReduce(uySqVolumeLocal, walberla::mpi::SUM);
            const double uSqVolume = walberla::mpi::allReduce(uSqVolumeLocal, walberla::mpi::SUM);
            const double energy = walberla::mpi::allReduce(energyLocal, walberla::mpi::SUM);
            const double mass = walberla::mpi::allReduce(massLocal, walberla::mpi::SUM);
            const double uMax = std::sqrt(uMaxSq);
            const double maMax = uMax * std::sqrt(3.0);
            const double uySqMean = (fluidVolume > 0.0) ? (uySqVolume / fluidVolume) : 0.0;
            const double uSqMean = (fluidVolume > 0.0) ? (uSqVolume / fluidVolume) : 0.0;
            const double wrmsY = std::sqrt(std::max(0.0, uySqMean));
            const double ek = 0.5 * uSqMean;
            const auto now = std::chrono::steady_clock::now();
            const double dtWindow = std::chrono::duration<double>(now - lastTime).count();
            const double stepsPerS = (dtWindow > 0.0) ? (double(step - lastStep) / dtWindow) : 0.0;
            const double mlups = updatesPerCoarseStep * stepsPerS / 1.0e6;
            const double dE = energyInit ? (energy - lastEnergy) : 0.0;
            const double dM = massInit ? (mass - lastMass) : 0.0;
            const double rhoMean = (fluidVolume > 0.0) ? (mass / fluidVolume) : 0.0;

            if (!std::isfinite(uMaxSq) || !std::isfinite(uMax) || !std::isfinite(maMax) ||
                !std::isfinite(mlups) || !std::isfinite(energy) || !std::isfinite(mass) ||
                !std::isfinite(dE) || !std::isfinite(dM) || !std::isfinite(rhoMean) ||
                !std::isfinite(wrmsY) || !std::isfinite(ek))
            {
                WALBERLA_ABORT(
                    "Non-finite diagnostics at step " << step
                    << " (uMaxSq=" << uMaxSq
                    << ", uMax=" << uMax
                    << ", MaMax=" << maMax
                    << ", MLUPS=" << mlups
                    << ", E=" << energy
                    << ", M=" << mass
                    << ", dE=" << dE
                    << ", dM=" << dM
                    << ", rhoMean=" << rhoMean
                    << ", wrms_y=" << wrmsY
                    << ", Ek=" << ek
                    << "). Aborting (fail-fast).");
            }

            energyInit = true;
            massInit = true;

            if (isRoot)
            {
                WALBERLA_LOG_INFO("MINIMAL t=" << step
                                 << " MLUPS=" << mlups
                                 << " MaMax=" << maMax
                                 << " dE=" << dE
                                 << " M=" << mass
                                 << " dM=" << dM
                                 << " rhoMean=" << rhoMean
                                 << " wrms_y=" << wrmsY
                                 << " Ek=" << ek);
            }
            lastEnergy = energy;
            lastMass = mass;
            lastTime = now;
            lastStep = step;
        }, "MinimalLogger");
    }

    // Thermal diagnostics logger.
    if (thermalLogCadence > uint_t(0))
    {
        std::vector<std::string> nuOutputLabels;
        nuOutputLabels.reserve(nuOutputRegions.size());
        for (const auto& region : nuOutputRegions)
            nuOutputLabels.push_back(nuOutputLabelFromRegionName(region.regionName));

        auto nuGpuCache = std::make_shared<gpureduce::NuBoundaryCacheCuda>();
        auto warnedNuZeroArea = std::make_shared<std::vector<walberla::uint8_t>>(nuOutputRegions.size(), walberla::uint8_t(0));
        auto warnedNuZeroDeltaTheta = std::make_shared<std::vector<walberla::uint8_t>>(nuOutputRegions.size(), walberla::uint8_t(0));
        std::vector<double> reduceLocal(nuOutputRegions.size() * size_t(2), 0.0);
        std::vector<double> reduceGlobal(nuOutputRegions.size() * size_t(2), 0.0);
        std::vector<double> localFluxArea(nuOutputRegions.size(), 0.0);
        std::vector<double> localArea(nuOutputRegions.size(), 0.0);
        std::vector<double> nuByRegion(nuOutputRegions.size(), std::numeric_limits<double>::quiet_NaN());
        loop.addFuncAfterTimeStep(
            [&, nuOutputSlotByRegionId, nuOutputLabels, nuGpuCache, warnedNuZeroArea, warnedNuZeroDeltaTheta,
             reduceLocal = std::move(reduceLocal), reduceGlobal = std::move(reduceGlobal),
             localFluxArea = std::move(localFluxArea), localArea = std::move(localArea),
             nuByRegion = std::move(nuByRegion)]() mutable {
            const uint_t step = loop.getCurrentTimeStep();
            if (!cadenceDue(step, thermalLogCadence, false))
                return;

            if (!nuOutputRegions.empty())
            {
                std::fill(reduceLocal.begin(), reduceLocal.end(), 0.0);
                std::fill(reduceGlobal.begin(), reduceGlobal.end(), 0.0);
                gpureduce::reduceNuLocal(
                    thermalBCBlocks,
                    thetaRuntimeID,
                    nuOutputSlotByRegionId,
                    1.0,
                    1.0,
                    *nuGpuCache,
                    localFluxArea,
                    localArea);

                for (size_t i = size_t(0); i < nuOutputRegions.size(); ++i)
                {
                    reduceLocal[size_t(2) * i + size_t(0)] = localFluxArea[i];
                    reduceLocal[size_t(2) * i + size_t(1)] = localArea[i];
                }
                WALBERLA_MPI_SECTION()
                {
                    MPI_Allreduce(
                        reduceLocal.data(),
                        reduceGlobal.data(),
                        int(reduceLocal.size()),
                        walberla::MPITrait<double>::type(),
                        walberla::MPITrait<double>::operation(walberla::mpi::SUM),
                        walberla::mpi::MPIManager::instance()->comm());
                }

                if (isRoot)
                {
                    for (size_t i = size_t(0); i < nuOutputRegions.size(); ++i)
                    {
                        const double area = reduceGlobal[size_t(2) * i + size_t(1)];
                        if (area <= 0.0 && (*warnedNuZeroArea)[i] == walberla::uint8_t(0))
                        {
                            (*warnedNuZeroArea)[i] = walberla::uint8_t(1);
                            WALBERLA_LOG_WARNING(
                                "Nu requested for region '" << nuOutputRegions[i].regionName
                                << "' but contributing boundary face area is 0. "
                                << "Check mesh coloring / BC mapping / thermal boundary cache. "
                                << "Nu will remain NaN for this region.");
                        }
                    }
                }

                nuByRegion.assign(nuOutputRegions.size(), std::numeric_limits<double>::quiet_NaN());
                for (size_t i = size_t(0); i < nuOutputRegions.size(); ++i)
                {
                    const double dTheta = nuOutputRegions[i].hasDeltaThetaOverride
                                            ? nuOutputRegions[i].deltaThetaOverride
                                            : 1.0;
                    if (std::abs(dTheta) <= 1e-15)
                    {
                        if (isRoot && (*warnedNuZeroDeltaTheta)[i] == walberla::uint8_t(0))
                        {
                            (*warnedNuZeroDeltaTheta)[i] = walberla::uint8_t(1);
                            WALBERLA_LOG_WARNING(
                                "Thermal Nu disabled for region '" << nuOutputRegions[i].regionName
                                << "': DeltaTheta is zero (source="
                                << (nuOutputRegions[i].hasDeltaThetaOverride ? "Nu_dTheta" : "default_1")
                                << "). Nu will be NaN for this region.");
                        }
                        continue;
                    }
                    const double fluxArea = reduceGlobal[size_t(2) * i + size_t(0)];
                    const double area = reduceGlobal[size_t(2) * i + size_t(1)];
                    if (area > 0.0)
                        nuByRegion[i] = (fluxArea / area) * nuOutputRegions[i].lCharLatFine / dTheta;
                }
            }

            if (isRoot)
            {
                std::ostringstream oss;
                oss << std::setprecision(9);
                oss << "THERMAL t=" << step
                    << " theta_ref=" << double(currentThetaRef);
                for (size_t i = size_t(0); i < nuOutputRegions.size(); ++i)
                    oss << " Nu_" << nuOutputLabels[i] << "=" << nuByRegion[i];
                WALBERLA_LOG_INFO(oss.str());
            }
        }, "ThermalLogger");
    }

    // Checkpoint writer hook.
    if (checkpointCadence > uint_t(0))
    {
        loop.addFuncAfterTimeStep([&]() {
            const uint_t step = loop.getCurrentTimeStep();
            if (!cadenceDue(step, checkpointCadence, false))
                return;
            ensureHostState(step, HostSyncScope::Full);
            writeCheckpointState(step, pdfID, densityID, velocityID, thetaID);
        }, "CheckpointWriter");
    }

    // VTK output hooks.
    if (vtkWriteFrequency > uint_t(0) || vtkWriteAtStepZero)
    {
        // Pre-compute constant Nu scale factors (all inputs are binding-time constants).
        std::unordered_map<walberla::uint16_t, size_t> nuVtkFieldSlotByRegionId;
        nuVtkFieldSlotByRegionId.reserve(nuVtkFields.size());
        for (size_t idx = size_t(0); idx < nuVtkFields.size(); ++idx)
            nuVtkFieldSlotByRegionId.emplace(nuVtkFields[idx].regionId, idx);

        std::vector<double> nuScaleBySlot(nuVtkFields.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<real_t> nuValueResetBySlot(nuVtkFields.size(), std::numeric_limits<real_t>::quiet_NaN());
        bool anyValidNuScale = false;
        {
            for (size_t slot = size_t(0); slot < nuVtkFields.size(); ++slot)
            {
                const auto infoIt = nuOutputSlotByRegionId.find(nuVtkFields[slot].regionId);
                if (infoIt == nuOutputSlotByRegionId.end())
                    continue;
                const auto& nuInfo = nuOutputRegions[infoIt->second];
                const double dTheta = nuInfo.hasDeltaThetaOverride ? nuInfo.deltaThetaOverride : 1.0;
                if (std::abs(dTheta) <= 1e-15)
                    continue;
                nuScaleBySlot[slot] = nuInfo.lCharLatFine / dTheta;
                nuValueResetBySlot[slot] = real_t(0);
                anyValidNuScale = true;
            }
        }

        // Pre-compute per-block lists of fluid cells with Nu-relevant Dirichlet wall neighbors.
        struct NuVtkWallNeighbor
        {
            size_t slot;
            real_t thermalValue;
        };
        struct NuVtkFluidCell
        {
            int x, y, z;
            std::vector<NuVtkWallNeighbor> walls;
        };
        struct NuVtkBlockRefs
        {
            walberla::IBlock* block = nullptr;
            std::vector<NuVtkFluidCell> cells;
        };
        std::vector<NuVtkBlockRefs> nuVtkBlocks;
        if (anyValidNuScale && !nuVtkFields.empty())
        {
            for (auto& block : *blocks)
            {
                const auto& boundaryFluidIndices = boundaryFluidIndexList.getVector(block);
                if (boundaryFluidIndices.empty())
                    continue;
                auto* cellType = block.getData<CellTypeField>(cellTypeID);
                auto* bcId = block.getData<BcField>(bcIdID);
                auto* thermalType = block.getData<ThermalTypeField>(thermalTypeID);
                auto* thermalValueF = block.getData<ScalarField>(thermalValueID);
                auto* regionId = block.getData<RegionIdField>(regionIdID);
                NuVtkBlockRefs nuBlock;
                nuBlock.block = &block;
                for (const auto& idx : boundaryFluidIndices)
                {
                    const int x = int(idx.x);
                    const int y = int(idx.y);
                    const int z = int(idx.z);
                    NuVtkFluidCell fluidCell;
                    fluidCell.x = x;
                    fluidCell.y = y;
                    fluidCell.z = z;
                    for (const auto& d : kFaceNbrDirs)
                    {
                        const int sx = x + d[0];
                        const int sy = y + d[1];
                        const int sz = z + d[2];
                        if ((*cellType)(sx, sy, sz, 0) != CELL_SOLID)
                            continue;
                        if ((*bcId)(sx, sy, sz, 0) != BC_DIRICHLET)
                            continue;
                        if ((*thermalType)(sx, sy, sz, 0) != THERMAL_DIRICHLET)
                            continue;
                        const auto slotIt = nuVtkFieldSlotByRegionId.find((*regionId)(sx, sy, sz, 0));
                        if (slotIt == nuVtkFieldSlotByRegionId.end())
                            continue;
                        const size_t slot = slotIt->second;
                        if (!std::isfinite(nuScaleBySlot[slot]))
                            continue;
                        fluidCell.walls.push_back(NuVtkWallNeighbor{slot, (*thermalValueF)(sx, sy, sz, 0)});
                    }
                    if (!fluidCell.walls.empty())
                        nuBlock.cells.push_back(std::move(fluidCell));
                }
                if (!nuBlock.cells.empty())
                    nuVtkBlocks.push_back(std::move(nuBlock));
            }
        }

        auto vtkStepDue = [&](uint_t step) -> bool { return cadenceDue(step, vtkWriteFrequency, vtkWriteAtStepZero); };

        auto updateNuFieldsForVtk = [&,
             nuScaleBySlot = std::move(nuScaleBySlot),
             nuValueResetBySlot = std::move(nuValueResetBySlot),
             nuVtkBlocks = std::move(nuVtkBlocks)]() {
            if (nuVtkFields.empty())
                return;

            // Reset pass: set all boundary fluid cells' Nu fields to NaN/0.
            for (auto& block : *blocks)
            {
                const auto& boundaryFluidIndices = boundaryFluidIndexList.getVector(block);
                if (boundaryFluidIndices.empty())
                    continue;
                std::vector<ScalarField*> nuValueFieldPtrs(nuVtkFields.size(), nullptr);
                std::vector<ScalarField*> nuCountFieldPtrs(nuVtkFields.size(), nullptr);
                for (size_t slot = size_t(0); slot < nuVtkFields.size(); ++slot)
                {
                    nuValueFieldPtrs[slot] = block.getData<ScalarField>(nuVtkFields[slot].valueFieldID);
                    nuCountFieldPtrs[slot] = block.getData<ScalarField>(nuVtkFields[slot].countFieldID);
                }
                for (const auto& idx : boundaryFluidIndices)
                {
                    const int x = int(idx.x);
                    const int y = int(idx.y);
                    const int z = int(idx.z);
                    for (size_t slot = size_t(0); slot < nuVtkFields.size(); ++slot)
                    {
                        (*nuValueFieldPtrs[slot])(x, y, z, 0) = nuValueResetBySlot[slot];
                        (*nuCountFieldPtrs[slot])(x, y, z, 0) = real_t(0);
                    }
                }
            }

            // Compute pass: iterate only pre-computed Nu-relevant fluid cells.
            for (const auto& nuBlock : nuVtkBlocks)
            {
                std::vector<ScalarField*> nuValueFieldPtrs(nuVtkFields.size(), nullptr);
                std::vector<ScalarField*> nuCountFieldPtrs(nuVtkFields.size(), nullptr);
                for (size_t slot = size_t(0); slot < nuVtkFields.size(); ++slot)
                {
                    nuValueFieldPtrs[slot] = nuBlock.block->getData<ScalarField>(nuVtkFields[slot].valueFieldID);
                    nuCountFieldPtrs[slot] = nuBlock.block->getData<ScalarField>(nuVtkFields[slot].countFieldID);
                }
                auto* theta = nuBlock.block->getData<ScalarField>(thetaID);
                for (const auto& cell : nuBlock.cells)
                {
                    const double thetaFluid = double((*theta)(cell.x, cell.y, cell.z, 0));
                    for (const auto& wall : cell.walls)
                    {
                        (*nuValueFieldPtrs[wall.slot])(cell.x, cell.y, cell.z, 0) +=
                            real_t(nuFacePrimitive(double(wall.thermalValue), thetaFluid, 1.0) * nuScaleBySlot[wall.slot]);
                        (*nuCountFieldPtrs[wall.slot])(cell.x, cell.y, cell.z, 0) += real_t(1);
                    }
                    for (size_t slot = size_t(0); slot < nuVtkFields.size(); ++slot)
                    {
                        const real_t count = (*nuCountFieldPtrs[slot])(cell.x, cell.y, cell.z, 0);
                        if (count > real_t(0))
                            (*nuValueFieldPtrs[slot])(cell.x, cell.y, cell.z, 0) /= count;
                    }
                }
            }
        };
        WALBERLA_ROOT_SECTION()
        {
            ensureDirectory(std::filesystem::path(outputBaseDir) / kVtkDirName, "create vtk output directory");
        }
        WALBERLA_MPI_SECTION()
        {
            MPI_Barrier(walberla::mpi::MPIManager::instance()->comm());
        }
        auto vtkOutput = walberla::vtk::createVTKOutput_BlockData(
            *blocks,
            kVtkDirName,
            uint_t(1),
            uint_t(0),
            true,
            outputBaseDir,
            "simulation_step",
            false,
            true,
            true,
            false,
            uint_t(0),
            false,
            false);
        vtkOutput->addCellDataWriter(
            std::make_shared<walberla::field::VTKWriter<ScalarField, walberla::float32>>(densityID, "density"));
        vtkOutput->addCellDataWriter(
            std::make_shared<walberla::field::VTKWriter<VecField, walberla::float32>>(velocityID, "velocity"));
        vtkOutput->addCellDataWriter(
            std::make_shared<walberla::field::VTKWriter<ScalarField, walberla::float32>>(thetaID, "theta"));
        vtkOutput->addCellDataWriter(std::make_shared<walberla::field::VTKWriter<CellTypeField>>(cellTypeID, "cellType"));
        vtkOutput->addCellDataWriter(std::make_shared<walberla::field::VTKWriter<BcField>>(bcIdID, "bcId"));
        for (const auto& nuField : nuVtkFields)
            vtkOutput->addCellDataWriter(
                std::make_shared<walberla::field::VTKWriter<ScalarField, walberla::float32>>(
                    nuField.valueFieldID,
                    "Nu_" + nuOutputLabelFromRegionName(nuField.regionName)));
        loop.addFuncAfterTimeStep([&, vtkStepDue, updateNuFieldsForVtk]() {
            const uint_t step = loop.getCurrentTimeStep();
            if (!vtkStepDue(step))
                return;
            // VTK uses density/velocity/theta + derived Nu fields; syncing PDFs here is unnecessary overhead.
            ensureHostState(step, HostSyncScope::RhoVelTheta);
            updateNuFieldsForVtk();
        }, "NuFieldForVTK");
        loop.addFuncAfterTimeStep([&, vtkStepDue, vtkOutput]() {
            const uint_t step = loop.getCurrentTimeStep();
            if (!vtkStepDue(step))
                return;
            // Write VTK artifacts with the physical simulation step as file numbering.
            vtkOutput->forceWrite(step);
            WALBERLA_MPI_SECTION()
            {
                MPI_Barrier(walberla::mpi::MPIManager::instance()->comm());
            }
        }, "VTK");
    }

    // Emit setup marker then enter main loop.
    if (isRoot)
        WALBERLA_LOG_INFO("SETUP");

    loop.run();

    // Rescale VTK time metadata AFTER the loop finishes. waLBerla's PVD writer
    // caches a byte offset (pvdEnd_) for seek-based appending; rewriting the
    // file mid-run can invalidate that offset and corrupt later appends.
    WALBERLA_ROOT_SECTION()
    {
        appsupport::rescaleVtkTimeMetadata(outputBaseDir, kVtkDirName, dtPhysFine);
    }

    cleanupCheckpointAuxDirs(checkpointPaths.forestFile.parent_path(), "cleanup checkpoint auxiliary directory at shutdown");
    return 0;
}

} // namespace fluidsim
