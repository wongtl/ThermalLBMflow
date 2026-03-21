#include "core/all.h"
#include "blockforest/all.h"
#include "field/all.h"
#include "stencil/all.h"

#include "blockforest/communication/UniformBufferedScheme.h"
#include "field/FileIO.h"
#include "mesh/boundary/BoundaryInfo.h"
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh_common/MatrixVectorOperations.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"
#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "walberla/experimental/sweep/SparseIndexList.hpp"

#include "gen/FluidSim.hpp"
#include "gen/ThetaUpdate.hpp"

#include "../../shared/helpers/MeshHelpers.hpp"

#ifdef FLUIDSIM_GPU_BUILD
#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/ErrorChecking.h"
#include "gpu/FieldCopy.h"
#include "gpu/GPUField.h"
#include "gpu/communication/MemcpyPackInfo.h"
#include "gpu/communication/UniformGPUScheme.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using walberla::real_t;
using walberla::uint_t;

#include "src/FluidSimInternal.hpp"
#include "src/GpuReductions.hpp"
#include "src/FluidSimRuntime.hpp"
#include "src/FluidSimSetup.hpp"

namespace fluidsim
{

int runFluidSimSetupAndRuntime(int argc, char** argv)
{
    // Command-line and environment setup.
    walberla::Environment env(argc, argv);
    walberla::mpi::MPIManager::instance()->useWorldComm();
    CmdOptions cmd = parseArgs(argc, argv);
    const bool isRoot = walberla::mpi::MPIManager::instance()->worldRank() == 0;

#ifdef FLUIDSIM_GPU_BUILD
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
#endif

    const std::filesystem::path appRootDir = resolveAppRootPath(argc, argv);

    // GPU-only rank/device selection and one-rank-per-GPU validation.
#ifdef FLUIDSIM_GPU_BUILD
    {
        const bool enforceOneRankPerGpu = true;
        auto abortOnMpiFailure = [&](const char* callName, int rc) {
            if (rc == MPI_SUCCESS)
                return;

            char errString[MPI_MAX_ERROR_STRING] = {};
            int errLen = 0;
            const int errRc = MPI_Error_string(rc, errString, &errLen);
            if (errRc == MPI_SUCCESS && errLen > 0)
            {
                WALBERLA_ABORT("MPI call failed in GPU rank/device setup: " << callName
                               << " rc=" << rc << " err='" << std::string(errString, size_t(errLen)) << "'");
            }
            WALBERLA_ABORT("MPI call failed in GPU rank/device setup: " << callName << " rc=" << rc);
        };

        MPI_Comm nodeComm = MPI_COMM_NULL;
        abortOnMpiFailure(
            "MPI_Comm_split_type",
            MPI_Comm_split_type(
                walberla::mpi::MPIManager::instance()->comm(),
                MPI_COMM_TYPE_SHARED,
                0,
                MPI_INFO_NULL,
                &nodeComm));
        auto freeNodeComm = [&]() {
            if (nodeComm != MPI_COMM_NULL)
                abortOnMpiFailure("MPI_Comm_free(nodeComm)", MPI_Comm_free(&nodeComm));
        };

        int localRank = 0;
        int localSize = 1;
        abortOnMpiFailure("MPI_Comm_rank(nodeComm)", MPI_Comm_rank(nodeComm, &localRank));
        abortOnMpiFailure("MPI_Comm_size(nodeComm)", MPI_Comm_size(nodeComm, &localSize));
        int worldSize = 1;
        abortOnMpiFailure(
            "MPI_Comm_size(world)",
            MPI_Comm_size(walberla::mpi::MPIManager::instance()->comm(), &worldSize));
        if (localSize != worldSize)
        {
            freeNodeComm();
            WALBERLA_ABORT("Multi-node run detected (unsupported): worldSize=" << worldSize
                           << " localSize=" << localSize << ". Policy: single-node only.");
        }

        int visibleGpuCount = 0;
        WALBERLA_GPU_CHECK(gpuGetDeviceCount(&visibleGpuCount));
        if (visibleGpuCount <= 0)
        {
            freeNodeComm();
            WALBERLA_ABORT("No visible GPU device found for local rank "
                           << localRank << " on node. Check Slurm GRES/GPU allocation and launcher settings.");
        }

        const int selectedDevice = localRank % visibleGpuCount;
        gpuDeviceProp selectedDeviceProp{};
        WALBERLA_GPU_CHECK(gpuGetDeviceProperties(&selectedDeviceProp, selectedDevice));
        WALBERLA_GPU_CHECK(gpuSetDevice(selectedDevice));

        WALBERLA_LOG_INFO("GPU device selection: worldRank=" << walberla::mpi::MPIManager::instance()->worldRank()
                         << " localRank=" << localRank << "/" << localSize
                         << " visibleGpuCount=" << visibleGpuCount
                         << " selectedDevice=" << selectedDevice
                         << " policy=one-rank-per-gpu");

        if (enforceOneRankPerGpu && localSize > 1)
        {
            constexpr int kUuidBytes = int(sizeof(selectedDeviceProp.uuid));
            std::vector<unsigned char> selectedGpuUuids(
                size_t(localSize) * size_t(kUuidBytes),
                static_cast<unsigned char>(0));
            const auto* selectedUuidBytes = reinterpret_cast<const unsigned char*>(&selectedDeviceProp.uuid);
            abortOnMpiFailure(
                "MPI_Allgather(selected GPU UUIDs)",
                MPI_Allgather(
                    selectedUuidBytes,
                    kUuidBytes,
                    MPI_BYTE,
                    selectedGpuUuids.data(),
                    kUuidBytes,
                    MPI_BYTE,
                    nodeComm));

            bool duplicateDetected = false;
            for (int i = 0; i < localSize && !duplicateDetected; ++i)
            {
                const auto* uuidI = selectedGpuUuids.data() + size_t(i) * size_t(kUuidBytes);
                for (int j = i + 1; j < localSize; ++j)
                {
                    const auto* uuidJ = selectedGpuUuids.data() + size_t(j) * size_t(kUuidBytes);
                    if (std::memcmp(uuidI, uuidJ, size_t(kUuidBytes)) == 0)
                    {
                        duplicateDetected = true;
                        break;
                    }
                }
            }
            if (duplicateDetected)
            {
                freeNodeComm();
                WALBERLA_ABORT(
                    "Duplicate physical GPU UUID detected on local node. One-rank-per-GPU policy is enforced."
                    " localSize=" << localSize << ", visibleGpuCount=" << visibleGpuCount << ".");
            }
            else if (visibleGpuCount == 1 && localRank == 0)
            {
                WALBERLA_LOG_INFO(
                    "One-rank-per-GPU policy is enforced and each rank sees one visible GPU device."
                    " Physical UUID uniqueness was validated across local ranks.");
            }
        }

        freeNodeComm();
    }
#endif

    // Configuration blocks and core physical/numerical parameters.
    auto config = env.config();
    walberla::Config::BlockHandle simParams;
    if (config->getNumBlocks("Parameters") > 0)
        simParams = config->getBlock("Parameters");
    else
        WALBERLA_ABORT("FluidSim requires a Parameters block (theta_ref, theta_update).");
    walberla::Config::BlockHandle physicalParams;
    if (config->getNumBlocks("Physical") > 0)
        physicalParams = config->getBlock("Physical");
    else
        WALBERLA_ABORT("FluidSim requires a Physical block (nu_phys, alpha_phys, beta, g, domain_size, etc.).");
    walberla::Config::BlockHandle numericsParams;
    if (config->getNumBlocks("Numerics") > 0)
        numericsParams = config->getBlock("Numerics");
    else
        WALBERLA_ABORT("FluidSim requires a Numerics block (nu_lat_target).");
    walberla::Config::BlockHandle resolutionParams;
    if (config->getNumBlocks("Resolution") > 0)
        resolutionParams = config->getBlock("Resolution");
    else
        WALBERLA_ABORT("FluidSim requires a Resolution block with interiorFineCells and cellsPerBlock.");

    const uint_t numTimesteps = cmd.timesteps;
    const real_t thetaRef0 = simParams.getParameter<real_t>("theta_ref", real_t(0.0));
    const uint_t thetaUpdateEvery = simParams.getParameter<uint_t>("theta_update", uint_t(1));
    const uint_t vtkWriteFrequency = cmd.vtkEvery;

    const double nuPhys = physicalParams.getParameter<double>("nu_phys");
    const double alphaPhys = physicalParams.getParameter<double>("alpha_phys");
    const double kPhys = physicalParams.getParameter<double>("k_phys");
    const double beta = physicalParams.getParameter<double>("beta");
    const double g = physicalParams.getParameter<double>("g");
    const walberla::Vector3<double> domainSizePhys = physicalParams.getParameter<walberla::Vector3<double>>(
        "domain_size", walberla::Vector3<double>(0.0, 0.0, 0.0));
    if (physicalParams.isDefined("L_char"))
    {
        WALBERLA_ABORT("Physical.L_char is not supported. Use Physical.fluid_height instead.");
    }
    const double fluidHeightInput = physicalParams.getParameter<double>("fluid_height");
    const double deltaTK = physicalParams.getParameter<double>("deltaT_K", 1.0);
    const double raFactor = physicalParams.getParameter<double>("Ra_factor", 1.0);
    const double nuLatTargetFine = numericsParams.getParameter<double>("nu_lat_target", 0.01);

    walberla::Vector3<uint_t> interiorFineCells = resolutionParams.getParameter<walberla::Vector3<uint_t>>(
        "interiorFineCells", walberla::Vector3<uint_t>(uint_t(0), uint_t(0), uint_t(0)));
    walberla::Vector3<uint_t> cellsPerBlock = resolutionParams.getParameter<walberla::Vector3<uint_t>>(
        "cellsPerBlock", walberla::Vector3<uint_t>(uint_t(32), uint_t(32), uint_t(32)));

    constexpr uint_t ghostLayers = uint_t(1);

    if (config->getNumBlocks("MeshGeometry") == 0)
        WALBERLA_ABORT("FluidSim is mesh-only. Add a MeshGeometry block to the parameter file.");
    const auto meshParams = config->getBlock("MeshGeometry");
    const walberla::Vector3<uint_t> paddingFineCells = meshParams.getParameter<walberla::Vector3<uint_t>>(
        "paddingCells", walberla::Vector3<uint_t>(uint_t(2), uint_t(2), uint_t(2)));

    if (domainSizePhys[0] <= 0.0 || domainSizePhys[1] <= 0.0 || domainSizePhys[2] <= 0.0)
        WALBERLA_ABORT("Physical.domain_size entries must be > 0.");
    if (interiorFineCells[0] == uint_t(0) || interiorFineCells[1] == uint_t(0) || interiorFineCells[2] == uint_t(0))
        WALBERLA_ABORT("Resolution.interiorFineCells entries must be > 0.");
    if (cellsPerBlock[0] == uint_t(0) || cellsPerBlock[1] == uint_t(0) || cellsPerBlock[2] == uint_t(0))
        WALBERLA_ABORT("Resolution.cellsPerBlock entries must be > 0.");
    if (!std::isfinite(nuPhys) || nuPhys <= 0.0)
        WALBERLA_ABORT("Physical.nu_phys must be finite and > 0.");
    if (!std::isfinite(alphaPhys) || alphaPhys <= 0.0)
        WALBERLA_ABORT("Physical.alpha_phys must be finite and > 0.");
    if (!std::isfinite(kPhys) || kPhys <= 0.0)
        WALBERLA_ABORT("Physical.k_phys must be finite and > 0.");
    if (!std::isfinite(beta))
        WALBERLA_ABORT("Physical.beta must be finite.");
    if (!std::isfinite(deltaTK))
        WALBERLA_ABORT("Physical.deltaT_K must be finite.");
    if (!std::isfinite(g))
        WALBERLA_ABORT("Physical.g must be finite.");
    if (nuLatTargetFine <= 0.0)
        WALBERLA_ABORT("Numerics.nu_lat_target must be > 0.");
    if (fluidHeightInput <= 0.0) WALBERLA_ABORT("Physical.fluid_height must be > 0.");
    if (raFactor < 0.0) WALBERLA_ABORT("Physical.Ra_factor must be >= 0.");

    const auto fineDomainCells = walberla::Vector3<uint_t>(
        interiorFineCells[0] + uint_t(2) * paddingFineCells[0],
        interiorFineCells[1] + uint_t(2) * paddingFineCells[1],
        interiorFineCells[2] + uint_t(2) * paddingFineCells[2]);

    const walberla::Vector3<double> interiorCellCounts{
        double(interiorFineCells[0]),
        double(interiorFineCells[1]),
        double(interiorFineCells[2])};
    const walberla::Vector3<double> fullSizePhys(
        domainSizePhys[0] * double(fineDomainCells[0]) / interiorCellCounts[0],
        domainSizePhys[1] * double(fineDomainCells[1]) / interiorCellCounts[1],
        domainSizePhys[2] * double(fineDomainCells[2]) / interiorCellCounts[2]);
    const walberla::Vector3<double> paddingSizePhys(
        fullSizePhys[0] - domainSizePhys[0],
        fullSizePhys[1] - domainSizePhys[1],
        fullSizePhys[2] - domainSizePhys[2]);
    if (fullSizePhys[0] <= 0.0 || fullSizePhys[1] <= 0.0 || fullSizePhys[2] <= 0.0)
        WALBERLA_ABORT("Computed non-positive Physical.full_size from domain_size/interiorFineCells/paddingCells.");

    const double dxX = fullSizePhys[0] / double(fineDomainCells[0]);
    const double dxY = fullSizePhys[1] / double(fineDomainCells[1]);
    const double dxZ = fullSizePhys[2] / double(fineDomainCells[2]);
    const double dxMin = std::min({dxX, dxY, dxZ});
    const double dxMax = std::max({dxX, dxY, dxZ});
    constexpr double dxTol = 1e-12;
    // In the nominal path these are equal by construction, but keep this as a
    // fail-fast guard for malformed inputs and floating-point drift.
    if (dxMin <= 0.0 || ((dxMax / dxMin) - 1.0) > dxTol)
    {
        WALBERLA_ABORT("Resolution.interiorFineCells + MeshGeometry.paddingCells do not match Physical.full_size isotropically."
                       << " dx=<" << dxX << "," << dxY << "," << dxZ << ">"
                       << " ratio_error=" << ((dxMin > 0.0) ? ((dxMax / dxMin) - 1.0) : std::numeric_limits<double>::infinity())
                       << " tol=" << dxTol);
    }

    const double dxPhysFine = dxX;
    const double fluidHeightPhys = fluidHeightInput;
    const double dtPhysFine = nuLatTargetFine * dxPhysFine * dxPhysFine / nuPhys;
    const double raBase = g * beta * deltaTK * (fluidHeightPhys * fluidHeightPhys * fluidHeightPhys) / (nuPhys * alphaPhys);
    const double alphaLatFine = alphaPhys * dtPhysFine / (dxPhysFine * dxPhysFine);
    if (alphaLatFine <= 0.0)
        WALBERLA_ABORT("Computed non-positive alpha_lat on finest level.");
    const double aLatFine = raFactor * g * beta * deltaTK * (dtPhysFine * dtPhysFine) / dxPhysFine;

    // MeshGeometry parsing.
    MeshGeometryConfig meshCfg;
    std::vector<GeometryRegionConfig> geometryRegionConfigs;

    meshCfg.checkpointFolder = stripQuotes(meshParams.getParameter<std::string>("checkpointFolder", ""));
    meshCfg.scale = meshParams.getParameter<real_t>("scale", real_t(1));
    meshCfg.translateFraction = meshParams.getParameter<walberla::Vector3<real_t>>(
        "translate", walberla::Vector3<real_t>(real_t(0)));
    walberla::Config::Blocks meshRegionBlocks;
    meshParams.getBlocks("Region", meshRegionBlocks, 0);
    if (!meshRegionBlocks.empty())
    {
        bool hasContainer = false;
        for (const auto& rb : meshRegionBlocks)
        {
            GeometryRegionConfig regionCfg;
            regionCfg.name = stripQuotes(rb.getParameter<std::string>("name", ""));
            if (regionCfg.name.empty())
                WALBERLA_ABORT("MeshGeometry.Region.name must be set.");
            regionCfg.meshFile = stripQuotes(rb.getParameter<std::string>("meshFile", ""));
            if (regionCfg.meshFile.empty())
                WALBERLA_ABORT("MeshGeometry.Region.meshFile must be set.");
            regionCfg.role = geometryRoleFromString(rb.getParameter<std::string>("role", "FLUID_CONTAINER"));
            regionCfg.scale = rb.getParameter<real_t>("scale", meshCfg.scale);
            regionCfg.translateFraction = rb.getParameter<walberla::Vector3<real_t>>("translate", meshCfg.translateFraction);
            if (regionCfg.role == GeometryRole::FluidContainer)
                hasContainer = true;
            geometryRegionConfigs.push_back(regionCfg);
        }
        if (!hasContainer)
            WALBERLA_ABORT("MeshGeometry.Region requires at least one role FLUID_CONTAINER.");
    }
    else
    {
        WALBERLA_ABORT("MeshGeometry requires at least one Region block.");
    }
    if (!meshCfg.checkpointFolder.empty())
    {
        const std::filesystem::path checkpointFolderPath(meshCfg.checkpointFolder);
        if (checkpointFolderPath.is_absolute() || checkpointFolderPath.has_parent_path())
            WALBERLA_ABORT("MeshGeometry.checkpointFolder must be a folder name without path separators."
                           << " Got '" << meshCfg.checkpointFolder << "'.");
    }
    if (config->getNumBlocks("DomainSetup") == 0)
        WALBERLA_ABORT("FluidSim requires a DomainSetup block with periodic settings.");
    const auto domainParams = config->getBlock("DomainSetup");
    const walberla::Vector3<bool> periodicFlags =
        domainParams.getParameter<walberla::Vector3<bool>>("periodic", walberla::Vector3<bool>(false));

    struct GeometryRegionRuntime
    {
        std::string name;
        std::filesystem::path sourcePath;
        std::uintmax_t sourceFileBytes = std::uintmax_t(0);
        std::string sourceFileHash;
        std::filesystem::path loadPath;
        GeometryRole role = GeometryRole::SolidObstacle;
        real_t scale = real_t(1);
        walberla::Vector3<real_t> translateFraction = walberla::Vector3<real_t>(real_t(0));
        std::shared_ptr<walberla::mesh::TriangleMesh> mesh;
        std::shared_ptr<walberla::mesh::TriangleDistance<walberla::mesh::TriangleMesh>> triDist;
        std::shared_ptr<walberla::mesh::DistanceOctree<walberla::mesh::TriangleMesh>> distanceOctree;
        std::shared_ptr<walberla::mesh::BoundaryLocation<walberla::mesh::TriangleMesh>> boundaryLocations;
    };
    std::vector<GeometryRegionRuntime> geometryRegions;
    geometryRegions.reserve(geometryRegionConfigs.size());
    for (const auto& regionCfg : geometryRegionConfigs)
    {
        GeometryRegionRuntime region;
        region.name = regionCfg.name;
        region.sourcePath = resolveMeshPath(regionCfg.meshFile, argc, argv);
        region.role = regionCfg.role;
        region.scale = regionCfg.scale;
        region.translateFraction = regionCfg.translateFraction;

        bool sourceExists = false;
        std::uint64_t sourceBytes = std::uint64_t(0);
        std::string sourceHash;
        if (isRoot)
        {
            sourceExists = std::filesystem::exists(region.sourcePath);
            if (sourceExists)
            {
                const auto sourceFingerprint = computeFileFingerprint(region.sourcePath);
                if (sourceFingerprint.bytes > std::uintmax_t(std::numeric_limits<std::uint64_t>::max()))
                    WALBERLA_ABORT("Mesh file too large for checkpoint fingerprint metadata: " << region.sourcePath.string());
                sourceBytes = std::uint64_t(sourceFingerprint.bytes);
                sourceHash = sourceFingerprint.hashHex;
            }
        }
        WALBERLA_MPI_SECTION()
        {
            int sourceExistsFlag = (isRoot && sourceExists) ? 1 : 0;
            MPI_Bcast(
                &sourceExistsFlag,
                1,
                walberla::MPITrait<int>::type(),
                0,
                walberla::mpi::MPIManager::instance()->comm());
            sourceExists = (sourceExistsFlag != 0);

            MPI_Bcast(
                &sourceBytes,
                1,
                walberla::MPITrait<std::uint64_t>::type(),
                0,
                walberla::mpi::MPIManager::instance()->comm());

            std::uint64_t hashLen = isRoot ? std::uint64_t(sourceHash.size()) : std::uint64_t(0);
            MPI_Bcast(
                &hashLen,
                1,
                walberla::MPITrait<std::uint64_t>::type(),
                0,
                walberla::mpi::MPIManager::instance()->comm());
            if (!isRoot)
                sourceHash.resize(size_t(hashLen));
            if (hashLen > std::uint64_t(0))
            {
                MPI_Bcast(
                    sourceHash.data(),
                    int(hashLen),
                    walberla::MPITrait<char>::type(),
                    0,
                    walberla::mpi::MPIManager::instance()->comm());
            }
        }
        if (!sourceExists)
            WALBERLA_ABORT("MeshGeometry region mesh file does not exist: " << region.sourcePath.string());
        region.sourceFileBytes = std::uintmax_t(sourceBytes);
        region.sourceFileHash = sourceHash;
        geometryRegions.push_back(std::move(region));
    }
    for (const auto& region : geometryRegions)
    {
        const auto& t = region.translateFraction;
        if (t[0] < real_t(-0.5) || t[0] > real_t(0.5) ||
            t[1] < real_t(-0.5) || t[1] > real_t(0.5) ||
            t[2] < real_t(-0.5) || t[2] > real_t(0.5))
        {
            WALBERLA_ABORT("MeshGeometry translate fraction for region '" << region.name
                           << "' must be within [-0.5, 0.5] in each axis."
                           << " Got <" << t[0] << "," << t[1] << "," << t[2] << ">.");
        }
    }
    const std::filesystem::path checkpointArchiveDir = appRootDir.parent_path() / "storage";
    const std::filesystem::path checkpointOutputDir = std::filesystem::path(kOutputBaseDir) / kCheckpointDirName;
    const std::string checkpointPrefix = (checkpointOutputDir / kCheckpointStatePrefix).string();
    std::string restartPrefix;
    if (!meshCfg.checkpointFolder.empty())
        restartPrefix = (checkpointArchiveDir / meshCfg.checkpointFolder / kCheckpointDirName / kCheckpointStatePrefix).string();
    const bool restartEnabled = !restartPrefix.empty();
    const CheckpointPaths checkpointPaths = makeCheckpointPaths(checkpointPrefix);
    const CheckpointPaths restartPaths = restartEnabled ? makeCheckpointPaths(restartPrefix) : CheckpointPaths{};

    WALBERLA_ROOT_SECTION()
    {
        if (restartEnabled)
        {
            std::error_code ec;
            const bool exists = std::filesystem::exists(checkpointArchiveDir, ec);
            if (ec || !exists || !std::filesystem::is_directory(checkpointArchiveDir, ec) || ec)
            {
                WALBERLA_ABORT("Restart archive root does not exist or is not a directory: "
                               << checkpointArchiveDir.string()
                               << ". Expected restart input under "
                               << (checkpointArchiveDir / meshCfg.checkpointFolder / kCheckpointDirName).string());
            }
        }
    }
    WALBERLA_MPI_SECTION()
    {
        MPI_Barrier(walberla::mpi::MPIManager::instance()->comm());
    }

    for (auto& region : geometryRegions)
    {
        const auto meshCompatDir = compatMeshDirectoryForSource(region.sourcePath);
        bool meshWasPreConverted = false;
        std::string meshPathToLoadStr = region.sourcePath.string();
        if (isRoot)
        {
            const auto convertedPath = maybeCreateOpenMeshCompatiblePly(region.sourcePath, meshCompatDir, meshWasPreConverted);
            meshPathToLoadStr = convertedPath.string();
        }
        WALBERLA_MPI_SECTION()
        {
            int convertedFlag = meshWasPreConverted ? 1 : 0;
            MPI_Bcast(
                &convertedFlag,
                1,
                walberla::MPITrait<int>::type(),
                0,
                walberla::mpi::MPIManager::instance()->comm());
            meshWasPreConverted = (convertedFlag != 0);

            std::uint64_t pathLen = isRoot ? std::uint64_t(meshPathToLoadStr.size()) : std::uint64_t(0);
            MPI_Bcast(
                &pathLen,
                1,
                walberla::MPITrait<std::uint64_t>::type(),
                0,
                walberla::mpi::MPIManager::instance()->comm());
            if (!isRoot)
                meshPathToLoadStr.resize(size_t(pathLen));
            if (pathLen > std::uint64_t(0))
            {
                MPI_Bcast(
                    meshPathToLoadStr.data(),
                    int(pathLen),
                    walberla::MPITrait<char>::type(),
                    0,
                    walberla::mpi::MPIManager::instance()->comm());
            }
        }
        region.loadPath = std::filesystem::path(meshPathToLoadStr);
        if (isRoot && meshWasPreConverted)
        {
            WALBERLA_LOG_INFO_ON_ROOT(
                "MESH pre-conversion applied for OpenMesh compatibility: "
                << region.sourcePath.string() << " -> " << region.loadPath.string());
        }

        bool plySawHeaderEnd = false;
        bool plyHasVertexRgb = false;
        bool plyHasFaceRgb = false;
        if (isRoot)
        {
            const PlyColorInfo plyColorInfo = scanPlyColorInfo(region.loadPath);
            plySawHeaderEnd = plyColorInfo.sawHeaderEnd;
            plyHasVertexRgb = plyColorInfo.hasVertexRgb();
            plyHasFaceRgb = plyColorInfo.hasFaceRgb();
        }
        std::array<int, 3> plyColorFlags{
            plySawHeaderEnd ? 1 : 0,
            plyHasVertexRgb ? 1 : 0,
            plyHasFaceRgb ? 1 : 0};
        WALBERLA_MPI_SECTION()
        {
            MPI_Bcast(
                plyColorFlags.data(),
                3,
                walberla::MPITrait<int>::type(),
                0,
                walberla::mpi::MPIManager::instance()->comm());
        }
        plySawHeaderEnd = (plyColorFlags[0] != 0);
        plyHasVertexRgb = (plyColorFlags[1] != 0);
        plyHasFaceRgb = (plyColorFlags[2] != 0);
        if (!plySawHeaderEnd || (!plyHasVertexRgb && !plyHasFaceRgb))
            WALBERLA_ABORT("Mesh file must contain RGB properties on vertex or face elements in the PLY header: "
                           << region.loadPath.string());

        region.mesh = std::make_shared<walberla::mesh::TriangleMesh>();
        if (plyHasVertexRgb) region.mesh->request_vertex_colors();
        if (plyHasFaceRgb) region.mesh->request_face_colors();
        appsupport::readMeshFromRootAndBroadcast(region.loadPath, *region.mesh);
        constexpr real_t unitScaleTol = real_t(1e-12);
        if (std::abs(region.scale - real_t(1)) > unitScaleTol)
            walberla::mesh::scale(*region.mesh, walberla::Vector3<real_t>(region.scale));
        if (!region.mesh->has_face_colors())
        {
            if (!region.mesh->has_vertex_colors())
                WALBERLA_ABORT("Mesh has no face or vertex colors. Colored mesh is required.");
            vertexToFaceColor(*region.mesh, walberla::mesh::TriangleMesh::Color(255, 255, 255));
        }
    }

    walberla::math::AABB meshAabbUnscaled = computeMeshAabb(*geometryRegions.front().mesh);
    for (size_t i = size_t(1); i < geometryRegions.size(); ++i)
    {
        const auto aabb = computeMeshAabb(*geometryRegions[i].mesh);
        meshAabbUnscaled = walberla::math::AABB(
            std::min(meshAabbUnscaled.xMin(), aabb.xMin()),
            std::min(meshAabbUnscaled.yMin(), aabb.yMin()),
            std::min(meshAabbUnscaled.zMin(), aabb.zMin()),
            std::max(meshAabbUnscaled.xMax(), aabb.xMax()),
            std::max(meshAabbUnscaled.yMax(), aabb.yMax()),
            std::max(meshAabbUnscaled.zMax(), aabb.zMax()));
    }
    if (meshAabbUnscaled.xSize() <= real_t(0) || meshAabbUnscaled.ySize() <= real_t(0) ||
        meshAabbUnscaled.zSize() <= real_t(0))
        WALBERLA_ABORT("Mesh has non-positive extent in at least one direction.");

    if ((fineDomainCells[0] % cellsPerBlock[0]) != uint_t(0) ||
        (fineDomainCells[1] % cellsPerBlock[1]) != uint_t(0) ||
        (fineDomainCells[2] % cellsPerBlock[2]) != uint_t(0))
    {
        WALBERLA_ABORT("Resolution.totalFineCells (derived from interiorFineCells + paddingCells*2) must be divisible by Resolution.cellsPerBlock."
                       << " totalFineCells=<" << fineDomainCells[0] << "," << fineDomainCells[1] << "," << fineDomainCells[2] << ">"
                       << " cellsPerBlock=<" << cellsPerBlock[0] << "," << cellsPerBlock[1] << "," << cellsPerBlock[2] << ">");
    }
    const auto rootBlockCounts = walberla::Vector3<uint_t>(
        fineDomainCells[0] / cellsPerBlock[0],
        fineDomainCells[1] / cellsPerBlock[1],
        fineDomainCells[2] / cellsPerBlock[2]);
    if (rootBlockCounts[0] == uint_t(0) || rootBlockCounts[1] == uint_t(0) || rootBlockCounts[2] == uint_t(0))
        WALBERLA_ABORT("Derived rootBlocks entries must be > 0.");

    const real_t physToLatticeScale = real_t(1.0) / real_t(dxPhysFine);
    if (physToLatticeScale <= real_t(0))
        WALBERLA_ABORT("Computed non-positive mesh scale; check Physical.full_size and Resolution.interiorFineCells.");
    for (auto& region : geometryRegions)
        walberla::mesh::scale(*region.mesh, walberla::Vector3<real_t>(physToLatticeScale));

    const walberla::math::AABB domainAabb(
        real_t(0), real_t(0), real_t(0),
        real_t(fineDomainCells[0]), real_t(fineDomainCells[1]), real_t(fineDomainCells[2]));
    real_t requiredInteriorX = real_t(0);
    real_t requiredInteriorY = real_t(0);
    real_t requiredInteriorZ = real_t(0);
    for (const auto& region : geometryRegions)
    {
        const auto aabb = computeMeshAabb(*region.mesh);
        requiredInteriorX = std::max(requiredInteriorX, aabb.xSize());
        requiredInteriorY = std::max(requiredInteriorY, aabb.ySize());
        requiredInteriorZ = std::max(requiredInteriorZ, aabb.zSize());
    }
    constexpr real_t fitTol = real_t(1e-6);
    if (requiredInteriorX > real_t(interiorFineCells[0]) + fitTol ||
        requiredInteriorY > real_t(interiorFineCells[1]) + fitTol ||
        requiredInteriorZ > real_t(interiorFineCells[2]) + fitTol)
    {
        WALBERLA_ABORT("One or more mesh regions do not fit in interior domain with configured padding."
                       << " requiredInteriorFine=<" << requiredInteriorX << "," << requiredInteriorY << "," << requiredInteriorZ << ">"
                       << " interiorFine=<" << interiorFineCells[0] << "," << interiorFineCells[1] << "," << interiorFineCells[2] << ">"
                       << " totalFineCells=<" << fineDomainCells[0] << "," << fineDomainCells[1] << "," << fineDomainCells[2] << ">"
                       << " paddingCells=<" << paddingFineCells[0] << "," << paddingFineCells[1] << "," << paddingFineCells[2] << ">");
    }
    const walberla::math::AABB interiorAabb(
        domainAabb.xMin() + real_t(paddingFineCells[0]),
        domainAabb.yMin() + real_t(paddingFineCells[1]),
        domainAabb.zMin() + real_t(paddingFineCells[2]),
        domainAabb.xMax() - real_t(paddingFineCells[0]),
        domainAabb.yMax() - real_t(paddingFineCells[1]),
        domainAabb.zMax() - real_t(paddingFineCells[2]));
    const walberla::Vector3<real_t> domainCenter(
        interiorAabb.xMin() + real_t(0.5) * interiorAabb.xSize(),
        interiorAabb.yMin() + real_t(0.5) * interiorAabb.ySize(),
        interiorAabb.zMin() + real_t(0.5) * interiorAabb.zSize());
    const walberla::Vector3<real_t> interiorSize(
        interiorAabb.xSize(), interiorAabb.ySize(), interiorAabb.zSize());
    constexpr real_t placementTol = real_t(1e-6);
    std::vector<walberla::math::AABB> regionAabbs(geometryRegions.size());
    for (size_t regionIdx = size_t(0); regionIdx < geometryRegions.size(); ++regionIdx)
    {
        auto& region = geometryRegions[regionIdx];
        const auto preShiftAabb = computeMeshAabb(*region.mesh);
        const walberla::Vector3<real_t> regionCenter(
            preShiftAabb.xMin() + real_t(0.5) * preShiftAabb.xSize(),
            preShiftAabb.yMin() + real_t(0.5) * preShiftAabb.ySize(),
            preShiftAabb.zMin() + real_t(0.5) * preShiftAabb.zSize());
        const walberla::Vector3<real_t> regionOffset(
            region.translateFraction[0] * interiorSize[0],
            region.translateFraction[1] * interiorSize[1],
            region.translateFraction[2] * interiorSize[2]);
        walberla::mesh::translate(*region.mesh, domainCenter - regionCenter + regionOffset);

        const auto shiftedAabb = computeMeshAabb(*region.mesh);
        regionAabbs[regionIdx] = shiftedAabb;
        if (shiftedAabb.xMin() < interiorAabb.xMin() - placementTol ||
            shiftedAabb.yMin() < interiorAabb.yMin() - placementTol ||
            shiftedAabb.zMin() < interiorAabb.zMin() - placementTol ||
            shiftedAabb.xMax() > interiorAabb.xMax() + placementTol ||
            shiftedAabb.yMax() > interiorAabb.yMax() + placementTol ||
            shiftedAabb.zMax() > interiorAabb.zMax() + placementTol)
        {
            WALBERLA_ABORT("MeshGeometry.Region '" << region.name
                           << "' is outside interior domain after centering and translate offset."
                           << " translate=<" << region.translateFraction[0] << ","
                           << region.translateFraction[1] << "," << region.translateFraction[2] << ">"
                           << " shiftedAabb=[" << shiftedAabb.xMin() << "," << shiftedAabb.xMax() << "]x["
                           << shiftedAabb.yMin() << "," << shiftedAabb.yMax() << "]x["
                           << shiftedAabb.zMin() << "," << shiftedAabb.zMax() << "]"
                           << " interior=[" << interiorAabb.xMin() << "," << interiorAabb.xMax() << "]x["
                           << interiorAabb.yMin() << "," << interiorAabb.yMax() << "]x["
                           << interiorAabb.zMin() << "," << interiorAabb.zMax() << "]");
        }
    }

    if (restartEnabled)
    {
        if (isRoot)
        {
            if (!std::filesystem::exists(restartPaths.forestFile) ||
                !std::filesystem::exists(restartPaths.pdfFile) ||
                !std::filesystem::exists(restartPaths.densityFile) ||
                !std::filesystem::exists(restartPaths.velocityFile) ||
                !std::filesystem::exists(restartPaths.thetaFile) ||
                !std::filesystem::exists(restartPaths.metaFile))
            {
                WALBERLA_ABORT("Restart requested but one or more checkpoint files are missing for prefix '"
                               << restartPrefix << "'");
            }

            const auto meta = readCheckpointMetadata(restartPaths.metaFile);
            auto getMeta = [&](const std::string& key) -> std::string {
                const auto it = meta.find(key);
                if (it == meta.end())
                    WALBERLA_ABORT("Checkpoint metadata is missing key '" << key << "': " << restartPaths.metaFile.string());
                return it->second;
            };
            const std::string checkpointFormat = toLower(getMeta("checkpoint_format"));
            if (checkpointFormat != "field_io_v1")
                WALBERLA_ABORT("Unsupported checkpoint_format '" << checkpointFormat
                               << "'. Expected field_io_v1.");
            auto hasOnlyTrailingWhitespace = [](const std::string& value, size_t parsedChars) {
                for (size_t i = parsedChars; i < value.size(); ++i)
                {
                    if (!std::isspace(static_cast<unsigned char>(value[i])))
                        return false;
                }
                return true;
            };
            // WALBERLA_ABORT terminates via std::exit, so these parse helpers do not need fallback returns.
            auto parseUIntMeta = [&](const std::string& key) -> uint_t {
                const std::string value = getMeta(key);
                try
                {
                    size_t parsedChars = size_t(0);
                    const auto parsedValue = std::stoull(value, &parsedChars);
                    if (!hasOnlyTrailingWhitespace(value, parsedChars))
                    {
                        WALBERLA_ABORT("Invalid checkpoint metadata: key='" << key
                                       << "' value='" << value << "' file='" << restartPaths.metaFile.string()
                                       << "' err='trailing characters after integer'");
                    }
                    const auto uintMax = static_cast<unsigned long long>(std::numeric_limits<uint_t>::max());
                    if (parsedValue > uintMax)
                    {
                        WALBERLA_ABORT("Invalid checkpoint metadata: key='" << key
                                       << "' value='" << value << "' file='" << restartPaths.metaFile.string()
                                       << "' err='integer out of range for uint_t'");
                    }
                    return static_cast<uint_t>(parsedValue);
                }
                catch (const std::exception& e)
                {
                    WALBERLA_ABORT("Invalid checkpoint metadata: key='" << key
                                   << "' value='" << value << "' file='" << restartPaths.metaFile.string()
                                   << "' err='" << e.what() << "'");
                }
            };
            auto parseDoubleMeta = [&](const std::string& key, const std::string& value) -> double {
                try
                {
                    size_t parsedChars = size_t(0);
                    const auto parsedValue = std::stod(value, &parsedChars);
                    if (!hasOnlyTrailingWhitespace(value, parsedChars))
                    {
                        WALBERLA_ABORT("Invalid checkpoint metadata: key='" << key
                                       << "' value='" << value << "' file='" << restartPaths.metaFile.string()
                                       << "' err='trailing characters after floating-point value'");
                    }
                    return parsedValue;
                }
                catch (const std::exception& e)
                {
                    WALBERLA_ABORT("Invalid checkpoint metadata: key='" << key
                                   << "' value='" << value << "' file='" << restartPaths.metaFile.string()
                                   << "' err='" << e.what() << "'");
                }
            };
            constexpr double kRestartMetaRelTol = 1e-5;
            constexpr double kRestartMetaAbsTol = 1e-5;
            auto nearlyEqual = [kRestartMetaRelTol, kRestartMetaAbsTol](double a, double b) {
                const double scale = std::max(std::abs(a), std::abs(b));
                const double tol = std::max(kRestartMetaAbsTol, kRestartMetaRelTol * scale);
                return std::abs(a - b) <= tol;
            };
            auto splitByPipe = [](const std::string& value) {
                std::vector<std::string> parts;
                std::stringstream ss(value);
                std::string item;
                while (std::getline(ss, item, '|'))
                    parts.push_back(item);
                return parts;
            };

            const uint_t metaRegionCount = parseUIntMeta("region_count");
            if (metaRegionCount != uint_t(geometryRegions.size()))
                WALBERLA_ABORT("Restart region_count mismatch.");

            struct MetaRegionInfo
            {
                std::string role;
                std::uintmax_t bytes = std::uintmax_t(0);
                std::string hash;
                double scale = 1.0;
                walberla::Vector3<double> translate = walberla::Vector3<double>(0.0, 0.0, 0.0);
            };
            std::unordered_map<std::string, MetaRegionInfo> metaRegionsByName;
            metaRegionsByName.reserve(size_t(metaRegionCount));
            for (size_t regionIdx = size_t(0); regionIdx < size_t(metaRegionCount); ++regionIdx)
            {
                const std::string key = std::string("region_") + std::to_string(regionIdx);
                const auto parts = splitByPipe(getMeta(key));
                if (parts.size() != size_t(6))
                    WALBERLA_ABORT("Restart metadata format mismatch for key '" << key
                                   << "'. Expected name|role|bytes|hash|scale|translate.");
                const std::string& regionName = parts[0];
                if (metaRegionsByName.find(regionName) != metaRegionsByName.end())
                    WALBERLA_ABORT("Restart metadata contains duplicate region name '" << regionName << "'.");

                MetaRegionInfo info;
                info.role = parts[1];
                try
                {
                    size_t parsedChars = size_t(0);
                    const auto parsedBytes = std::stoull(parts[2], &parsedChars);
                    if (!hasOnlyTrailingWhitespace(parts[2], parsedChars))
                    {
                        WALBERLA_ABORT("Invalid checkpoint metadata: key='" << key
                                       << "' field='bytes' value='" << parts[2] << "' file='"
                                       << restartPaths.metaFile.string()
                                       << "' err='trailing characters after integer'");
                    }
                    const std::uintmax_t bytesAsUintmax = static_cast<std::uintmax_t>(parsedBytes);
                    if (static_cast<unsigned long long>(bytesAsUintmax) != parsedBytes)
                    {
                        WALBERLA_ABORT("Invalid checkpoint metadata: key='" << key
                                       << "' field='bytes' value='" << parts[2] << "' file='"
                                       << restartPaths.metaFile.string()
                                       << "' err='integer out of range for uintmax_t'");
                    }
                    info.bytes = bytesAsUintmax;
                }
                catch (const std::exception& e)
                {
                    WALBERLA_ABORT("Invalid checkpoint metadata: key='" << key
                                   << "' field='bytes' value='" << parts[2] << "' file='"
                                   << restartPaths.metaFile.string() << "' err='" << e.what() << "'");
                }
                info.hash = parts[3];
                info.scale = parseDoubleMeta(key + ".scale", parts[4]);
                info.translate = parseVec3Csv<double>(parts[5]);
                metaRegionsByName.emplace(regionName, std::move(info));
            }

            for (const auto& region : geometryRegions)
            {
                const auto it = metaRegionsByName.find(region.name);
                if (it == metaRegionsByName.end())
                    WALBERLA_ABORT("Restart metadata missing region '" << region.name << "'.");
                const auto& metaRegion = it->second;

                const std::string currentRole = (region.role == GeometryRole::FluidContainer) ? "FLUID_CONTAINER" : "SOLID_OBSTACLE";
                if (metaRegion.role != currentRole)
                    WALBERLA_ABORT("Restart region role mismatch for '" << region.name << "'.");

                if (metaRegion.bytes != region.sourceFileBytes)
                    WALBERLA_ABORT("Restart region mesh byte-size mismatch for '" << region.name << "'.");
                if (toLower(metaRegion.hash) != toLower(region.sourceFileHash))
                    WALBERLA_ABORT("Restart region mesh content hash mismatch for '" << region.name << "'.");

                if (!nearlyEqual(metaRegion.scale, double(region.scale)))
                    WALBERLA_ABORT("Restart region scale mismatch for '" << region.name << "'.");
                if (!nearlyEqual(metaRegion.translate[0], double(region.translateFraction[0])) ||
                    !nearlyEqual(metaRegion.translate[1], double(region.translateFraction[1])) ||
                    !nearlyEqual(metaRegion.translate[2], double(region.translateFraction[2])))
                {
                    WALBERLA_ABORT("Restart region translate mismatch for '" << region.name << "'.");
                }
            }

            const auto metaDomainSize = parseVec3Csv<double>(getMeta("domain_size"));
            if (!nearlyEqual(metaDomainSize[0], domainSizePhys[0]) ||
                !nearlyEqual(metaDomainSize[1], domainSizePhys[1]) ||
                !nearlyEqual(metaDomainSize[2], domainSizePhys[2]))
                WALBERLA_ABORT("Restart Physical.domain_size mismatch.");
            const auto metaPaddingSize = parseVec3Csv<double>(getMeta("padding_size"));
            if (!nearlyEqual(metaPaddingSize[0], paddingSizePhys[0]) ||
                !nearlyEqual(metaPaddingSize[1], paddingSizePhys[1]) ||
                !nearlyEqual(metaPaddingSize[2], paddingSizePhys[2]))
                WALBERLA_ABORT("Restart Physical.padding_size mismatch.");
            const auto metaFullSize = parseVec3Csv<double>(getMeta("full_size"));
            if (!nearlyEqual(metaFullSize[0], fullSizePhys[0]) ||
                !nearlyEqual(metaFullSize[1], fullSizePhys[1]) ||
                !nearlyEqual(metaFullSize[2], fullSizePhys[2]))
                WALBERLA_ABORT("Restart Physical.full_size mismatch.");
            if (parseVec3Csv<uint_t>(getMeta("interior_fine_cells")) != interiorFineCells)
                WALBERLA_ABORT("Restart Resolution.interiorFineCells mismatch.");
            if (parseVec3Csv<uint_t>(getMeta("total_fine_cells")) != fineDomainCells)
                WALBERLA_ABORT("Restart Resolution.totalFineCells mismatch.");
            if (parseVec3Csv<uint_t>(getMeta("cells_per_block")) != cellsPerBlock)
                WALBERLA_ABORT("Restart Resolution.cellsPerBlock mismatch.");
            if (parseVec3Csv<uint_t>(getMeta("padding_cells")) != paddingFineCells)
                WALBERLA_ABORT("Restart MeshGeometry.paddingCells mismatch.");
            const auto periodicMeta = parseVec3Csv<int>(getMeta("periodic"));
            if (periodicMeta != walberla::Vector3<int>(periodicFlags[0] ? 1 : 0, periodicFlags[1] ? 1 : 0, periodicFlags[2] ? 1 : 0))
                WALBERLA_ABORT("Restart DomainSetup.periodic mismatch.");

        }
    }

    for (auto& region : geometryRegions)
    {
        region.triDist = std::make_shared<walberla::mesh::TriangleDistance<walberla::mesh::TriangleMesh>>(region.mesh);
        region.distanceOctree = std::make_shared<walberla::mesh::DistanceOctree<walberla::mesh::TriangleMesh>>(region.triDist);
    }
    auto pointInsideAabb = [](const walberla::math::AABB& aabb, const walberla::Vector3<real_t>& p) {
        return p[0] >= aabb.xMin() && p[0] <= aabb.xMax() &&
               p[1] >= aabb.yMin() && p[1] <= aabb.yMax() &&
               p[2] >= aabb.zMin() && p[2] <= aabb.zMax();
    };
    auto isFluidAtPoint = [&geometryRegions, &regionAabbs, pointInsideAabb](const walberla::Vector3<real_t>& p) -> bool {
        const auto pOpenMesh = walberla::mesh::toOpenMesh(p);

        bool insideContainer = false;
        for (size_t regionIdx = size_t(0); regionIdx < geometryRegions.size(); ++regionIdx)
        {
            const auto& region = geometryRegions[regionIdx];
            if (region.role != GeometryRole::FluidContainer)
                continue;
            if (!pointInsideAabb(regionAabbs[regionIdx], p))
                continue;
            const real_t signedDistance = real_t(region.distanceOctree->sqSignedDistance(pOpenMesh));
            if (signedDistance < real_t(0))
            {
                insideContainer = true;
                break;
            }
        }
        if (!insideContainer)
            return false;

        for (size_t regionIdx = size_t(0); regionIdx < geometryRegions.size(); ++regionIdx)
        {
            const auto& region = geometryRegions[regionIdx];
            if (region.role != GeometryRole::SolidObstacle)
                continue;
            if (!pointInsideAabb(regionAabbs[regionIdx], p))
                continue;
            const real_t signedDistance = real_t(region.distanceOctree->sqSignedDistance(pOpenMesh));
            if (signedDistance < real_t(0))
                return false;
        }
        return true;
    };

    // Block-forest setup/load and structured block storage construction.
    std::shared_ptr<walberla::BlockForest> blockForest;
    if (!restartEnabled)
    {
        walberla::blockforest::SetupBlockForest setupForest;
        auto isSolidOnlyBlock = [&geometryRegions, &isFluidAtPoint](const walberla::math::AABB& aabb) {
            constexpr real_t maxError = real_t(0.5);
            const std::array<walberla::Vector3<real_t>, 9> samples = {
                walberla::Vector3<real_t>(aabb.xMin(), aabb.yMin(), aabb.zMin()),
                walberla::Vector3<real_t>(aabb.xMax(), aabb.yMin(), aabb.zMin()),
                walberla::Vector3<real_t>(aabb.xMin(), aabb.yMax(), aabb.zMin()),
                walberla::Vector3<real_t>(aabb.xMax(), aabb.yMax(), aabb.zMin()),
                walberla::Vector3<real_t>(aabb.xMin(), aabb.yMin(), aabb.zMax()),
                walberla::Vector3<real_t>(aabb.xMax(), aabb.yMin(), aabb.zMax()),
                walberla::Vector3<real_t>(aabb.xMin(), aabb.yMax(), aabb.zMax()),
                walberla::Vector3<real_t>(aabb.xMax(), aabb.yMax(), aabb.zMax()),
                aabb.center()};
            for (const auto& p : samples)
                if (isFluidAtPoint(p))
                    return false;

            for (const auto& region : geometryRegions)
            {
                const bool intersects = walberla::mesh::isIntersecting(*region.distanceOctree, aabb, maxError).value_or(true);
                if (intersects)
                    return false;
            }
            return true;
        };

        auto excludeSolidRootBlocks = [isSolidOnlyBlock](
                                          std::vector<uint8_t>& excludeBlock,
                                          const walberla::blockforest::SetupBlockForest::RootBlockAABB& rootBlockAabb) {
            for (size_t idx = size_t(0); idx < excludeBlock.size(); ++idx)
            {
                const auto rootAabb = rootBlockAabb(uint_t(idx));
                if (isSolidOnlyBlock(rootAabb))
                    excludeBlock[idx] = uint8_t(1);
            }
        };
        auto excludeSolidChildBlock = [isSolidOnlyBlock](const walberla::blockforest::SetupBlock& block) {
            return isSolidOnlyBlock(block.getAABB());
        };
        setupForest.addRootBlockExclusionFunction(excludeSolidRootBlocks);
        setupForest.addBlockExclusionFunction(excludeSolidChildBlock);

        auto amrWorkloadMemoryAssignment = [](walberla::blockforest::SetupBlockForest& forest) {
            std::vector<walberla::blockforest::SetupBlock*> setupBlocks;
            forest.getBlocks(setupBlocks);
            for (auto* setupBlock : setupBlocks)
            {
                setupBlock->setWorkload(walberla::blockforest::workload_c(uint_t(1)));
                setupBlock->setMemory(walberla::blockforest::memory_c(1));
            }
        };
        setupForest.addWorkloadMemorySUIDAssignmentFunction(amrWorkloadMemoryAssignment);
        setupForest.init(
            domainAabb,
            rootBlockCounts[0], rootBlockCounts[1], rootBlockCounts[2],
            periodicFlags[0], periodicFlags[1], periodicFlags[2]);
        setupForest.balanceLoad(
            walberla::blockforest::StaticLevelwiseCurveBalanceWeighted(),
            uint_t(walberla::mpi::MPIManager::instance()->numProcesses()));
        blockForest = std::make_shared<walberla::BlockForest>(
            uint_t(walberla::mpi::MPIManager::instance()->worldRank()),
            setupForest);
    }
    else
    {
        blockForest = std::make_shared<walberla::BlockForest>(
            uint_t(walberla::mpi::MPIManager::instance()->worldRank()),
            restartPaths.forestFile.string().c_str(),
            true);
    }
    auto blocks = std::make_shared<walberla::StructuredBlockForest>(
        blockForest, cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2]);
    blocks->createCellBoundingBoxes();
    const std::uint64_t totalBlocksGlobal = walberla::mpi::allReduce(std::uint64_t(blocks->getNumberOfBlocks()), walberla::mpi::SUM);

    const uint_t levels = blocks->getNumberOfLevels();
    if (levels != uint_t(1))
    {
        WALBERLA_ABORT("FluidSim_gpu supports a single grid level only."
                       << " Received levels=" << levels
                       << ". Use a single-level block layout.");
    }
    if (isRoot)
    {
        WALBERLA_LOG_INFO("STARTUP Ra=" << (raBase * raFactor)
                         << " Pr=" << (nuLatTargetFine / alphaLatFine)
                         << " dx_phys=" << dxPhysFine
                         << " dt_phys=" << dtPhysFine
                         << " fluid_height=" << fluidHeightPhys
                         << " dT=" << deltaTK
                         << " nu_target_lat=" << nuLatTargetFine
                         << " initPerturb=" << cmd.initPerturb
                         << " restart=" << (restartEnabled ? "true" : "false")
                         << " checkpointImport=" << (meshCfg.checkpointFolder.empty() ? "none" : meshCfg.checkpointFolder)
                         << " geometryRegions=" << geometryRegions.size()
                         << " domain_size=<" << domainSizePhys[0] << "," << domainSizePhys[1] << "," << domainSizePhys[2] << ">"
                         << " padding_size=<" << paddingSizePhys[0] << "," << paddingSizePhys[1] << "," << paddingSizePhys[2] << ">"
                         << " full_size=<" << fullSizePhys[0] << "," << fullSizePhys[1] << "," << fullSizePhys[2] << ">"
                         << " interiorFineCells=<" << interiorFineCells[0] << "," << interiorFineCells[1] << "," << interiorFineCells[2] << ">"
                         << " totalFineCells=<" << fineDomainCells[0] << "," << fineDomainCells[1] << "," << fineDomainCells[2] << ">"
                         << " paddingCells=<" << paddingFineCells[0] << "," << paddingFineCells[1] << "," << paddingFineCells[2] << ">"
                         << " levels=" << levels
                         << " cellsPerBlock=<" << cellsPerBlock[0] << "," << cellsPerBlock[1] << "," << cellsPerBlock[2] << ">"
                         << " rootBlocks=<" << rootBlockCounts[0] << "," << rootBlockCounts[1] << "," << rootBlockCounts[2] << ">");
    }

    // Host and GPU runtime field allocation plus host communication setup.
    using LbStencil = walberla::stencil::D3Q19;
    using PdfField = walberla::field::GhostLayerField<real_t, LbStencil::Q>;
    using ScalarField = walberla::field::GhostLayerField<real_t, 1>;
    using VecField = walberla::field::GhostLayerField<real_t, 3>;
    using CellTypeField = walberla::field::GhostLayerField<walberla::uint8_t, 1>;
    using ThermalTypeField = walberla::field::GhostLayerField<walberla::uint8_t, 1>;
    using BcField = walberla::field::GhostLayerField<walberla::uint16_t, 1>;
    using RegionIdField = walberla::field::GhostLayerField<walberla::uint16_t, 1>;
    using CommStencil = walberla::stencil::D3Q19;
    using ThetaTmpCommStencil = walberla::stencil::D3Q7;
    using ScalarCommScheme = walberla::blockforest::communication::UniformBufferedScheme<CommStencil>;

    if (config->getNumBlocks("ColorBC") == 0)
        WALBERLA_ABORT("FluidSim mesh mode requires a ColorBC block.");
    const auto colorParams = config->getBlock("ColorBC");
    walberla::Config::Blocks regionBlocks;
    colorParams.getBlocks("Region", regionBlocks, 0);
    if (regionBlocks.empty())
        WALBERLA_ABORT("ColorBC requires at least one Region block.");
    bool useOpenBoundary = false;
    bool hasEnabledColorRegion = false;
    for (const auto& rb : regionBlocks)
    {
        const bool regionEnabled = rb.getParameter<bool>("enabled", false);
        if (!regionEnabled)
            continue;
        hasEnabledColorRegion = true;
        const std::string regionUid = toUpper(stripQuotes(rb.getParameter<std::string>("name")));
        const auto regionBc = bcIdFromRegionName(regionUid);
        if (regionBc == BC_NONE)
        {
            WALBERLA_ABORT("Unsupported ColorBC.Region name: " << regionUid
                           << ". Expected <PREFIX><positive integer suffix>. Bare names (DIRICHLET) and zero-only suffixes (DIRICHLET0)"
                           << " are invalid.");
        }
        if (regionBc == BC_INLET || regionBc == BC_OUTLET || regionBc == BC_PRESSURE)
        {
            useOpenBoundary = true;
            break;
        }
    }
    if (!hasEnabledColorRegion)
        WALBERLA_ABORT("ColorBC has no enabled Region blocks. Set ColorBC.Region.enabled true for at least one region.");

    walberla::BlockDataID pdfID;
    walberla::BlockDataID densityID;
    walberla::BlockDataID velocityID;
    walberla::BlockDataID thetaID;
    walberla::BlockDataID flowRhoID{};
    walberla::BlockDataID flowThetaID{};
    walberla::BlockDataID flowVelocityID{};

    pdfID = walberla::field::addToStorage<PdfField>(blocks, "pdfs", real_t(0.0), walberla::field::fzyx, ghostLayers);
    densityID = walberla::field::addToStorage<ScalarField>(blocks, "density", real_t(1.0), walberla::field::fzyx, uint_t(1));
    velocityID = walberla::field::addToStorage<VecField>(blocks, "velocity", real_t(0.0), walberla::field::fzyx, uint_t(1));
    thetaID = walberla::field::addToStorage<ScalarField>(blocks, "theta", real_t(0.0), walberla::field::fzyx, ghostLayers);
    if (restartEnabled)
    {
        walberla::field::readFromFile<PdfField>(restartPaths.pdfFile.string(), blocks->getBlockStorage(), pdfID);
        walberla::field::readFromFile<ScalarField>(restartPaths.densityFile.string(), blocks->getBlockStorage(), densityID);
        walberla::field::readFromFile<VecField>(restartPaths.velocityFile.string(), blocks->getBlockStorage(), velocityID);
        walberla::field::readFromFile<ScalarField>(restartPaths.thetaFile.string(), blocks->getBlockStorage(), thetaID);
    }

    const auto thetaTmpID = walberla::field::addToStorage<ScalarField>(blocks, "theta_tmp", real_t(0.0), walberla::field::fzyx, ghostLayers);
    const auto cellTypeID = walberla::field::addToStorage<CellTypeField>(blocks, "cellType", CELL_SOLID, walberla::field::fzyx, uint_t(1));
    const auto thermalTypeID = walberla::field::addToStorage<ThermalTypeField>(blocks, "thermalType", THERMAL_NONE, walberla::field::fzyx, uint_t(1));
    const auto thermalValueID = walberla::field::addToStorage<ScalarField>(blocks, "thermalValue", real_t(0.0), walberla::field::fzyx, uint_t(1));
    const auto bcIdID = walberla::field::addToStorage<BcField>(blocks, "bcId", BC_NONE, walberla::field::fzyx, uint_t(1));
    const auto regionIdID = walberla::field::addToStorage<RegionIdField>(blocks, "regionId", walberla::uint16_t(0), walberla::field::fzyx, uint_t(1));
    if (useOpenBoundary)
    {
        flowRhoID = walberla::field::addToStorage<ScalarField>(blocks, "flowRho", real_t(1.0), walberla::field::fzyx, uint_t(1));
        flowThetaID = walberla::field::addToStorage<ScalarField>(blocks, "flowTheta", real_t(0.0), walberla::field::fzyx, uint_t(1));
        flowVelocityID = walberla::field::addToStorage<VecField>(blocks, "flowVelocity", real_t(0.0), walberla::field::fzyx, uint_t(1));
    }

#ifdef FLUIDSIM_GPU_BUILD
    using RuntimeField = walberla::gpu::GPUField<real_t>;
    using CellTypeRuntimeField = walberla::gpu::GPUField<walberla::uint8_t>;
    using ThermalTypeRuntimeField = walberla::gpu::GPUField<walberla::uint8_t>;
    using BcRuntimeField = walberla::gpu::GPUField<walberla::uint16_t>;
    using RuntimeGpuCommScheme = walberla::gpu::communication::UniformGPUScheme<CommStencil>;
    using RuntimeGpuThetaTmpCommScheme = walberla::gpu::communication::UniformGPUScheme<ThetaTmpCommStencil>;
    using RuntimeGpuPackInfo = walberla::gpu::communication::MemcpyPackInfo<RuntimeField>;
    walberla::BlockDataID flowRhoSimID{};
    walberla::BlockDataID flowThetaSimID{};
    walberla::BlockDataID flowVelocitySimID{};
    // Canonical simulation IDs for the timestep hot path: use GPU fields in FLUIDSIM_GPU_BUILD.
    const auto pdfSimID = walberla::gpu::addGPUFieldToStorage<PdfField>(blocks, pdfID, "pdfs_gpu", true);
    const auto densitySimID = walberla::gpu::addGPUFieldToStorage<ScalarField>(blocks, densityID, "density_gpu", true);
    const auto velocitySimID = walberla::gpu::addGPUFieldToStorage<VecField>(blocks, velocityID, "velocity_gpu", true);
    const auto thetaSimID = walberla::gpu::addGPUFieldToStorage<ScalarField>(blocks, thetaID, "theta_gpu", true);
    const auto thetaTmpSimID = walberla::gpu::addGPUFieldToStorage<ScalarField>(blocks, thetaTmpID, "theta_tmp_gpu", true);
    const auto cellTypeSimID = walberla::gpu::addGPUFieldToStorage<CellTypeField>(blocks, cellTypeID, "cellType_gpu", true);
    const auto thermalTypeSimID = walberla::gpu::addGPUFieldToStorage<ThermalTypeField>(blocks, thermalTypeID, "thermalType_gpu", true);
    const auto thermalValueSimID = walberla::gpu::addGPUFieldToStorage<ScalarField>(blocks, thermalValueID, "thermalValue_gpu", true);
    const auto bcIdSimID = walberla::gpu::addGPUFieldToStorage<BcField>(blocks, bcIdID, "bcId_gpu", true);
    if (useOpenBoundary)
    {
        flowRhoSimID = walberla::gpu::addGPUFieldToStorage<ScalarField>(blocks, flowRhoID, "flowRho_gpu", true);
        flowThetaSimID = walberla::gpu::addGPUFieldToStorage<ScalarField>(blocks, flowThetaID, "flowTheta_gpu", true);
        flowVelocitySimID = walberla::gpu::addGPUFieldToStorage<VecField>(blocks, flowVelocityID, "flowVelocity_gpu", true);
    }

    // Validate GPU-resident hot fields use SoA (fzyx) with contiguous x-stride.
    // This is required for coalesced x-major access in custom CUDA kernels.
    for (auto& block : *blocks)
    {
        auto checkGpuSoA = [](const RuntimeField* f) {
            WALBERLA_CHECK_NOT_NULLPTR(f);
            WALBERLA_CHECK_EQUAL(f->layout(), walberla::gpu::fzyx);
            WALBERLA_CHECK_EQUAL(f->xStride(), walberla::cell_idx_t(1));
        };
        checkGpuSoA(block.getData<RuntimeField>(pdfSimID));
        checkGpuSoA(block.getData<RuntimeField>(densitySimID));
        checkGpuSoA(block.getData<RuntimeField>(velocitySimID));
        checkGpuSoA(block.getData<RuntimeField>(thetaSimID));
        checkGpuSoA(block.getData<RuntimeField>(thetaTmpSimID));
    }
#else
#error "FluidSim_gpu requires FLUIDSIM_GPU_BUILD (CUDA) configuration."
#endif

    for (auto& block : *blocks)
    {
        block.getData<CellTypeField>(cellTypeID)->setWithGhostLayer(CELL_SOLID);
        block.getData<ThermalTypeField>(thermalTypeID)->setWithGhostLayer(THERMAL_NONE);
        block.getData<ScalarField>(thermalValueID)->setWithGhostLayer(real_t(0.0));
        block.getData<BcField>(bcIdID)->setWithGhostLayer(BC_NONE);
        block.getData<RegionIdField>(regionIdID)->setWithGhostLayer(walberla::uint16_t(0));
        if (useOpenBoundary)
        {
            block.getData<ScalarField>(flowRhoID)->setWithGhostLayer(real_t(1.0));
            block.getData<ScalarField>(flowThetaID)->setWithGhostLayer(real_t(0.0));
            block.getData<VecField>(flowVelocityID)->setWithGhostLayer(real_t(0.0));
        }
    }

    ScalarCommScheme scalarComm(blocks, 1101);
    scalarComm.addPackInfo(std::make_shared<walberla::field::communication::PackInfo<ScalarField>>(thetaID));
    ScalarCommScheme cellTypeComm(blocks, 1201);
    cellTypeComm.addPackInfo(std::make_shared<walberla::field::communication::PackInfo<CellTypeField>>(cellTypeID));
    ScalarCommScheme bcIdComm(blocks, 1202);
    bcIdComm.addPackInfo(std::make_shared<walberla::field::communication::PackInfo<BcField>>(bcIdID));
    ScalarCommScheme regionIdComm(blocks, 1208);
    regionIdComm.addPackInfo(std::make_shared<walberla::field::communication::PackInfo<RegionIdField>>(regionIdID));
    ScalarCommScheme thermalTypeComm(blocks, 1203);
    thermalTypeComm.addPackInfo(std::make_shared<walberla::field::communication::PackInfo<ThermalTypeField>>(thermalTypeID));
    ScalarCommScheme thermalValueComm(blocks, 1204);
    thermalValueComm.addPackInfo(std::make_shared<walberla::field::communication::PackInfo<ScalarField>>(thermalValueID));
    std::unique_ptr<ScalarCommScheme> flowRhoComm;
    std::unique_ptr<ScalarCommScheme> flowThetaComm;
    std::unique_ptr<ScalarCommScheme> flowVelocityComm;
    if (useOpenBoundary)
    {
        flowRhoComm = std::make_unique<ScalarCommScheme>(blocks, 1205);
        flowRhoComm->addPackInfo(std::make_shared<walberla::field::communication::PackInfo<ScalarField>>(flowRhoID));
        flowThetaComm = std::make_unique<ScalarCommScheme>(blocks, 1206);
        flowThetaComm->addPackInfo(std::make_shared<walberla::field::communication::PackInfo<ScalarField>>(flowThetaID));
        flowVelocityComm = std::make_unique<ScalarCommScheme>(blocks, 1207);
        flowVelocityComm->addPackInfo(std::make_shared<walberla::field::communication::PackInfo<VecField>>(flowVelocityID));
    }

    auto syncRuntimeToHostPdfThetaRhoVel = [&]() {
        walberla::gpu::fieldCpy<PdfField, RuntimeField>(blocks, pdfID, pdfSimID);
        walberla::gpu::fieldCpy<ScalarField, RuntimeField>(blocks, densityID, densitySimID);
        walberla::gpu::fieldCpy<VecField, RuntimeField>(blocks, velocityID, velocitySimID);
        walberla::gpu::fieldCpy<ScalarField, RuntimeField>(blocks, thetaID, thetaSimID);
        WALBERLA_GPU_CHECK(gpuDeviceSynchronize());
    };
    auto syncRuntimeToHostTheta = [&]() {
        walberla::gpu::fieldCpy<ScalarField, RuntimeField>(blocks, thetaID, thetaSimID);
        WALBERLA_GPU_CHECK(gpuDeviceSynchronize());
    };
    auto syncRuntimeToHostRhoVelTheta = [&]() {
        walberla::gpu::fieldCpy<ScalarField, RuntimeField>(blocks, densityID, densitySimID);
        walberla::gpu::fieldCpy<VecField, RuntimeField>(blocks, velocityID, velocitySimID);
        walberla::gpu::fieldCpy<ScalarField, RuntimeField>(blocks, thetaID, thetaSimID);
        WALBERLA_GPU_CHECK(gpuDeviceSynchronize());
    };
    auto syncHostToRuntimeThetaBuffers = [&]() {
        walberla::gpu::fieldCpy<RuntimeField, ScalarField>(blocks, thetaSimID, thetaID);
        walberla::gpu::fieldCpy<RuntimeField, ScalarField>(blocks, thetaTmpSimID, thetaTmpID);
        WALBERLA_GPU_CHECK(gpuDeviceSynchronize());
    };
    auto syncHostToRuntimeBoundaryFields = [&]() {
        walberla::gpu::fieldCpy<CellTypeRuntimeField, CellTypeField>(blocks, cellTypeSimID, cellTypeID);
        walberla::gpu::fieldCpy<ThermalTypeRuntimeField, ThermalTypeField>(blocks, thermalTypeSimID, thermalTypeID);
        walberla::gpu::fieldCpy<RuntimeField, ScalarField>(blocks, thermalValueSimID, thermalValueID);
        walberla::gpu::fieldCpy<BcRuntimeField, BcField>(blocks, bcIdSimID, bcIdID);
        if (useOpenBoundary)
        {
            walberla::gpu::fieldCpy<RuntimeField, ScalarField>(blocks, flowRhoSimID, flowRhoID);
            walberla::gpu::fieldCpy<RuntimeField, ScalarField>(blocks, flowThetaSimID, flowThetaID);
            walberla::gpu::fieldCpy<RuntimeField, VecField>(blocks, flowVelocitySimID, flowVelocityID);
        }
        WALBERLA_GPU_CHECK(gpuDeviceSynchronize());
    };
    // Mesh-to-grid classification and boundary labeling.
    real_t thetaDirichletMin = real_t(0);
    real_t thetaDirichletMax = real_t(0);
    real_t thetaInit = real_t(0);
    std::vector<ColorRegionConfig> colorRegions;
    constexpr const char* kUnmappedBoundaryUid = "__UNMAPPED_COLOR__";
    bool hasDirichlet = false;
    auto nearlyEqualColorValue = [](real_t a, real_t b) {
        const double ad = double(a);
        const double bd = double(b);
        const double scale = std::max<double>(1.0, std::max(std::abs(ad), std::abs(bd)));
        return std::abs(ad - bd) <= 1e-12 * scale;
    };
    auto equalColorRegion = [&](const ColorRegionConfig& a, const ColorRegionConfig& b) {
        return a.uidName == b.uidName &&
               a.bcId == b.bcId &&
               a.thermalType == b.thermalType &&
               a.pressureFlowMode == b.pressureFlowMode &&
               a.nuOutput == b.nuOutput &&
               a.hasNuDeltaThetaOverride == b.hasNuDeltaThetaOverride &&
               a.r == b.r && a.g == b.g && a.b == b.b &&
               nearlyEqualColorValue(a.theta, b.theta) &&
               nearlyEqualColorValue(a.nuLCharPhys, b.nuLCharPhys) &&
               nearlyEqualColorValue(a.nuDeltaThetaOverride, b.nuDeltaThetaOverride) &&
               nearlyEqualColorValue(a.heatload, b.heatload) &&
               nearlyEqualColorValue(a.flowRho, b.flowRho) &&
               nearlyEqualColorValue(a.flowTheta, b.flowTheta) &&
               nearlyEqualColorValue(a.flowVelocity[0], b.flowVelocity[0]) &&
               nearlyEqualColorValue(a.flowVelocity[1], b.flowVelocity[1]) &&
               nearlyEqualColorValue(a.flowVelocity[2], b.flowVelocity[2]);
    };
    std::unordered_map<std::string, ColorRegionConfig> colorRegionByUid;
    colorRegionByUid.reserve(regionBlocks.size());
    std::unordered_map<std::uint32_t, std::string> uidByRgb;
    uidByRgb.reserve(regionBlocks.size());
    size_t disabledColorRegionCount = size_t(0);
    for (const auto& rb : regionBlocks)
    {
        const bool regionEnabled = rb.getParameter<bool>("enabled", false);
        if (!regionEnabled)
        {
            ++disabledColorRegionCount;
            continue;
        }
        ColorRegionConfig region;
        region.uidName = toUpper(stripQuotes(rb.getParameter<std::string>("name")));
        region.bcId = bcIdFromRegionName(region.uidName);
        if (region.bcId == BC_NONE)
            WALBERLA_ABORT("Unsupported ColorBC.Region name: " << region.uidName
                           << ". Expected <PREFIX><positive integer suffix>. Bare names (DIRICHLET) and zero-only suffixes (DIRICHLET0)"
                           << " are invalid.");
        const walberla::Vector3<int> rgb = rb.getParameter<walberla::Vector3<int>>("rgb");
        region.r = rgb[0];
        region.g = rgb[1];
        region.b = rgb[2];
        if (region.r < 0 || region.r > 255 || region.g < 0 || region.g > 255 || region.b < 0 || region.b > 255)
            WALBERLA_ABORT("ColorBC.Region RGB values must be in [0,255].");

        if (region.bcId == BC_INLET || region.bcId == BC_OUTLET)
        {
            region.flowVelocity = rb.getParameter<walberla::Vector3<real_t>>("velocity");
            region.flowRho = rb.getParameter<real_t>("rho");
            if (region.flowRho <= real_t(0))
                WALBERLA_ABORT("ColorBC.Region '" << region.uidName << "' requires rho > 0 for inlet/outlet.");
            const auto inletOutletThermal = thermalTypeFromString(rb.getParameter<std::string>("thermal"));
            if (inletOutletThermal != THERMAL_DIRICHLET && inletOutletThermal != THERMAL_ADIABATIC)
            {
                WALBERLA_ABORT("ColorBC.Region '" << region.uidName
                               << "' supports thermal only as Dirichlet or Adiabatic for inlet/outlet boundaries.");
            }
            region.thermalType = inletOutletThermal;
            region.flowTheta = rb.getParameter<real_t>("theta");
            region.theta = region.flowTheta;
        }
        else if (region.bcId == BC_PRESSURE)
        {
            region.flowVelocity = walberla::Vector3<real_t>(real_t(0));
            region.flowRho = rb.getParameter<real_t>("rho");
            if (region.flowRho <= real_t(0))
                WALBERLA_ABORT("ColorBC.Region '" << region.uidName << "' requires rho > 0 for pressure boundaries.");
            if (!rb.isDefined("flow"))
                WALBERLA_ABORT("ColorBC.Region '" << region.uidName << "' requires flow=\"in\" or flow=\"out\" for pressure boundaries.");
            region.pressureFlowMode = pressureFlowModeFromString(rb.getParameter<std::string>("flow"));
            const auto pressureThermal = thermalTypeFromString(rb.getParameter<std::string>("thermal"));
            if (pressureThermal != THERMAL_DIRICHLET && pressureThermal != THERMAL_ADIABATIC)
            {
                WALBERLA_ABORT("ColorBC.Region '" << region.uidName
                               << "' supports thermal only as Dirichlet or Adiabatic for pressure boundaries.");
            }
            region.thermalType = pressureThermal;
            region.flowTheta = rb.getParameter<real_t>("theta");
            region.theta = region.flowTheta;
        }
        else
        {
            if (region.bcId == BC_DIRICHLET)
            {
                region.thermalType = THERMAL_DIRICHLET;
                region.theta = rb.getParameter<real_t>("theta");
                region.nuOutput = rb.getParameter<bool>("Nu", false);
                region.nuLCharPhys = rb.getParameter<real_t>("L_char");
                if (region.nuLCharPhys <= real_t(0))
                {
                    WALBERLA_ABORT("ColorBC.Region '" << region.uidName
                                   << "' requires L_char > 0 for DIRICHLET regions.");
                }
                if (rb.isDefined("Nu_dTheta"))
                {
                    region.hasNuDeltaThetaOverride = true;
                    region.nuDeltaThetaOverride = rb.getParameter<real_t>("Nu_dTheta");
                    if (region.nuDeltaThetaOverride <= real_t(0))
                    {
                        WALBERLA_ABORT("ColorBC.Region '" << region.uidName
                                       << "' requires Nu_dTheta > 0 when provided.");
                    }
                }
            }
            else if (region.bcId == BC_ADIABATIC)
            {
                region.thermalType = THERMAL_ADIABATIC;
            }
            else
            {
                if (region.bcId != BC_HEATLOAD)
                {
                    WALBERLA_ABORT("Unexpected ColorBC bcId for region '" << region.uidName
                                   << "': " << int(region.bcId)
                                   << " (expected BC_HEATLOAD=" << int(BC_HEATLOAD) << ").");
                }
                region.thermalType = THERMAL_HEATLOAD;
                const real_t heatloadWatts = rb.getParameter<real_t>("heatload_watts");
                if (!std::isfinite(double(heatloadWatts)))
                {
                    WALBERLA_ABORT("ColorBC.Region '" << region.uidName
                                   << "' requires finite heatload_watts.");
                }
                if (std::abs(deltaTK) <= 1e-15)
                {
                    WALBERLA_ABORT("ColorBC.Region '" << region.uidName
                                   << "' cannot convert heatload_watts with Physical.deltaT_K == 0.");
                }
                // Convert physical heat flux input to finest-level lattice heatload proxy:
                // heatload_lat = q'' * dx_phys / (k_phys * deltaT_K)
                // where q'' is provided as heatload_watts [W/m^2].
                const double heatloadLat = double(heatloadWatts) * dxPhysFine / (kPhys * deltaTK);
                if (!std::isfinite(heatloadLat))
                {
                    WALBERLA_ABORT("ColorBC.Region '" << region.uidName
                                   << "' produced non-finite lattice heatload from heatload_watts="
                                   << double(heatloadWatts) << ", dx_phys=" << dxPhysFine
                                   << ", k_phys=" << kPhys << ", deltaT_K=" << deltaTK << ".");
                }
                region.heatload = real_t(heatloadLat);
            }
        }

        const auto uidIt = colorRegionByUid.find(region.uidName);
        if (uidIt != colorRegionByUid.end())
        {
            if (!equalColorRegion(uidIt->second, region))
            {
                WALBERLA_ABORT("Duplicate ColorBC.Region name '" << region.uidName
                               << "' with conflicting attributes.");
            }
            continue;
        }

        const std::uint32_t rgbKey =
            (std::uint32_t(std::uint8_t(region.r)) << 16u) |
            (std::uint32_t(std::uint8_t(region.g)) << 8u) |
            (std::uint32_t(std::uint8_t(region.b)));
        const auto rgbIt = uidByRgb.find(rgbKey);
        if (rgbIt != uidByRgb.end() && rgbIt->second != region.uidName)
        {
            WALBERLA_ABORT("Duplicate ColorBC.Region rgb=<" << region.r << "," << region.g << "," << region.b
                           << "> maps to multiple region names: '" << rgbIt->second
                           << "' and '" << region.uidName << "'.");
        }
        uidByRgb[rgbKey] = region.uidName;
        WALBERLA_CHECK_LESS(colorRegions.size(), size_t(std::numeric_limits<walberla::uint16_t>::max()));
        region.regionIndex = walberla::uint16_t(colorRegions.size() + size_t(1));

        if (region.bcId == BC_DIRICHLET && region.thermalType == THERMAL_DIRICHLET)
        {
            if (!hasDirichlet)
            {
                thetaDirichletMin = region.theta;
                thetaDirichletMax = region.theta;
                hasDirichlet = true;
            }
            else
            {
                thetaDirichletMin = std::min(thetaDirichletMin, region.theta);
                thetaDirichletMax = std::max(thetaDirichletMax, region.theta);
            }
        }
        colorRegionByUid.emplace(region.uidName, region);
        colorRegions.push_back(region);
    }
    if (isRoot && disabledColorRegionCount > size_t(0))
        WALBERLA_LOG_INFO("ColorBC skipped " << disabledColorRegionCount << " disabled Region block(s).");
    if (hasDirichlet)
        thetaInit = thetaDirichletMin;
    else if (isRoot)
        WALBERLA_LOG_WARNING("ColorBC has no DIRICHLET* region. Using thetaInit=0 and Nu_* outputs will be undefined.");
    walberla::mesh::ColorToBoundaryMapper<walberla::mesh::TriangleMesh> colorMapper{
        walberla::mesh::BoundaryInfo(walberla::BoundaryUID(kUnmappedBoundaryUid))};
    for (const auto& region : colorRegions)
    {
        colorMapper.set(
            walberla::mesh::TriangleMesh::Color(
                static_cast<walberla::mesh::TriangleMesh::Color::value_type>(region.r),
                static_cast<walberla::mesh::TriangleMesh::Color::value_type>(region.g),
                static_cast<walberla::mesh::TriangleMesh::Color::value_type>(region.b)),
            walberla::mesh::BoundaryInfo(walberla::BoundaryUID(region.uidName)));
    }
    for (auto& region : geometryRegions)
        region.boundaryLocations = colorMapper.addBoundaryInfoToMesh(*region.mesh);

    const bool periodicX = blocks->isPeriodic(0);
    const bool periodicY = blocks->isPeriodic(1);
    const bool periodicZ = blocks->isPeriodic(2);
    const bool periodicAny = periodicX || periodicY || periodicZ;
    struct BoundaryRegionMapping
    {
        bool found = false;
        walberla::uint16_t bcId = BC_NONE;
        walberla::uint16_t regionId = walberla::uint16_t(0);
        walberla::uint8_t thermalType = THERMAL_NONE;
        real_t thermalValue = real_t(0);
        real_t flowRho = real_t(1);
        real_t flowTheta = real_t(0);
        walberla::Vector3<real_t> flowVelocity = walberla::Vector3<real_t>(real_t(0));
        walberla::uint8_t pressureFlowMode = PRESSURE_FLOW_INVALID;
        int nearestRegionIdx = -1;
    };
    auto mappingFromColorRegion = [](const ColorRegionConfig& cfg) -> BoundaryRegionMapping {
        BoundaryRegionMapping mapping;
        mapping.found = true;
        mapping.bcId = cfg.bcId;
        mapping.regionId = cfg.regionIndex;
        mapping.thermalType = cfg.thermalType;
        mapping.thermalValue = (cfg.thermalType == THERMAL_DIRICHLET) ? cfg.theta : cfg.heatload;
        mapping.flowRho = cfg.flowRho;
        mapping.flowTheta = cfg.flowTheta;
        mapping.flowVelocity = cfg.flowVelocity;
        mapping.pressureFlowMode = cfg.pressureFlowMode;
        return mapping;
    };
    std::vector<std::vector<BoundaryRegionMapping>> faceBoundaryMappingsByRegion(geometryRegions.size());
    for (size_t regionIdx = size_t(0); regionIdx < geometryRegions.size(); ++regionIdx)
    {
        const auto& region = geometryRegions[regionIdx];
        auto& faceMappings = faceBoundaryMappingsByRegion[regionIdx];
        faceMappings.resize(size_t(region.mesh->n_faces()));
        for (auto faceIt = region.mesh->faces_begin(); faceIt != region.mesh->faces_end(); ++faceIt)
        {
            const auto fh = *faceIt;
            BoundaryRegionMapping mapping;
            const std::string uidRaw = (*(region.boundaryLocations))[fh].getUid().getIdentifier();
            if (uidRaw != kUnmappedBoundaryUid)
            {
                const std::string uid = toUpper(uidRaw);
                const auto mapIt = colorRegionByUid.find(uid);
                if (mapIt != colorRegionByUid.end())
                    mapping = mappingFromColorRegion(mapIt->second);
            }
            const int faceIdx = fh.idx();
            if (faceIdx >= 0)
                faceMappings[size_t(faceIdx)] = mapping;
        }
    }
    auto mappedBoundaryAtPoint = [&](const walberla::Vector3<real_t>& p, int nearestRegionHint = -1) -> BoundaryRegionMapping {
        BoundaryRegionMapping mapping;
        // DistanceOctree::sqSignedDistance returns a signed squared distance (±d^2).
        // Keep all pruning comparisons in squared-distance units.
        real_t minAbsDistSq = std::numeric_limits<real_t>::max();
        const auto pOpenMesh = walberla::mesh::toOpenMesh(p);

        auto tryRegion = [&](size_t regionIdx) {
            const auto& region = geometryRegions[regionIdx];
            const auto& aabb = regionAabbs[regionIdx];
            real_t dx = real_t(0);
            real_t dy = real_t(0);
            real_t dz = real_t(0);
            if (p[0] < aabb.xMin())
                dx = aabb.xMin() - p[0];
            else if (p[0] > aabb.xMax())
                dx = p[0] - aabb.xMax();
            if (p[1] < aabb.yMin())
                dy = aabb.yMin() - p[1];
            else if (p[1] > aabb.yMax())
                dy = p[1] - aabb.yMax();
            if (p[2] < aabb.zMin())
                dz = aabb.zMin() - p[2];
            else if (p[2] > aabb.zMax())
                dz = p[2] - aabb.zMax();
            const real_t aabbLowerBoundDistSq = dx * dx + dy * dy + dz * dz;
            if (aabbLowerBoundDistSq > minAbsDistSq)
                return;

            walberla::mesh::TriangleMesh::FaceHandle fh;
            const real_t signedDistSq = real_t(region.distanceOctree->sqSignedDistance(pOpenMesh, fh));
            const real_t absDistSq = std::abs(signedDistSq);
            if (absDistSq > minAbsDistSq)
                return;

            minAbsDistSq = absDistSq;
            mapping.nearestRegionIdx = int(regionIdx);
            BoundaryRegionMapping nearestFaceMapping;
            if (fh.is_valid())
            {
                const int faceIdx = fh.idx();
                if (faceIdx >= 0)
                {
                    const auto& faceMappings = faceBoundaryMappingsByRegion[regionIdx];
                    const size_t faceIdxSz = size_t(faceIdx);
                    if (faceIdxSz < faceMappings.size())
                        nearestFaceMapping = faceMappings[faceIdxSz];
                }
            }
            mapping = nearestFaceMapping;
            mapping.nearestRegionIdx = int(regionIdx);
        };

        if (nearestRegionHint >= 0 && size_t(nearestRegionHint) < geometryRegions.size())
            tryRegion(size_t(nearestRegionHint));

        for (size_t regionIdx = size_t(0); regionIdx < geometryRegions.size(); ++regionIdx)
        {
            if (int(regionIdx) == nearestRegionHint)
                continue;
            tryRegion(regionIdx);
        }
        return mapping;
    };

    for (auto& block : *blocks)
    {
        auto* cellType = block.getData<CellTypeField>(cellTypeID);
        auto* bcId = block.getData<BcField>(bcIdID);
        auto* regionId = block.getData<RegionIdField>(regionIdID);
        auto* thermalType = block.getData<ThermalTypeField>(thermalTypeID);
        auto* thermalValue = block.getData<ScalarField>(thermalValueID);
        ScalarField* flowRho = nullptr;
        ScalarField* flowTheta = nullptr;
        VecField* flowVelocity = nullptr;
        if (useOpenBoundary)
        {
            flowRho = block.getData<ScalarField>(flowRhoID);
            flowTheta = block.getData<ScalarField>(flowThetaID);
            flowVelocity = block.getData<VecField>(flowVelocityID);
        }
        const auto bb = blocks->getBlockCellBB(block);
        const auto ci = cellType->xyzSize();
        for (auto cell = ci.begin(); cell != ci.end(); ++cell)
        {
            const int x = cell->x();
            const int y = cell->y();
            const int z = cell->z();
            const int gx = int(bb.xMin()) + x;
            const int gy = int(bb.yMin()) + y;
            const int gz = int(bb.zMin()) + z;
            walberla::Cell globalCell(gx, gy, gz);
            auto center = blocks->getCellCenter(globalCell, uint_t(0));
            if (periodicAny)
                blocks->mapToPeriodicDomain(center);
            const auto cType = isFluidAtPoint(center) ? CELL_FLUID : CELL_SOLID;
            (*cellType)(x, y, z, 0) = cType;
            (*bcId)(x, y, z, 0) = BC_NONE;
            (*regionId)(x, y, z, 0) = walberla::uint16_t(0);
            (*thermalType)(x, y, z, 0) = THERMAL_NONE;
            (*thermalValue)(x, y, z, 0) = real_t(0);
            if (useOpenBoundary)
            {
                (*flowRho)(x, y, z, 0) = real_t(1);
                (*flowTheta)(x, y, z, 0) = thetaInit;
                (*flowVelocity)(x, y, z, 0) = real_t(0);
                (*flowVelocity)(x, y, z, 1) = real_t(0);
                (*flowVelocity)(x, y, z, 2) = real_t(0);
            }
        }
    }
    cellTypeComm();

    uint_t unknownBoundaryUidLocal = uint_t(0);
    for (auto& block : *blocks)
    {
        auto* cellType = block.getData<CellTypeField>(cellTypeID);
        auto* regionId = block.getData<RegionIdField>(regionIdID);
        auto* thermalType = block.getData<ThermalTypeField>(thermalTypeID);
        auto* thermalValue = block.getData<ScalarField>(thermalValueID);
        auto* bcId = block.getData<BcField>(bcIdID);
        ScalarField* flowRho = nullptr;
        ScalarField* flowTheta = nullptr;
        VecField* flowVelocity = nullptr;
        if (useOpenBoundary)
        {
            flowRho = block.getData<ScalarField>(flowRhoID);
            flowTheta = block.getData<ScalarField>(flowThetaID);
            flowVelocity = block.getData<VecField>(flowVelocityID);
        }
        const auto domainBB = blocks->getDomainCellBB(uint_t(0));
        const auto bb = blocks->getBlockCellBB(block);
        const auto ci = cellType->xyzSize();
        int nearestRegionHint = -1;
        for (auto cell = ci.begin(); cell != ci.end(); ++cell)
        {
            const int x = cell->x();
            const int y = cell->y();
            const int z = cell->z();
            if ((*cellType)(x, y, z, 0) != CELL_SOLID) continue;

            const int gx = int(bb.xMin()) + x;
            const int gy = int(bb.yMin()) + y;
            const int gz = int(bb.zMin()) + z;
            const bool hasFaceFluidNeighbor = hasFaceNeighborTypeInDomain(
                cellType, domainBB, gx, gy, gz, x, y, z, periodicX, periodicY, periodicZ, CELL_FLUID);
            const bool hasStencilFluidNeighbor = hasFaceFluidNeighbor || hasStencilNeighborTypeInDomain<CellTypeField, LbStencil>(
                cellType, domainBB, gx, gy, gz, x, y, z, periodicX, periodicY, periodicZ, CELL_FLUID);
            if (!hasStencilFluidNeighbor)
                continue;

            walberla::Cell globalCell(gx, gy, gz);
            auto center = blocks->getCellCenter(globalCell, uint_t(0));
            if (periodicAny)
                blocks->mapToPeriodicDomain(center);
            const auto mapping = mappedBoundaryAtPoint(center, nearestRegionHint);
            nearestRegionHint = mapping.nearestRegionIdx;
            if (mapping.found)
            {
                if (!hasFaceFluidNeighbor &&
                    mapping.bcId != BC_INLET &&
                    mapping.bcId != BC_OUTLET &&
                    mapping.bcId != BC_PRESSURE)
                    continue;
                (*bcId)(x, y, z, 0) = mapping.bcId;
                (*regionId)(x, y, z, 0) = mapping.regionId;
                (*thermalType)(x, y, z, 0) = mapping.thermalType;
                (*thermalValue)(x, y, z, 0) = mapping.thermalValue;
                if (useOpenBoundary)
                {
                    (*flowRho)(x, y, z, 0) = mapping.flowRho;
                    if (mapping.bcId == BC_INLET || mapping.bcId == BC_OUTLET || mapping.bcId == BC_PRESSURE)
                    {
                        (*flowTheta)(x, y, z, 0) = mapping.flowTheta;
                        (*flowVelocity)(x, y, z, 0) = mapping.flowVelocity[0];
                        (*flowVelocity)(x, y, z, 1) = mapping.flowVelocity[1];
                        (*flowVelocity)(x, y, z, 2) = mapping.flowVelocity[2];
                    }
                }
            }
            else
            {
                ++unknownBoundaryUidLocal;
            }
        }
    }
    const uint_t unknownBoundaryUid = walberla::mpi::allReduce(unknownBoundaryUidLocal, walberla::mpi::SUM);
    if (unknownBoundaryUid > uint_t(0))
        WALBERLA_ABORT("Mesh contains boundary faces with colors not mapped by ColorBC.Region."
                       << " Unmapped boundary solids: " << unknownBoundaryUid
                       << ". Add explicit Region entries for every mesh face color.");
    bcIdComm();
    regionIdComm();
    thermalTypeComm();
    thermalValueComm();
    if (useOpenBoundary)
    {
        (*flowRhoComm)();
        (*flowThetaComm)();
        (*flowVelocityComm)();
    }
    const auto thermalBoundaryCache = buildThermalBoundaryCache<CellTypeField, BcField, RegionIdField, ThermalTypeField, ScalarField>(
        blocks, cellTypeID, bcIdID, regionIdID, thermalTypeID, thermalValueID, periodicX, periodicY, periodicZ);

    // Geometry diagnostics.
    std::uint64_t fluidLocal = 0;
    std::uint64_t solidLocal = 0;
    std::vector<std::uint64_t> boundarySolidByRegionLocal(colorRegions.size(), std::uint64_t(0));
    std::uint64_t noneBoundarySolidLocal = 0;
    std::uint64_t fluidOnDomainBoundaryLocal = 0;
    std::uint64_t blocksWithBoundaryLocal = 0;
    std::uint64_t blocksWithFluidNoBoundaryLocal = 0;
    const auto domainBBL0 = blocks->getDomainCellBB(uint_t(0));
    for (auto& block : *blocks)
    {
        auto* cellType = block.getData<CellTypeField>(cellTypeID);
        auto* bcId = block.getData<BcField>(bcIdID);
        auto* regionId = block.getData<RegionIdField>(regionIdID);
        const auto& domainBB = domainBBL0;
        const auto bb = blocks->getBlockCellBB(block);
        const int gx0 = int(bb.xMin());
        const int gy0 = int(bb.yMin());
        const int gz0 = int(bb.zMin());
        const int bnx = int(bb.xSize());
        const int bny = int(bb.ySize());
        const int bnz = int(bb.zSize());
        bool blockHasFluid = false;
        bool blockHasBoundary = false;
        for (int z = 0; z < bnz; ++z)
            for (int y = 0; y < bny; ++y)
                for (int x = 0; x < bnx; ++x)
                {
                    const int gx = gx0 + x;
                    const int gy = gy0 + y;
                    const int gz = gz0 + z;
                    if ((*cellType)(x, y, z, 0) == CELL_FLUID)
                    {
                        ++fluidLocal;
                        blockHasFluid = true;
                        const bool onBoundary =
                            gx == int(domainBB.xMin()) || gx == int(domainBB.xMax()) ||
                            gy == int(domainBB.yMin()) || gy == int(domainBB.yMax()) ||
                            gz == int(domainBB.zMin()) || gz == int(domainBB.zMax());
                        if (onBoundary) ++fluidOnDomainBoundaryLocal;
                        continue;
                    }
                    ++solidLocal;
                    if (!hasFaceNeighborTypeInDomain(cellType, domainBB, gx, gy, gz, x, y, z, periodicX, periodicY, periodicZ, CELL_FLUID))
                        continue;
                    blockHasBoundary = true;
                    const auto bc = (*bcId)(x, y, z, 0);
                    const auto rid = (*regionId)(x, y, z, 0);
                    if (bc != BC_NONE && rid >= walberla::uint16_t(1) && rid <= walberla::uint16_t(colorRegions.size()))
                    {
                        ++boundarySolidByRegionLocal[size_t(rid - walberla::uint16_t(1))];
                    }
                    else
                    {
                        ++noneBoundarySolidLocal;
                    }
                }
        if (blockHasBoundary)
        {
            ++blocksWithBoundaryLocal;
        }
        else if (blockHasFluid)
        {
            ++blocksWithFluidNoBoundaryLocal;
        }
    }
    const std::uint64_t fluidGlobal = walberla::mpi::allReduce(fluidLocal, walberla::mpi::SUM);
    const std::uint64_t solidGlobal = walberla::mpi::allReduce(solidLocal, walberla::mpi::SUM);
    std::vector<std::uint64_t> boundarySolidByRegionGlobal(boundarySolidByRegionLocal.size(), std::uint64_t(0));
    if (!boundarySolidByRegionLocal.empty())
    {
        WALBERLA_MPI_SECTION()
        {
            MPI_Allreduce(
                boundarySolidByRegionLocal.data(),
                boundarySolidByRegionGlobal.data(),
                int(boundarySolidByRegionLocal.size()),
                walberla::MPITrait<std::uint64_t>::type(),
                walberla::MPITrait<std::uint64_t>::operation(walberla::mpi::SUM),
                walberla::mpi::MPIManager::instance()->comm());
        }
    }
    std::uint64_t inletBoundarySolidGlobal = 0;
    std::uint64_t outletBoundarySolidGlobal = 0;
    std::uint64_t pressureBoundarySolidGlobal = 0;
    for (size_t regionIdx = size_t(0); regionIdx < colorRegions.size(); ++regionIdx)
    {
        const auto count = boundarySolidByRegionGlobal[regionIdx];
        if (colorRegions[regionIdx].bcId == BC_INLET)
            inletBoundarySolidGlobal += count;
        else if (colorRegions[regionIdx].bcId == BC_OUTLET)
            outletBoundarySolidGlobal += count;
        else if (colorRegions[regionIdx].bcId == BC_PRESSURE)
            pressureBoundarySolidGlobal += count;
    }
    const std::uint64_t noneBoundarySolidGlobal = walberla::mpi::allReduce(noneBoundarySolidLocal, walberla::mpi::SUM);
    const std::uint64_t fluidOnDomainBoundaryGlobal = walberla::mpi::allReduce(fluidOnDomainBoundaryLocal, walberla::mpi::SUM);
    const std::uint64_t blocksWithBoundaryGlobal = walberla::mpi::allReduce(blocksWithBoundaryLocal, walberla::mpi::SUM);
    const std::uint64_t blocksWithFluidNoBoundaryGlobal = walberla::mpi::allReduce(blocksWithFluidNoBoundaryLocal, walberla::mpi::SUM);
    if (isRoot)
    {
        std::ostringstream geomLine;
        geomLine << "GEOM mode=mesh_regions"
                 << " fluid_cells=" << fluidGlobal
                 << " solid_cells=" << solidGlobal;
        for (size_t regionIdx = size_t(0); regionIdx < colorRegions.size(); ++regionIdx)
            geomLine << " bc_" << toLower(colorRegions[regionIdx].uidName) << "=" << boundarySolidByRegionGlobal[regionIdx];
        geomLine
                 << " LT=" << totalBlocksGlobal
                 << " bb=" << blocksWithBoundaryGlobal
                 << " bf=" << blocksWithFluidNoBoundaryGlobal;
        WALBERLA_LOG_INFO(geomLine.str());
        if (noneBoundarySolidGlobal > 0) WALBERLA_LOG_WARNING("Boundary solid cells with bcId==NONE: " << noneBoundarySolidGlobal);
        if (fluidOnDomainBoundaryGlobal > 0) WALBERLA_LOG_WARNING("Fluid cells on domain boundary: " << fluidOnDomainBoundaryGlobal);
    }

    // Thermal initialization and optional deterministic perturbation.
    auto applyThetaPerturb = [&]() {
        if (cmd.initPerturb <= 0.0)
            return;
        const real_t perturbAmp = real_t(cmd.initPerturb);
        for (auto& block : *blocks)
        {
            auto* cellType = block.getData<CellTypeField>(cellTypeID);
            auto* theta = block.getData<ScalarField>(thetaID);
            const auto bb = blocks->getBlockCellBB(block);
            const auto ci = cellType->xyzSize();
            for (auto cell = ci.begin(); cell != ci.end(); ++cell)
            {
                const int x = cell->x();
                const int y = cell->y();
                const int z = cell->z();
                if ((*cellType)(x, y, z, 0) != CELL_FLUID)
                    continue;
                const int gx = int(bb.xMin()) + x;
                const int gy = int(bb.yMin()) + y;
                const int gz = int(bb.zMin()) + z;
                const real_t noise = real_t(deterministicCenteredNoise(gx, gy, gz));
                (*theta)(x, y, z, 0) += perturbAmp * noise;
            }
        }
    };

    if (!restartEnabled)
    {
        initializeThetaFromCellTypes<ScalarField, CellTypeField, ThermalTypeField, ScalarField>(
            blocks, thetaID, cellTypeID, thermalTypeID, thermalValueID, thetaInit);
        applyThetaPerturb();
    }
    else
    {
        applyThetaPerturb();
        for (auto& block : *blocks)
        {
            auto* theta = block.getData<ScalarField>(thetaID);
            auto* thetaTmp = block.getData<ScalarField>(thetaTmpID);
            thetaTmp->set(*theta);
        }
    }
    scalarComm();
    syncHostToRuntimeThetaBuffers();

    const real_t initialThetaRefForForce = real_t(thetaRef0);
    if (!restartEnabled)
    {
        auto initMacro = mphys::hotplate::gen::InitializeMacroFields{densitySimID, velocitySimID};
        for (auto& block : *blocks)
        {
            auto initPdfs = mphys::hotplate::gen::LBM::InitPdfs{
                pdfSimID, densitySimID, thetaSimID, velocitySimID, real_t(aLatFine), double(initialThetaRefForForce)};
            initMacro(&block);
            initPdfs(&block);
        }
    }

    // GPU-only CUDA-aware MPI capability checks and runtime GPU comm setup.
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    constexpr bool compileTimeCudaAwareMpiAvailable = true;
#else
    constexpr bool compileTimeCudaAwareMpiAvailable = false;
#endif

    bool runtimeCudaAwareQueryAvailable = false;
    int runtimeCudaAwareSupport = -1;
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    runtimeCudaAwareQueryAvailable = true;
    runtimeCudaAwareSupport = MPIX_Query_cuda_support();
#endif

    bool cudaAwareMpiAvailable = compileTimeCudaAwareMpiAvailable;
    if (runtimeCudaAwareQueryAvailable)
        cudaAwareMpiAvailable = cudaAwareMpiAvailable && (runtimeCudaAwareSupport != 0);

    auto mpiErrorToString = [&](int rc) -> std::string {
        char errString[MPI_MAX_ERROR_STRING] = {};
        int errLen = 0;
        const int errRc = MPI_Error_string(rc, errString, &errLen);
        if (errRc == MPI_SUCCESS && errLen > 0)
            return std::string(errString, size_t(errLen));
        return std::string("<unavailable>");
    };

    auto abortOnMpiDiagnosticsFailure = [&](const char* callName, int rc) {
        if (rc == MPI_SUCCESS)
            return;

        const std::string errText = mpiErrorToString(rc);
        WALBERLA_ABORT("MPI call failed in CUDA-aware MPI diagnostics: " << callName
                       << " rc=" << rc << " err='" << errText << "'");
    };

    int mpiVersionMajor = 0;
    int mpiVersionMinor = 0;
    abortOnMpiDiagnosticsFailure("MPI_Get_version", MPI_Get_version(&mpiVersionMajor, &mpiVersionMinor));
    char mpiLibraryVersion[MPI_MAX_LIBRARY_VERSION_STRING];
    int mpiLibraryVersionLen = 0;
    abortOnMpiDiagnosticsFailure(
        "MPI_Get_library_version",
        MPI_Get_library_version(mpiLibraryVersion, &mpiLibraryVersionLen));
    const size_t mpiLibraryVersionSafeLen = strnlen(mpiLibraryVersion, size_t(MPI_MAX_LIBRARY_VERSION_STRING));

    int mpiWorldSize = 1;
    abortOnMpiDiagnosticsFailure(
        "MPI_Comm_size(world)",
        MPI_Comm_size(walberla::mpi::MPIManager::instance()->comm(), &mpiWorldSize));
    const bool requireGpuDirectMpi = (mpiWorldSize > 1);
    bool strictDeviceProbeRan = false;
    bool strictDeviceProbePassed = false;
    int strictDeviceProbeFirstRc = MPI_SUCCESS;

    if (requireGpuDirectMpi)
    {
        strictDeviceProbeRan = true;
        constexpr int kProbeWordCount = 32;
        using ProbeWord = uint32_t;
        std::array<ProbeWord, size_t(kProbeWordCount)> hostSend{};
        std::array<ProbeWord, size_t(kProbeWordCount)> hostRecv{};

        const int worldRank = walberla::mpi::MPIManager::instance()->worldRank();
        const int recvFrom = (worldRank - 1 + mpiWorldSize) % mpiWorldSize;
        const int sendTo = (worldRank + 1) % mpiWorldSize;
        for (int i = 0; i < kProbeWordCount; ++i)
        {
            hostSend[size_t(i)] = ProbeWord(0x5EED0000u) ^ ProbeWord(uint32_t(worldRank * 97 + i));
            hostRecv[size_t(i)] = ProbeWord(0u);
        }

        ProbeWord* devSend = nullptr;
        ProbeWord* devRecv = nullptr;
        WALBERLA_GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devSend), sizeof(ProbeWord) * size_t(kProbeWordCount)));
        WALBERLA_GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devRecv), sizeof(ProbeWord) * size_t(kProbeWordCount)));
        WALBERLA_GPU_CHECK(
            gpuMemcpy(devSend, hostSend.data(), sizeof(ProbeWord) * size_t(kProbeWordCount), gpuMemcpyHostToDevice));
        WALBERLA_GPU_CHECK(
            gpuMemcpy(devRecv, hostRecv.data(), sizeof(ProbeWord) * size_t(kProbeWordCount), gpuMemcpyHostToDevice));

        MPI_Comm comm = walberla::mpi::MPIManager::instance()->comm();
        MPI_Errhandler originalErrHandler = MPI_ERRHANDLER_NULL;
        abortOnMpiDiagnosticsFailure("MPI_Comm_get_errhandler(device probe)", MPI_Comm_get_errhandler(comm, &originalErrHandler));
        abortOnMpiDiagnosticsFailure("MPI_Comm_set_errhandler(device probe)", MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN));

        const int sendrecvRc = MPI_Sendrecv(
            devSend,
            kProbeWordCount,
            walberla::MPITrait<ProbeWord>::type(),
            sendTo,
            9917,
            devRecv,
            kProbeWordCount,
            walberla::MPITrait<ProbeWord>::type(),
            recvFrom,
            9917,
            comm,
            MPI_STATUS_IGNORE);

        const int restoreErrHandlerRc = MPI_Comm_set_errhandler(comm, originalErrHandler);
        abortOnMpiDiagnosticsFailure("MPI_Comm_set_errhandler(restore device probe)", restoreErrHandlerRc);

        bool localProbeOk = (sendrecvRc == MPI_SUCCESS);
        if (!localProbeOk)
            strictDeviceProbeFirstRc = sendrecvRc;

        if (localProbeOk)
        {
            WALBERLA_GPU_CHECK(
                gpuMemcpy(hostRecv.data(), devRecv, sizeof(ProbeWord) * size_t(kProbeWordCount), gpuMemcpyDeviceToHost));
        }
        WALBERLA_GPU_CHECK(gpuFree(devSend));
        WALBERLA_GPU_CHECK(gpuFree(devRecv));

        if (localProbeOk)
        {
            for (int i = 0; i < kProbeWordCount; ++i)
            {
                const ProbeWord expected = ProbeWord(0x5EED0000u) ^ ProbeWord(uint32_t(recvFrom * 97 + i));
                if (hostRecv[size_t(i)] != expected)
                {
                    localProbeOk = false;
                    if (strictDeviceProbeFirstRc == MPI_SUCCESS)
                        strictDeviceProbeFirstRc = MPI_ERR_OTHER;
                    break;
                }
            }
        }

        int localOk = localProbeOk ? 1 : 0;
        int globalOk = 0;
        abortOnMpiDiagnosticsFailure(
            "MPI_Allreduce(device probe)",
            MPI_Allreduce(
                &localOk, &globalOk, 1, MPI_INT, MPI_MIN, walberla::mpi::MPIManager::instance()->comm()));
        strictDeviceProbePassed = (globalOk == 1);
        cudaAwareMpiAvailable = strictDeviceProbePassed;
    }

    if (isRoot)
    {
        WALBERLA_LOG_INFO("CUDA-aware MPI runtime decision: compile_support="
                         << (compileTimeCudaAwareMpiAvailable ? 1 : 0)
                         << " runtime_query_available=" << (runtimeCudaAwareQueryAvailable ? 1 : 0)
                         << " runtime_query_support=" << runtimeCudaAwareSupport
                         << " device_probe_ran=" << (strictDeviceProbeRan ? 1 : 0)
                         << " device_probe_passed=" << (strictDeviceProbePassed ? 1 : 0)
                         << " effective=" << (cudaAwareMpiAvailable ? 1 : 0));
        WALBERLA_LOG_INFO("MPI runtime: std=" << mpiVersionMajor << "." << mpiVersionMinor
                         << " library=\"" << std::string(mpiLibraryVersion, mpiLibraryVersionSafeLen) << "\"");
    }

    if (requireGpuDirectMpi && !cudaAwareMpiAvailable)
    {
        WALBERLA_ABORT("CUDA-aware MPI is required for multi-rank runs (worldSize=" << mpiWorldSize
                       << "), but direct device-buffer MPI probing failed."
                       << " compile_support=" << (compileTimeCudaAwareMpiAvailable ? 1 : 0)
                       << " runtime_query_available=" << (runtimeCudaAwareQueryAvailable ? 1 : 0)
                       << " runtime_query_support=" << runtimeCudaAwareSupport
                       << " device_probe_ran=" << (strictDeviceProbeRan ? 1 : 0)
                       << " device_probe_passed=" << (strictDeviceProbePassed ? 1 : 0)
                       << " device_probe_first_rc=" << strictDeviceProbeFirstRc
                       << " device_probe_first_err='" << mpiErrorToString(strictDeviceProbeFirstRc) << "'."
                       << " Ensure CUDA-aware MPI/device-pointer MPI is available for this MPI stack.");
    }

    const bool useGpuDirectComm = (mpiWorldSize > 1) ? true : cudaAwareMpiAvailable;
    auto runtimePdfThetaComm = std::make_shared<RuntimeGpuCommScheme>(blocks, useGpuDirectComm, true, 2301);
    // Phase-2: runtime CommStart/CommWait carries PDFs only.
    runtimePdfThetaComm->addPackInfo(std::make_shared<RuntimeGpuPackInfo>(pdfSimID));

    auto runtimeThetaTmpComm = std::make_shared<RuntimeGpuThetaTmpCommScheme>(blocks, useGpuDirectComm, true, 2302);
    runtimeThetaTmpComm->addPackInfo(std::make_shared<RuntimeGpuPackInfo>(thetaTmpSimID));
    // One-time startup theta ghost synchronization for restart and non-restart consistency.
    auto runtimeThetaInitComm = std::make_shared<RuntimeGpuThetaTmpCommScheme>(blocks, useGpuDirectComm, true, 2303);
    runtimeThetaInitComm->addPackInfo(std::make_shared<RuntimeGpuPackInfo>(thetaSimID));

    if (!cudaAwareMpiAvailable && isRoot && mpiWorldSize == 1)
        WALBERLA_LOG_INFO(
            "CUDA-aware MPI not detected by MPIX precheck; MPI world size == 1 is allowed."
            " Multi-rank runs will require strict device-buffer MPI probing at startup.");

    (*runtimePdfThetaComm)();
    (*runtimeThetaInitComm)();

    SparseCellIndexList fluidCellIndexList(*blocks);
    SparseCellIndexList boundaryFluidIndexList(*blocks);
    SparseCellIndexList inletBoundaryFluidIndexList(*blocks);
    SparseCellIndexList outletBoundaryFluidIndexList(*blocks);
    SparseCellIndexList pressureBoundaryFluidIndexList(*blocks);
    SparseCellIndexList pressureInBoundaryFluidIndexList(*blocks);
    SparseCellIndexList pressureOutBoundaryFluidIndexList(*blocks);
    std::vector<walberla::Block*> fullFluidBlocks;
    std::vector<walberla::Block*> mixedBlocks;
    std::vector<walberla::Block*> boundaryBlocks;
    std::vector<walberla::Block*> inletBlocks;
    std::vector<walberla::Block*> outletBlocks;
    std::vector<walberla::Block*> pressureBlocks;
    std::vector<walberla::Block*> pressureInBlocks;
    std::vector<walberla::Block*> pressureOutBlocks;
    const size_t localBlockCountReserve = size_t(blocks->getNumberOfBlocks());
    fullFluidBlocks.reserve(localBlockCountReserve);
    mixedBlocks.reserve(localBlockCountReserve);
    boundaryBlocks.reserve(localBlockCountReserve);
    inletBlocks.reserve(localBlockCountReserve);
    outletBlocks.reserve(localBlockCountReserve);
    pressureBlocks.reserve(localBlockCountReserve);
    pressureInBlocks.reserve(localBlockCountReserve);
    pressureOutBlocks.reserve(localBlockCountReserve);
    const bool storeBoundaryFluidIndices = (vtkWriteFrequency > uint_t(0));
    const bool hasInletBoundarySolids = (inletBoundarySolidGlobal > std::uint64_t(0));
    const bool hasOutletBoundarySolids = (outletBoundarySolidGlobal > std::uint64_t(0));
    const bool hasPressureBoundarySolids = (pressureBoundarySolidGlobal > std::uint64_t(0));
    const auto domainBB0 = blocks->getDomainCellBB(uint_t(0));
    std::uint64_t openBoundaryOverlapFluidLocal = 0;
    std::uint64_t fluidCellCountLocal = 0;

    for (auto& block : *blocks)
    {
        auto* blockPtr = dynamic_cast<walberla::Block*>(&block);
        if (blockPtr == nullptr)
        {
            WALBERLA_ABORT(
                "Internal error: expected walberla::Block during FluidSim setup, but dynamic_cast failed.");
        }
        auto* cellType = block.getData<CellTypeField>(cellTypeID);
        auto* bcId = block.getData<BcField>(bcIdID);
        const auto& domainBB = domainBB0;
        const auto bb = blocks->getBlockCellBB(block);

        bool isFullFluid = true;
        {
            const auto ci = cellType->xyzSize();
            for (auto cell = ci.begin(); cell != ci.end(); ++cell)
            {
                if ((*cellType)(cell->x(), cell->y(), cell->z(), 0) != CELL_FLUID)
                {
                    isFullFluid = false;
                    break;
                }
            }
        }

        auto* fluidIndicesPtr = !isFullFluid ? &fluidCellIndexList.getVector(block) : nullptr;
        auto* boundaryIndicesPtr = storeBoundaryFluidIndices ? &boundaryFluidIndexList.getVector(block) : nullptr;
        auto* inletIndicesPtr = hasInletBoundarySolids ? &inletBoundaryFluidIndexList.getVector(block) : nullptr;
        auto* outletIndicesPtr = hasOutletBoundarySolids ? &outletBoundaryFluidIndexList.getVector(block) : nullptr;
        auto* pressureIndicesPtr = hasPressureBoundarySolids ? &pressureBoundaryFluidIndexList.getVector(block) : nullptr;
        if (fluidIndicesPtr != nullptr)
            fluidIndicesPtr->clear();
        if (boundaryIndicesPtr != nullptr)
            boundaryIndicesPtr->clear();
        if (inletIndicesPtr != nullptr)
            inletIndicesPtr->clear();
        if (outletIndicesPtr != nullptr)
            outletIndicesPtr->clear();
        if (pressureIndicesPtr != nullptr)
            pressureIndicesPtr->clear();
        const size_t nx = size_t(bb.xSize());
        const size_t ny = size_t(bb.ySize());
        const size_t nz = size_t(bb.zSize());
        const size_t blockCellCount = nx * ny * nz;
        const size_t faceAreaEstimate = nx * ny + ny * nz + nx * nz;
        const size_t boundaryReserve = std::min(blockCellCount, size_t(2) * faceAreaEstimate);
        if (fluidIndicesPtr != nullptr)
            fluidIndicesPtr->reserve(blockCellCount);
        if (boundaryIndicesPtr != nullptr)
            boundaryIndicesPtr->reserve(boundaryReserve);
        if (inletIndicesPtr != nullptr)
            inletIndicesPtr->reserve(boundaryReserve);
        if (outletIndicesPtr != nullptr)
            outletIndicesPtr->reserve(boundaryReserve);
        if (pressureIndicesPtr != nullptr)
            pressureIndicesPtr->reserve(boundaryReserve);
        bool blockHasBoundary = false;
        const auto ci = cellType->xyzSize();
        for (auto cell = ci.begin(); cell != ci.end(); ++cell)
        {
            const int x = cell->x();
            const int y = cell->y();
            const int z = cell->z();
            if ((*cellType)(x, y, z, 0) != CELL_FLUID)
                continue;
            ++fluidCellCountLocal;
            if (fluidIndicesPtr != nullptr)
                fluidIndicesPtr->emplace_back(walberla::Cell(x, y, z));

            const int gx = int(bb.xMin()) + x;
            const int gy = int(bb.yMin()) + y;
            const int gz = int(bb.zMin()) + z;
            bool boundaryAffected =
                ((!periodicX) && (gx == int(domainBB.xMin()) || gx == int(domainBB.xMax()))) ||
                ((!periodicY) && (gy == int(domainBB.yMin()) || gy == int(domainBB.yMax()))) ||
                ((!periodicZ) && (gz == int(domainBB.zMin()) || gz == int(domainBB.zMax())));

            if (!boundaryAffected)
            {
                boundaryAffected = hasStencilNeighborTypeInDomain<CellTypeField, LbStencil>(
                    cellType, domainBB, gx, gy, gz, x, y, z, periodicX, periodicY, periodicZ, CELL_SOLID);
            }

            if (boundaryAffected)
            {
                blockHasBoundary = true;
                if (boundaryIndicesPtr != nullptr)
                    boundaryIndicesPtr->emplace_back(walberla::Cell(x, y, z));
            }

            bool hasInletOpenLink = false;
            bool hasOutletOpenLink = false;
            bool hasPressureOpenLink = false;
            unsigned int openTypeCount = 0u;
            if (boundaryAffected &&
                (hasInletBoundarySolids || hasOutletBoundarySolids || hasPressureBoundarySolids))
            {
                for (uint_t qi = uint_t(0); qi < LbStencil::Q; ++qi)
                {
                    const auto dir = LbStencil::dir[qi];
                    if (dir == walberla::stencil::C)
                        continue;
                    const int dx = walberla::stencil::cx[dir];
                    const int dy = walberla::stencil::cy[dir];
                    const int dz = walberla::stencil::cz[dir];
                    const int ngx = gx + dx;
                    const int ngy = gy + dy;
                    const int ngz = gz + dz;
                    if (!isNeighborReachable(domainBB, ngx, ngy, ngz, periodicX, periodicY, periodicZ))
                        continue;

                    const int sx = x + dx;
                    const int sy = y + dy;
                    const int sz = z + dz;
                    const auto wallBc = (*bcId)(sx, sy, sz, 0);
                    if (wallBc != BC_INLET && wallBc != BC_OUTLET && wallBc != BC_PRESSURE)
                        continue;

                    if (wallBc == BC_INLET)
                    {
                        hasInletOpenLink = true;
                    }
                    else if (wallBc == BC_OUTLET)
                    {
                        hasOutletOpenLink = true;
                    }
                    else
                    {
                        hasPressureOpenLink = true;
                    }

                    openTypeCount = unsigned(hasInletOpenLink) +
                                    unsigned(hasOutletOpenLink) +
                                    unsigned(hasPressureOpenLink);
                    if (openTypeCount > 1u)
                        break;
                }
            }

            // Keep target lists consistent with detected open-boundary connectivity.
            if (hasInletOpenLink && inletIndicesPtr != nullptr)
                inletIndicesPtr->emplace_back(walberla::Cell(x, y, z));
            if (hasOutletOpenLink && outletIndicesPtr != nullptr)
                outletIndicesPtr->emplace_back(walberla::Cell(x, y, z));
            if (hasPressureOpenLink && pressureIndicesPtr != nullptr)
                pressureIndicesPtr->emplace_back(walberla::Cell(x, y, z));

            if (openTypeCount > 1u)
                ++openBoundaryOverlapFluidLocal;
        }

        if (isFullFluid)
            fullFluidBlocks.push_back(blockPtr);
        else
            mixedBlocks.push_back(blockPtr);

        if (blockHasBoundary)
            boundaryBlocks.push_back(blockPtr);
        if (inletIndicesPtr != nullptr && !inletIndicesPtr->empty())
            inletBlocks.push_back(blockPtr);
        if (outletIndicesPtr != nullptr && !outletIndicesPtr->empty())
            outletBlocks.push_back(blockPtr);
        if (pressureIndicesPtr != nullptr && !pressureIndicesPtr->empty())
            pressureBlocks.push_back(blockPtr);

    }

    const std::uint64_t fluidCellCountGlobal = walberla::mpi::allReduce(fluidCellCountLocal, walberla::mpi::SUM);
    if (fluidCellCountGlobal == std::uint64_t(0))
    {
        WALBERLA_ABORT("No fluid cells detected (CELL_FLUID count is zero). "
                       "Check mesh coloring / region IDs.");
    }

    {
        std::uint64_t inletLocal = 0;
        std::uint64_t outletLocal = 0;
        std::uint64_t pressureLocal = 0;
        for (auto& block : *blocks)
        {
            if (hasInletBoundarySolids)
                inletLocal += std::uint64_t(inletBoundaryFluidIndexList.getVector(block).size());
            if (hasOutletBoundarySolids)
                outletLocal += std::uint64_t(outletBoundaryFluidIndexList.getVector(block).size());
            if (hasPressureBoundarySolids)
                pressureLocal += std::uint64_t(pressureBoundaryFluidIndexList.getVector(block).size());
        }
        const std::uint64_t openBoundaryOverlapFluidGlobal =
            walberla::mpi::allReduce(openBoundaryOverlapFluidLocal, walberla::mpi::SUM);
        if (openBoundaryOverlapFluidGlobal > std::uint64_t(0))
        {
            WALBERLA_ABORT("Open boundary configuration is ambiguous: "
                           << openBoundaryOverlapFluidGlobal
                           << " fluid cells are adjacent to multiple open-boundary types "
                           << "(INLET/OUTLET/PRESSURE). "
                           << "Split regions/colors so each boundary fluid cell maps to exactly one open boundary type.");
        }
        const std::uint64_t inletGlobal = walberla::mpi::allReduce(inletLocal, walberla::mpi::SUM);
        const std::uint64_t outletGlobal = walberla::mpi::allReduce(outletLocal, walberla::mpi::SUM);
        const std::uint64_t pressureGlobal = walberla::mpi::allReduce(pressureLocal, walberla::mpi::SUM);
        if (inletBoundarySolidGlobal > std::uint64_t(0) && inletGlobal == std::uint64_t(0))
            WALBERLA_ABORT("ColorBC contains INLET boundary solids, but no adjacent fluid cells were found.");
        if (outletBoundarySolidGlobal > std::uint64_t(0) && outletGlobal == std::uint64_t(0))
            WALBERLA_ABORT("ColorBC contains OUTLET boundary solids, but no adjacent fluid cells were found.");
        if (pressureBoundarySolidGlobal > std::uint64_t(0) && pressureGlobal == std::uint64_t(0))
            WALBERLA_ABORT("ColorBC contains PRESSURE boundary solids, but no adjacent fluid cells were found.");
        if (isRoot && (inletGlobal > std::uint64_t(0) ||
                       outletGlobal > std::uint64_t(0) ||
                       pressureGlobal > std::uint64_t(0)))
        {
            WALBERLA_LOG_INFO("OPEN_BOUNDARY inletCells=" << inletGlobal
                             << " outletCells=" << outletGlobal
                             << " pressureCells=" << pressureGlobal);
        }
    }

    std::vector<walberla::uint8_t> pressureFlowModeByRegionId(colorRegions.size() + size_t(1), PRESSURE_FLOW_INVALID);
    for (const auto& region : colorRegions)
    {
        if (region.bcId == BC_PRESSURE)
            pressureFlowModeByRegionId[size_t(region.regionIndex)] = region.pressureFlowMode;
    }

    auto precomputeOpenBoundaryTargets = [&](walberla::uint16_t targetBc,
                                             const std::vector<walberla::Block*>& openBlocks,
                                             SparseCellIndexList& openBoundaryFluidList) {
        for (auto* block : openBlocks)
        {
            auto* bcId = block->getData<BcField>(bcIdID);
            auto* flowRho = block->getData<ScalarField>(flowRhoID);
            auto* flowTheta = block->getData<ScalarField>(flowThetaID);
            auto* flowVelocity = block->getData<VecField>(flowVelocityID);
            const auto bb = blocks->getBlockCellBB(*block);
            for (const auto& idx : openBoundaryFluidList.getVector(*block))
            {
                const int x = int(idx.x);
                const int y = int(idx.y);
                const int z = int(idx.z);

                real_t rhoSum = real_t(0);
                real_t thetaSum = real_t(0);
                real_t uxSum = real_t(0);
                real_t uySum = real_t(0);
                real_t uzSum = real_t(0);
                uint_t count = uint_t(0);

                for (uint_t qi = uint_t(0); qi < LbStencil::Q; ++qi)
                {
                    const auto dir = LbStencil::dir[qi];
                    if (dir == walberla::stencil::C)
                        continue;
                    const int dx = walberla::stencil::cx[dir];
                    const int dy = walberla::stencil::cy[dir];
                    const int dz = walberla::stencil::cz[dir];
                    const int nx = x + dx;
                    const int ny = y + dy;
                    const int nz = z + dz;
                    if ((*bcId)(nx, ny, nz, 0) != targetBc)
                        continue;
                    rhoSum += (*flowRho)(nx, ny, nz, 0);
                    thetaSum += (*flowTheta)(nx, ny, nz, 0);
                    uxSum += (*flowVelocity)(nx, ny, nz, 0);
                    uySum += (*flowVelocity)(nx, ny, nz, 1);
                    uzSum += (*flowVelocity)(nx, ny, nz, 2);
                    ++count;
                }

                if (count == uint_t(0))
                {
                    const int gx = int(bb.xMin()) + x;
                    const int gy = int(bb.yMin()) + y;
                    const int gz = int(bb.zMin()) + z;
                    WALBERLA_ABORT(
                        "Open-boundary target precompute failed: zero contributing neighbors."
                        << " level=0"
                        << " localCell=<" << x << "," << y << "," << z << ">"
                        << " globalCell=<" << gx << "," << gy << "," << gz << ">"
                        << " targetBc=" << targetBc
                        << ". Check boundary mapping / neighbor reachability.");
                }

                const real_t invCount = real_t(1) / real_t(count);
                (*flowRho)(x, y, z, 0) = rhoSum * invCount;
                (*flowTheta)(x, y, z, 0) = thetaSum * invCount;
                (*flowVelocity)(x, y, z, 0) = uxSum * invCount;
                (*flowVelocity)(x, y, z, 1) = uySum * invCount;
                (*flowVelocity)(x, y, z, 2) = uzSum * invCount;
            }
        }
    };
    auto precomputePressureBoundaryTargets = [&]() {
        for (auto* block : pressureBlocks)
        {
            auto* bcId = block->getData<BcField>(bcIdID);
            auto* regionId = block->getData<RegionIdField>(regionIdID);
            auto* flowRho = block->getData<ScalarField>(flowRhoID);
            auto* flowTheta = block->getData<ScalarField>(flowThetaID);
            auto* flowVelocity = block->getData<VecField>(flowVelocityID);
            auto& pressureInCells = pressureInBoundaryFluidIndexList.getVector(*block);
            auto& pressureOutCells = pressureOutBoundaryFluidIndexList.getVector(*block);
            pressureInCells.clear();
            pressureOutCells.clear();
            const auto bb = blocks->getBlockCellBB(*block);
            for (const auto& idx : pressureBoundaryFluidIndexList.getVector(*block))
            {
                const int x = int(idx.x);
                const int y = int(idx.y);
                const int z = int(idx.z);

                real_t rhoSum = real_t(0);
                real_t thetaSum = real_t(0);
                real_t nxSum = real_t(0);
                real_t nySum = real_t(0);
                real_t nzSum = real_t(0);
                uint_t count = uint_t(0);
                bool hasFlowIn = false;
                bool hasFlowOut = false;

                for (uint_t qi = uint_t(0); qi < LbStencil::Q; ++qi)
                {
                    const auto dir = LbStencil::dir[qi];
                    if (dir == walberla::stencil::C)
                        continue;
                    const int dx = walberla::stencil::cx[dir];
                    const int dy = walberla::stencil::cy[dir];
                    const int dz = walberla::stencil::cz[dir];
                    const int nx = x + dx;
                    const int ny = y + dy;
                    const int nz = z + dz;
                    const auto neighborBc = (*bcId)(nx, ny, nz, 0);
                    if (neighborBc != BC_PRESSURE)
                        continue;

                    rhoSum += (*flowRho)(nx, ny, nz, 0);
                    thetaSum += (*flowTheta)(nx, ny, nz, 0);
                    nxSum += real_t(dx);
                    nySum += real_t(dy);
                    nzSum += real_t(dz);
                    ++count;

                    const auto rid = (*regionId)(nx, ny, nz, 0);
                    if (rid == walberla::uint16_t(0) || size_t(rid) >= pressureFlowModeByRegionId.size())
                    {
                        const int gx = int(bb.xMin()) + x;
                        const int gy = int(bb.yMin()) + y;
                        const int gz = int(bb.zMin()) + z;
                        const int gnx = int(bb.xMin()) + nx;
                        const int gny = int(bb.yMin()) + ny;
                        const int gnz = int(bb.zMin()) + nz;
                        WALBERLA_ABORT(
                            "Pressure-boundary flow mode classification failed: invalid regionId at PRESSURE neighbor."
                            << " fluidLocal=<" << x << "," << y << "," << z << ">"
                            << " fluidGlobal=<" << gx << "," << gy << "," << gz << ">"
                            << " neighborLocal=<" << nx << "," << ny << "," << nz << ">"
                            << " neighborGlobal=<" << gnx << "," << gny << "," << gnz << ">"
                            << " rid=" << rid
                            << " neighborBc=" << int(neighborBc)
                            << " regionIdVectorSize=" << pressureFlowModeByRegionId.size() << ".");
                    }
                    const auto flowMode = pressureFlowModeByRegionId[size_t(rid)];
                    if (flowMode == PRESSURE_FLOW_IN)
                        hasFlowIn = true;
                    else if (flowMode == PRESSURE_FLOW_OUT)
                        hasFlowOut = true;
                    else
                    {
                        const int gx = int(bb.xMin()) + x;
                        const int gy = int(bb.yMin()) + y;
                        const int gz = int(bb.zMin()) + z;
                        const int gnx = int(bb.xMin()) + nx;
                        const int gny = int(bb.yMin()) + ny;
                        const int gnz = int(bb.zMin()) + nz;
                        WALBERLA_ABORT(
                            "Pressure-boundary flow mode classification failed: invalid flow mode at PRESSURE neighbor."
                            << " fluidLocal=<" << x << "," << y << "," << z << ">"
                            << " fluidGlobal=<" << gx << "," << gy << "," << gz << ">"
                            << " neighborLocal=<" << nx << "," << ny << "," << nz << ">"
                            << " neighborGlobal=<" << gnx << "," << gny << "," << gnz << ">"
                            << " flowMode=" << int(flowMode) << ".");
                    }
                }

                if (count == uint_t(0))
                {
                    const int gx = int(bb.xMin()) + x;
                    const int gy = int(bb.yMin()) + y;
                    const int gz = int(bb.zMin()) + z;
                    WALBERLA_ABORT(
                        "Pressure-boundary target precompute failed: zero contributing neighbors."
                        << " level=0"
                        << " localCell=<" << x << "," << y << "," << z << ">"
                        << " globalCell=<" << gx << "," << gy << "," << gz << ">"
                        << ". Check boundary mapping / neighbor reachability.");
                }
                if (hasFlowIn && hasFlowOut)
                {
                    const int gx = int(bb.xMin()) + x;
                    const int gy = int(bb.yMin()) + y;
                    const int gz = int(bb.zMin()) + z;
                    WALBERLA_ABORT(
                        "Pressure-boundary flow mode is ambiguous at fluid cell."
                        << " localCell=<" << x << "," << y << "," << z << ">"
                        << " globalCell=<" << gx << "," << gy << "," << gz << ">"
                        << ". Neighboring PRESSURE regions mix flow=\"in\" and flow=\"out\".");
                }

                const real_t normalNormSq = nxSum * nxSum + nySum * nySum + nzSum * nzSum;
                if (normalNormSq <= real_t(1.0e-20))
                {
                    const int gx = int(bb.xMin()) + x;
                    const int gy = int(bb.yMin()) + y;
                    const int gz = int(bb.zMin()) + z;
                    WALBERLA_ABORT(
                        "Pressure-boundary normal is degenerate (zero-length) at fluid cell."
                        << " localCell=<" << x << "," << y << "," << z << ">"
                        << " globalCell=<" << gx << "," << gy << "," << gz << ">"
                        << ". Ensure PRESSURE boundary neighbors define a consistent outward direction.");
                }

                const real_t invCount = real_t(1) / real_t(count);
                (*flowRho)(x, y, z, 0) = rhoSum * invCount;
                (*flowTheta)(x, y, z, 0) = thetaSum * invCount;
                (*flowVelocity)(x, y, z, 0) = nxSum * invCount;
                (*flowVelocity)(x, y, z, 1) = nySum * invCount;
                (*flowVelocity)(x, y, z, 2) = nzSum * invCount;

                if (hasFlowIn)
                    pressureInCells.emplace_back(walberla::Cell(x, y, z));
                else if (hasFlowOut)
                    pressureOutCells.emplace_back(walberla::Cell(x, y, z));
                else
                {
                    const int gx = int(bb.xMin()) + x;
                    const int gy = int(bb.yMin()) + y;
                    const int gz = int(bb.zMin()) + z;
                    WALBERLA_ABORT(
                        "Pressure-boundary flow mode classification failed: no PRESSURE neighbor provided flow=\"in\" or flow=\"out\"."
                        << " localCell=<" << x << "," << y << "," << z << ">"
                        << " globalCell=<" << gx << "," << gy << "," << gz << ">.");
                }
            }
        }

        pressureInBlocks.clear();
        pressureOutBlocks.clear();
        for (auto* block : pressureBlocks)
        {
            if (!pressureInBoundaryFluidIndexList.getVector(*block).empty())
                pressureInBlocks.push_back(block);
            if (!pressureOutBoundaryFluidIndexList.getVector(*block).empty())
                pressureOutBlocks.push_back(block);
        }
    };
    if (useOpenBoundary)
    {
        precomputeOpenBoundaryTargets(BC_INLET, inletBlocks, inletBoundaryFluidIndexList);
        precomputeOpenBoundaryTargets(BC_OUTLET, outletBlocks, outletBoundaryFluidIndexList);
        precomputePressureBoundaryTargets();
    }
    syncHostToRuntimeBoundaryFields();

    // Sweep construction and per-step callback assembly.
    const real_t nuLevel = real_t(nuLatTargetFine);
    const real_t alphaLevel = real_t(alphaLatFine);
    const real_t aLatLevel = real_t(aLatFine);

    std::shared_ptr<mphys::hotplate::gen::OpenBoundaryReconstructInletSerial> openBoundaryInletSweepSerial;
    std::shared_ptr<mphys::hotplate::gen::OpenBoundaryReconstructOutletSerial> openBoundaryOutletSweepSerial;
    std::shared_ptr<mphys::hotplate::gen::OpenBoundaryReconstructPressureInSerial> openBoundaryPressureInSweepSerial;
    std::shared_ptr<mphys::hotplate::gen::OpenBoundaryReconstructPressureOutSerial> openBoundaryPressureOutSweepSerial;
    std::shared_ptr<mphys::hotplate::gen::ClampOpenBoundaryThetaTmpInletSerial> clampInletSweepSerial;
    std::shared_ptr<mphys::hotplate::gen::ClampOpenBoundaryThetaTmpOutletSerial> clampOutletSweepSerial;
    std::shared_ptr<mphys::hotplate::gen::ClampOpenBoundaryThetaTmpPressureSerial> clampPressureSweepSerial;
    if (useOpenBoundary)
    {
        openBoundaryInletSweepSerial = std::make_shared<mphys::hotplate::gen::OpenBoundaryReconstructInletSerial>(
            bcIdSimID,
            flowRhoSimID,
            flowVelocitySimID,
            inletBoundaryFluidIndexList,
            pdfSimID,
            thetaSimID,
            double(aLatLevel),
            double(initialThetaRefForForce));
        openBoundaryOutletSweepSerial = std::make_shared<mphys::hotplate::gen::OpenBoundaryReconstructOutletSerial>(
            bcIdSimID,
            flowRhoSimID,
            flowVelocitySimID,
            outletBoundaryFluidIndexList,
            pdfSimID,
            thetaSimID,
            double(aLatLevel),
            double(initialThetaRefForForce));
        openBoundaryPressureInSweepSerial = std::make_shared<mphys::hotplate::gen::OpenBoundaryReconstructPressureInSerial>(
            bcIdSimID,
            flowRhoSimID,
            flowVelocitySimID,
            pressureInBoundaryFluidIndexList,
            pdfSimID,
            thetaSimID,
            velocitySimID,
            double(aLatLevel),
            double(initialThetaRefForForce));
        openBoundaryPressureOutSweepSerial = std::make_shared<mphys::hotplate::gen::OpenBoundaryReconstructPressureOutSerial>(
            bcIdSimID,
            flowRhoSimID,
            flowVelocitySimID,
            pressureOutBoundaryFluidIndexList,
            pdfSimID,
            thetaSimID,
            velocitySimID,
            double(aLatLevel),
            double(initialThetaRefForForce));
        clampInletSweepSerial = std::make_shared<mphys::hotplate::gen::ClampOpenBoundaryThetaTmpInletSerial>(
            bcIdSimID, flowThetaSimID, inletBoundaryFluidIndexList, thermalTypeSimID, thetaTmpSimID);
        clampOutletSweepSerial = std::make_shared<mphys::hotplate::gen::ClampOpenBoundaryThetaTmpOutletSerial>(
            bcIdSimID, flowThetaSimID, outletBoundaryFluidIndexList, thermalTypeSimID, thetaTmpSimID);
        clampPressureSweepSerial = std::make_shared<mphys::hotplate::gen::ClampOpenBoundaryThetaTmpPressureSerial>(
            bcIdSimID, flowThetaSimID, pressureBoundaryFluidIndexList, thermalTypeSimID, thetaTmpSimID);
    }
    auto streamCollideDenseSweepSerial = std::make_shared<mphys::hotplate::gen::LBM::StreamCollideDenseSerial>(
        pdfSimID, densitySimID, thetaSimID, velocitySimID, aLatLevel, nuLevel, double(initialThetaRefForForce));
    auto streamCollideSparseSweepSerial = std::make_shared<mphys::hotplate::gen::LBM::StreamCollideSerial>(
        pdfSimID, fluidCellIndexList, densitySimID, thetaSimID, velocitySimID, aLatLevel, nuLevel, double(initialThetaRefForForce));
    auto thermalBcDenseSweepSerial = std::make_shared<mphys::hotplate::gen::ApplyThermalBoundaryDenseSerial>(
        cellTypeSimID, thermalTypeSimID, thermalValueSimID, thetaTmpSimID, 1.0);
    mphys::hotplate::gen::ThetaUpdateDenseSerial thetaDenseSweepSerial(thetaSimID, thetaTmpSimID, velocitySimID, alphaLevel);
    mphys::hotplate::gen::ThetaUpdateSerial thetaSparseSweepSerial(fluidCellIndexList, thetaSimID, thetaTmpSimID, velocitySimID, alphaLevel);
#ifdef FLUIDSIM_GPU_BUILD
    SwapTheta<RuntimeField> swapTheta{thetaSimID, thetaTmpSimID};
#else
#error "FluidSim_gpu requires FLUIDSIM_GPU_BUILD (CUDA) configuration."
#endif

    auto wallLinks = [&](auto link) -> bool {
        blocks->transformBlockLocalToGlobalCell(link.wallCell, link.block);
        const auto& dom = domainBB0;
        const bool xWall = (!periodicX) && (link.wallCell.x() < dom.xMin() || link.wallCell.x() > dom.xMax());
        const bool yWall = (!periodicY) && (link.wallCell.y() < dom.yMin() || link.wallCell.y() > dom.yMax());
        const bool zWall = (!periodicZ) && (link.wallCell.z() < dom.zMin() || link.wallCell.z() > dom.zMax());
        if (!(xWall || yWall || zWall))
            return false;
        return true;
    };
    auto meshLinks = [&](auto link) -> bool {
        auto* cellType = link.block.template getData<CellTypeField>(cellTypeID);
        auto* bcId = link.block.template getData<BcField>(bcIdID);
        auto wallCellGlobal = link.wallCell;
        blocks->transformBlockLocalToGlobalCell(wallCellGlobal, link.block);
        const auto& domainBB = domainBB0;
        if (!inDomainCell(domainBB, int(wallCellGlobal.x()), int(wallCellGlobal.y()), int(wallCellGlobal.z())))
            return false;
        const auto wallBc = (*bcId)(link.wallCell.x(), link.wallCell.y(), link.wallCell.z(), 0);
        return (*cellType)(link.fluidCell.x(), link.fluidCell.y(), link.fluidCell.z(), 0) == CELL_FLUID &&
               (*cellType)(link.wallCell.x(), link.wallCell.y(), link.wallCell.z(), 0) == CELL_SOLID &&
               wallBc != BC_INLET &&
               wallBc != BC_OUTLET &&
               wallBc != BC_PRESSURE;
    };
    auto mergedNoSlipLinks = [&](auto link) -> bool {
        return wallLinks(link) || meshLinks(link);
    };
    auto noSlipMerged = mphys::hotplate::gen::NoSlipFactory{blocks, pdfSimID}.fromLinks(mergedNoSlipLinks);

    std::vector<ThermalBCBlockEntry> thermalBCBlocks;
    for (const auto& kv : *thermalBoundaryCache)
    {
        auto* block = const_cast<walberla::IBlock*>(kv.first);
        thermalBCBlocks.push_back(ThermalBCBlockEntry{block, &kv.second});
    }

    real_t currentThetaRef = initialThetaRefForForce;
    auto updateThetaRef = [&]() {
        double thetaSumLocal = 0.0;
        double volumeLocal = 0.0;
        gpureduce::reduceThetaRefLocal(
            fullFluidBlocks,
            mixedBlocks,
            fluidCellIndexList,
            thetaSimID,
            1.0,
            thetaSumLocal,
            volumeLocal);
        double thetaReduceLocal[2] = {thetaSumLocal, volumeLocal};
        double thetaReduceGlobal[2] = {0.0, 0.0};
        WALBERLA_MPI_SECTION()
        {
            MPI_Allreduce(
                thetaReduceLocal,
                thetaReduceGlobal,
                2,
                walberla::MPITrait<double>::type(),
                walberla::MPITrait<double>::operation(walberla::mpi::SUM),
                walberla::mpi::MPIManager::instance()->comm());
        }
        if (thetaReduceGlobal[1] <= 0.0)
        {
            WALBERLA_ABORT(
                "theta_ref update failed: global fluid volume is zero. "
                "This usually means geometry/cell-type mapping produced no fluid cells. "
                "Check mesh coloring/roles, BC mapping, and sparse fluid index lists.");
        }
        if (!std::isfinite(thetaReduceGlobal[0]))
        {
            WALBERLA_ABORT(
                "theta_ref update failed: non-finite global theta sum. "
                "thetaSumGlobal=" << thetaReduceGlobal[0]
                << ", fluidVolumeGlobal=" << thetaReduceGlobal[1] << ".");
        }
        const double thetaRef = thetaReduceGlobal[0] / thetaReduceGlobal[1];
        if (!std::isfinite(thetaRef))
        {
            WALBERLA_ABORT(
                "theta_ref update failed: non-finite theta_ref. "
                "thetaRef=" << thetaRef
                << ", thetaSumGlobal=" << thetaReduceGlobal[0]
                << ", fluidVolumeGlobal=" << thetaReduceGlobal[1] << ".");
        }
        currentThetaRef = real_t(thetaRef);
        streamCollideDenseSweepSerial->theta_ref() = thetaRef;
        streamCollideSparseSweepSerial->theta_ref() = thetaRef;
    };

    auto applyOpenBoundary = [&]() {
        if (!useOpenBoundary)
            return;
        const double thetaRef = double(currentThetaRef);
        auto& inletSweep = *openBoundaryInletSweepSerial;
        auto& outletSweep = *openBoundaryOutletSweepSerial;
        auto& pressureInSweep = *openBoundaryPressureInSweepSerial;
        auto& pressureOutSweep = *openBoundaryPressureOutSweepSerial;
        inletSweep.theta_ref() = thetaRef;
        outletSweep.theta_ref() = thetaRef;
        pressureInSweep.theta_ref() = thetaRef;
        pressureOutSweep.theta_ref() = thetaRef;
        for (auto* block : inletBlocks)
            inletSweep(block);
        for (auto* block : outletBlocks)
            outletSweep(block);
        for (auto* block : pressureInBlocks)
            pressureInSweep(block);
        for (auto* block : pressureOutBlocks)
            pressureOutSweep(block);
    };

    auto clampOpenBoundaryThetaTmp = [&]() {
        if (!useOpenBoundary)
            return;
        auto& inletSweep = *clampInletSweepSerial;
        auto& outletSweep = *clampOutletSweepSerial;
        auto& pressureSweep = *clampPressureSweepSerial;
        for (auto* block : inletBlocks)
            inletSweep(block);
        for (auto* block : outletBlocks)
            outletSweep(block);
        for (auto* block : pressureBlocks)
            pressureSweep(block);
    };

    auto startCommunicatePdfTheta = [&]() {
        runtimePdfThetaComm->startCommunication();
    };
    auto waitCommunicatePdfTheta = [&]() {
        runtimePdfThetaComm->wait();
    };
    auto communicateThetaTmp = [&]() {
        runtimeThetaTmpComm->startCommunication();
        runtimeThetaTmpComm->wait();
    };

    auto applyMeshThermalBC = [&]() {
        auto& sweep = *thermalBcDenseSweepSerial;
        for (auto& block : *blocks)
            sweep(&block);
    };
    if (!restartEnabled)
    {
        // One-time BC bootstrap so the initial state is consistent with the runtime
        // ghost-cell Dirichlet/adiabatic treatment before timestep 1.
        for (auto& block : *blocks)
        {
            auto* theta = block.getData<ScalarField>(thetaID);
            auto* thetaTmp = block.getData<ScalarField>(thetaTmpID);
            thetaTmp->set(*theta);
        }
        // Ensure GPU theta/thetaTmp buffers match host bootstrap values before GPU comm/BC/swap.
        syncHostToRuntimeThetaBuffers();
        clampOpenBoundaryThetaTmp();
        communicateThetaTmp();
        applyMeshThermalBC();
        for (auto& block : *blocks)
            swapTheta(&block);
        // Host theta is synchronized on-demand by runtime writers (VTK/checkpoint).
    }

    auto runThetaDense = [&](walberla::IBlock* block) {
        thetaDenseSweepSerial(block);
    };
    auto runThetaSparse = [&](walberla::IBlock* block) {
        thetaSparseSweepSerial(block);
    };
    auto runStreamDense = [&](walberla::IBlock* block) {
        (*streamCollideDenseSweepSerial)(block);
    };
    auto runStreamSparse = [&](walberla::IBlock* block) {
        (*streamCollideSparseSweepSerial)(block);
    };

    auto thermalStep = [&]() {
        for (auto* block : fullFluidBlocks)
            runThetaDense(block);
        for (auto* block : mixedBlocks)
            runThetaSparse(block);
        clampOpenBoundaryThetaTmp();
        communicateThetaTmp();
        applyMeshThermalBC();
        for (auto& block : *blocks)
            swapTheta(&block);
    };

    auto applyNoSlip = [&]() {
        for (auto* block : boundaryBlocks)
        {
            noSlipMerged(block);
        }
    };


    std::vector<CheckpointRegionInfo> checkpointRegions;
    checkpointRegions.reserve(geometryRegions.size());
    for (const auto& region : geometryRegions)
    {
        CheckpointRegionInfo info;
        info.name = region.name;
        info.role = region.role;
        info.sourceFileBytes = region.sourceFileBytes;
        info.sourceFileHash = region.sourceFileHash;
        info.scale = region.scale;
        info.translateFraction = region.translateFraction;
        checkpointRegions.push_back(std::move(info));
    }

    std::vector<NuRegionOutputInfo> nuOutputRegions;
    for (const auto& region : colorRegions)
    {
        if (region.bcId != BC_DIRICHLET || !region.nuOutput)
            continue;
        const double nuLCharLatFine = double(region.nuLCharPhys) / dxPhysFine;
        if (!std::isfinite(nuLCharLatFine) || nuLCharLatFine <= 0.0)
        {
            WALBERLA_ABORT("ColorBC.Region '" << region.uidName
                           << "' produced non-finite/invalid Nu L_char in lattice units.");
        }
        NuRegionOutputInfo info;
        info.regionId = region.regionIndex;
        info.regionName = region.uidName;
        info.lCharLatFine = nuLCharLatFine;
        info.hasDeltaThetaOverride = region.hasNuDeltaThetaOverride;
        info.deltaThetaOverride = double(region.nuDeltaThetaOverride);
        nuOutputRegions.push_back(std::move(info));
    }
    if (isRoot)
    {
        for (const auto& region : nuOutputRegions)
        {
            const bool useOverride = region.hasDeltaThetaOverride;
            const double dTheta = useOverride ? region.deltaThetaOverride : 1.0;
            WALBERLA_LOG_INFO("NU_REGION name=" << region.regionName
                             << " L_char_phys=" << (region.lCharLatFine * dxPhysFine)
                             << " L_char_lat=" << region.lCharLatFine
                             << " dT_source=" << (useOverride ? "Nu_dTheta" : "default_1")
                             << " dT=" << dTheta);
        }
    }
    std::vector<NuVtkFieldInfo> nuVtkFields;
    if (vtkWriteFrequency > uint_t(0))
    {
        nuVtkFields.reserve(nuOutputRegions.size());
        for (const auto& region : nuOutputRegions)
        {
            const std::string valueStorageName = "nu_local_" + region.regionName;
            const std::string countStorageName = "nu_count_local_" + region.regionName;
            const auto valueFieldID = walberla::field::addToStorage<ScalarField>(
                blocks, valueStorageName, real_t(0.0), walberla::field::fzyx, uint_t(1));
            const auto countFieldID = walberla::field::addToStorage<ScalarField>(
                blocks, countStorageName, real_t(0.0), walberla::field::fzyx, uint_t(1));
            nuVtkFields.push_back(NuVtkFieldInfo{region.regionId, region.regionName, valueFieldID, countFieldID});
        }
    }

    // Runtime binding handoff.
    FluidSimRuntimeBindings binding;
    binding.isRoot = isRoot;
    binding.cmd = cmd;

    binding.numTimesteps = numTimesteps;
    binding.thetaUpdateEvery = thetaUpdateEvery;

    binding.blocks = blocks;
    binding.blockForest = blockForest;

    binding.updateThetaRef = updateThetaRef;
    binding.syncRuntimeToHostTheta = syncRuntimeToHostTheta;
    binding.syncRuntimeToHostRhoVelTheta = syncRuntimeToHostRhoVelTheta;
    binding.syncRuntimeToHostPdfThetaRhoVel = syncRuntimeToHostPdfThetaRhoVel;
    binding.applyOpenBoundary = applyOpenBoundary;
    binding.startCommunicatePdfTheta = startCommunicatePdfTheta;
    binding.waitCommunicatePdfTheta = waitCommunicatePdfTheta;
    binding.applyNoSlip = applyNoSlip;
    binding.thermalStep = thermalStep;
    binding.runStreamDense = runStreamDense;
    binding.runStreamSparse = runStreamSparse;

    binding.fullFluidBlocks = &fullFluidBlocks;
    binding.mixedBlocks = &mixedBlocks;
    binding.thermalBCBlocks = &thermalBCBlocks;
    binding.fluidCellIndexList = &fluidCellIndexList;
    binding.boundaryFluidIndexList = &boundaryFluidIndexList;

    binding.pdfID = pdfID;
    binding.densityID = densityID;
    binding.velocityID = velocityID;
    binding.thetaID = thetaID;
    binding.densityRuntimeID = densitySimID;
    binding.velocityRuntimeID = velocitySimID;
    binding.thetaRuntimeID = thetaSimID;
    binding.cellTypeID = cellTypeID;
    binding.bcIdID = bcIdID;
    binding.thermalTypeID = thermalTypeID;
    binding.thermalValueID = thermalValueID;
    binding.regionIdID = regionIdID;

    binding.currentThetaRef = &currentThetaRef;
    binding.thetaDirichletMax = thetaDirichletMax;
    binding.thetaDirichletMin = thetaDirichletMin;

    binding.checkpointPaths = checkpointPaths;
    binding.checkpointRegions = std::move(checkpointRegions);
    binding.nuOutputRegions = std::move(nuOutputRegions);
    binding.nuVtkFields = std::move(nuVtkFields);

    binding.domainSizePhys = domainSizePhys;
    binding.paddingSizePhys = paddingSizePhys;
    binding.fullSizePhys = fullSizePhys;
    binding.interiorFineCells = interiorFineCells;
    binding.totalFineCells = fineDomainCells;
    binding.cellsPerBlock = cellsPerBlock;
    binding.paddingFineCells = paddingFineCells;
    binding.periodicFlags = periodicFlags;

    binding.vtkWriteFrequency = vtkWriteFrequency;
    binding.dtPhysFine = dtPhysFine;

    return runFluidSimRuntime(binding);
}

} // namespace fluidsim
