// SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
// SPDX-License-Identifier: GPL-3.0-or-later
#include "src/GpuReductions.hpp"

#if defined(FLUIDSIM_GPU_BUILD) && defined(WALBERLA_BUILD_WITH_CUDA)

#include "gpu/ErrorChecking.h"
#include "gpu/GPUField.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <vector>

namespace fluidsim::gpureduce
{
namespace
{
using RuntimeField = walberla::gpu::GPUField<walberla::real_t>;
using SparseCellIdx = walberla::experimental::sweep::CellIdx;

template <typename T>
class CachedPinnedHostBuffer
{
public:
    CachedPinnedHostBuffer() = default;

    ~CachedPinnedHostBuffer() noexcept
    {
        if (data_ != nullptr)
            (void) gpuFreeHost(data_);
        data_ = nullptr;
        capacity_ = size_t(0);
    }

    CachedPinnedHostBuffer(const CachedPinnedHostBuffer&) = delete;
    CachedPinnedHostBuffer& operator=(const CachedPinnedHostBuffer&) = delete;

    T* ensure(size_t count)
    {
        if (count == size_t(0))
            return nullptr;
        if (count > capacity_)
        {
            if (data_ != nullptr)
                WALBERLA_GPU_CHECK(gpuFreeHost(data_));
            WALBERLA_GPU_CHECK(gpuHostAlloc(reinterpret_cast<void**>(&data_), count * sizeof(T), gpuHostAllocDefault));
            capacity_ = count;
        }
        return data_;
    }

private:
    T* data_ = nullptr;
    size_t capacity_ = size_t(0);
};

template <typename T>
class CachedDeviceBuffer
{
public:
    CachedDeviceBuffer() = default;

    ~CachedDeviceBuffer() noexcept
    {
        if (data_ != nullptr)
            (void) gpuFree(data_);
        data_ = nullptr;
        capacity_ = size_t(0);
    }

    CachedDeviceBuffer(const CachedDeviceBuffer&) = delete;
    CachedDeviceBuffer& operator=(const CachedDeviceBuffer&) = delete;

    T* ensure(size_t count)
    {
        if (count == size_t(0))
            return nullptr;
        if (count > capacity_)
        {
            if (data_ != nullptr)
                WALBERLA_GPU_CHECK(gpuFree(data_));
            WALBERLA_GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
            capacity_ = count;
        }
        return data_;
    }

private:
    T* data_ = nullptr;
    size_t capacity_ = size_t(0);
};

template <typename T>
struct DeviceFieldViewT
{
    const T* ptr = nullptr;
    int64_t xStride = int64_t(0);
    int64_t yStride = int64_t(0);
    int64_t zStride = int64_t(0);
    int64_t fStride = int64_t(0);
    int64_t nx = int64_t(0);
    int64_t ny = int64_t(0);
    int64_t nz = int64_t(0);
};

using DeviceFieldView = DeviceFieldViewT<walberla::real_t>;

template <typename T>
inline DeviceFieldViewT<T> makeFieldViewT(const walberla::gpu::GPUField<T>& field)
{
    DeviceFieldViewT<T> view;
    view.ptr = field.dataAt(walberla::cell_idx_t(0), walberla::cell_idx_t(0), walberla::cell_idx_t(0), walberla::cell_idx_t(0));
    view.xStride = int64_t(field.xStride());
    view.yStride = int64_t(field.yStride());
    view.zStride = int64_t(field.zStride());
    view.fStride = int64_t(field.fStride());
    view.nx = int64_t(field.xSize());
    view.ny = int64_t(field.ySize());
    view.nz = int64_t(field.zSize());
    return view;
}

inline DeviceFieldView makeFieldView(const RuntimeField& field)
{
    return makeFieldViewT<walberla::real_t>(field);
}

template <typename T>
__device__ inline T loadValueT(const DeviceFieldViewT<T>& view, int64_t x, int64_t y, int64_t z, int64_t f = int64_t(0))
{
    const int64_t offset = x * view.xStride + y * view.yStride + z * view.zStride + f * view.fStride;
    return view.ptr[offset];
}

__device__ inline double loadValue(const DeviceFieldView& view, int64_t x, int64_t y, int64_t z, int64_t f = int64_t(0))
{
    return double(loadValueT<walberla::real_t>(view, x, y, z, f));
}

__device__ inline void atomicMaxDouble(double* address, double value)
{
    auto* addressUll = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *addressUll;
    while (__longlong_as_double(old) < value)
    {
        const unsigned long long assumed = old;
        old = atomicCAS(addressUll, assumed, __double_as_longlong(value));
        if (old == assumed)
            break;
    }
}

__device__ inline double atomicAddDouble(double* address, double value)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    return atomicAdd(address, value);
#else
    auto* addressAsUll = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *addressAsUll;
    while (true)
    {
        const unsigned long long assumed = old;
        const double updated = __longlong_as_double(assumed) + value;
        old = atomicCAS(addressAsUll, assumed, __double_as_longlong(updated));
        if (old == assumed)
            return __longlong_as_double(assumed);
    }
#endif
}

__global__ void reduceThetaRefFullKernel(
    DeviceFieldView theta,
    double cellVolume,
    double* thetaSumOut,
    double* volumeOut)
{
    const int64_t totalCells = theta.nx * theta.ny * theta.nz;
    const int64_t stride = int64_t(blockDim.x) * int64_t(gridDim.x);
    int64_t idx = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);

    double thetaSumLocal = 0.0;
    double volumeLocal = 0.0;
    while (idx < totalCells)
    {
        const int64_t z = idx / (theta.nx * theta.ny);
        const int64_t rem = idx - z * theta.nx * theta.ny;
        const int64_t y = rem / theta.nx;
        const int64_t x = rem - y * theta.nx;

        thetaSumLocal += loadValue(theta, x, y, z) * cellVolume;
        volumeLocal += cellVolume;

        idx += stride;
    }

    if (thetaSumLocal != 0.0)
        atomicAddDouble(thetaSumOut, thetaSumLocal);
    if (volumeLocal != 0.0)
        atomicAddDouble(volumeOut, volumeLocal);
}

__global__ void reduceThetaRefMixedKernel(
    DeviceFieldView theta,
    const SparseCellIdx* indices,
    size_t count,
    double cellVolume,
    double* thetaSumOut,
    double* volumeOut)
{
    const int64_t stride = int64_t(blockDim.x) * int64_t(gridDim.x);
    int64_t idx = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);

    double thetaSumLocal = 0.0;
    double volumeLocal = 0.0;
    while (idx < int64_t(count))
    {
        const auto cell = indices[size_t(idx)];
        thetaSumLocal += loadValue(theta, cell.x, cell.y, cell.z) * cellVolume;
        volumeLocal += cellVolume;
        idx += stride;
    }

    if (thetaSumLocal != 0.0)
        atomicAddDouble(thetaSumOut, thetaSumLocal);
    if (volumeLocal != 0.0)
        atomicAddDouble(volumeOut, volumeLocal);
}

__global__ void reduceMinimalFullKernel(
    DeviceFieldView velocity,
    DeviceFieldView theta,
    DeviceFieldView rho,
    double cellVolume,
    double* uMaxSqOut,
    double* uySqVolumeOut,
    double* uSqVolumeOut,
    double* energyOut,
    double* massOut,
    unsigned int* nonFiniteOut)
{
    const int64_t totalCells = velocity.nx * velocity.ny * velocity.nz;
    const int64_t stride = int64_t(blockDim.x) * int64_t(gridDim.x);
    int64_t idx = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);

    double uMaxSqLocal = 0.0;
    double uySqVolumeLocal = 0.0;
    double uSqVolumeLocal = 0.0;
    double energyLocal = 0.0;
    double massLocal = 0.0;

    while (idx < totalCells)
    {
        const int64_t z = idx / (velocity.nx * velocity.ny);
        const int64_t rem = idx - z * velocity.nx * velocity.ny;
        const int64_t y = rem / velocity.nx;
        const int64_t x = rem - y * velocity.nx;

        const double ux = loadValue(velocity, x, y, z, int64_t(0));
        const double uy = loadValue(velocity, x, y, z, int64_t(1));
        const double uz = loadValue(velocity, x, y, z, int64_t(2));
        const double rhoVal = loadValue(rho, x, y, z);
        const double thetaVal = loadValue(theta, x, y, z);
        if (!::isfinite(ux) || !::isfinite(uy) || !::isfinite(uz) || !::isfinite(rhoVal) || !::isfinite(thetaVal))
        {
            atomicExch(nonFiniteOut, 1u);
            idx += stride;
            continue;
        }
        const double uSq = ux * ux + uy * uy + uz * uz;
        if (uSq > uMaxSqLocal)
            uMaxSqLocal = uSq;
        uySqVolumeLocal += (uy * uy) * cellVolume;
        uSqVolumeLocal += uSq * cellVolume;
        energyLocal += thetaVal * cellVolume;
        massLocal += rhoVal * cellVolume;

        idx += stride;
    }

    atomicMaxDouble(uMaxSqOut, uMaxSqLocal);
    if (uySqVolumeLocal != 0.0)
        atomicAddDouble(uySqVolumeOut, uySqVolumeLocal);
    if (uSqVolumeLocal != 0.0)
        atomicAddDouble(uSqVolumeOut, uSqVolumeLocal);
    if (energyLocal != 0.0)
        atomicAddDouble(energyOut, energyLocal);
    if (massLocal != 0.0)
        atomicAddDouble(massOut, massLocal);
}

__global__ void reduceMinimalMixedKernel(
    DeviceFieldView velocity,
    DeviceFieldView theta,
    DeviceFieldView rho,
    const SparseCellIdx* indices,
    size_t count,
    double cellVolume,
    double* uMaxSqOut,
    double* uySqVolumeOut,
    double* uSqVolumeOut,
    double* energyOut,
    double* massOut,
    unsigned int* nonFiniteOut)
{
    const int64_t stride = int64_t(blockDim.x) * int64_t(gridDim.x);
    int64_t idx = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);

    double uMaxSqLocal = 0.0;
    double uySqVolumeLocal = 0.0;
    double uSqVolumeLocal = 0.0;
    double energyLocal = 0.0;
    double massLocal = 0.0;

    while (idx < int64_t(count))
    {
        const auto cell = indices[size_t(idx)];
        const double ux = loadValue(velocity, cell.x, cell.y, cell.z, int64_t(0));
        const double uy = loadValue(velocity, cell.x, cell.y, cell.z, int64_t(1));
        const double uz = loadValue(velocity, cell.x, cell.y, cell.z, int64_t(2));
        const double rhoVal = loadValue(rho, cell.x, cell.y, cell.z);
        const double thetaVal = loadValue(theta, cell.x, cell.y, cell.z);
        if (!::isfinite(ux) || !::isfinite(uy) || !::isfinite(uz) || !::isfinite(rhoVal) || !::isfinite(thetaVal))
        {
            atomicExch(nonFiniteOut, 1u);
            idx += stride;
            continue;
        }
        const double uSq = ux * ux + uy * uy + uz * uz;
        if (uSq > uMaxSqLocal)
            uMaxSqLocal = uSq;
        uySqVolumeLocal += (uy * uy) * cellVolume;
        uSqVolumeLocal += uSq * cellVolume;
        energyLocal += thetaVal * cellVolume;
        massLocal += rhoVal * cellVolume;
        idx += stride;
    }

    atomicMaxDouble(uMaxSqOut, uMaxSqLocal);
    if (uySqVolumeLocal != 0.0)
        atomicAddDouble(uySqVolumeOut, uySqVolumeLocal);
    if (uSqVolumeLocal != 0.0)
        atomicAddDouble(uSqVolumeOut, uSqVolumeLocal);
    if (energyLocal != 0.0)
        atomicAddDouble(energyOut, energyLocal);
    if (massLocal != 0.0)
        atomicAddDouble(massOut, massLocal);
}

__device__ inline void faceDirection(uint8_t dirIdx, int& dx, int& dy, int& dz)
{
    switch (dirIdx)
    {
    case uint8_t(0): dx = -1; dy = 0; dz = 0; return;
    case uint8_t(1): dx = 1;  dy = 0; dz = 0; return;
    case uint8_t(2): dx = 0;  dy = -1; dz = 0; return;
    case uint8_t(3): dx = 0;  dy = 1;  dz = 0; return;
    case uint8_t(4): dx = 0;  dy = 0; dz = -1; return;
    case uint8_t(5): dx = 0;  dy = 0; dz = 1; return;
    default: dx = 0; dy = 0; dz = 0; return;
    }
}

__global__ void reduceNuKernel(
    DeviceFieldView theta,
    const GpuNuEntryPacked* entries,
    size_t count,
    double invDx,
    double faceArea,
    double* fluxAreaOut,
    double* areaOut,
    size_t regionCount)
{
    const int64_t stride = int64_t(blockDim.x) * int64_t(gridDim.x);
    int64_t idx = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);

    while (idx < int64_t(count))
    {
        const GpuNuEntryPacked entry = entries[size_t(idx)];
        if (entry.slot >= 0 && size_t(entry.slot) < regionCount)
        {
            const double thetaWall = double(entry.thermalValue);
            double fluxAreaLocal = 0.0;
            double areaLocal = 0.0;

            for (uint8_t nbrIdx = uint8_t(0); nbrIdx < entry.fluidNeighborCount; ++nbrIdx)
            {
                int dx = 0;
                int dy = 0;
                int dz = 0;
                faceDirection(entry.fluidNeighborDirIndices[size_t(nbrIdx)], dx, dy, dz);
                const double thetaFluid = loadValue(
                    theta,
                    int64_t(entry.x + dx),
                    int64_t(entry.y + dy),
                    int64_t(entry.z + dz));
                const double flux = (thetaWall - thetaFluid) * (2.0 * invDx);
                fluxAreaLocal += flux * faceArea;
                areaLocal += faceArea;
            }

            if (fluxAreaLocal != 0.0)
                atomicAddDouble(&fluxAreaOut[size_t(entry.slot)], fluxAreaLocal);
            if (areaLocal != 0.0)
                atomicAddDouble(&areaOut[size_t(entry.slot)], areaLocal);
        }

        idx += stride;
    }
}

inline int launchBlockCount(int64_t itemCount, int threadsPerBlock, int gridCapX)
{
    if (itemCount <= int64_t(0))
        return 0;
    const int64_t blocks = (itemCount + int64_t(threadsPerBlock) - int64_t(1)) / int64_t(threadsPerBlock);
    return int(std::min<int64_t>(blocks, int64_t(gridCapX)));
}

constexpr int kThetaFullThreads = 256;
constexpr int kThetaMixedThreads = 256;
constexpr int kMinimalFullThreads = 256;
constexpr int kMinimalMixedThreads = 256;
constexpr int kNuThreads = 256;
constexpr int kGridCapX = 65535;

void logKernelLaunchConfigOnce()
{
    static bool logged = false;
    if (!logged)
    {
        logged = true;
        WALBERLA_LOG_INFO(
            "GPU launch config:"
            << " thetaFullTPB=" << kThetaFullThreads
            << " thetaMixedTPB=" << kThetaMixedThreads
            << " minimalFullTPB=" << kMinimalFullThreads
            << " minimalMixedTPB=" << kMinimalMixedThreads
            << " nuTPB=" << kNuThreads
            << " gridCapX=" << kGridCapX
            << " profile=production_default");
    }
}

} // namespace

NuBoundaryCacheCuda::~NuBoundaryCacheCuda()
{
    for (auto& kv : cache_)
    {
        if (kv.second.devicePtr != nullptr)
            (void) gpuFree(kv.second.devicePtr);
    }
    cache_.clear();
}

const GpuNuEntryPacked* NuBoundaryCacheCuda::getOrCreateDeviceEntries(
    const std::vector<ThermalBoundaryCell>& entries,
    const std::unordered_map<walberla::uint16_t, size_t>& regionSlotById,
    size_t& outCount)
{
    auto it = cache_.find(&entries);
    if (it != cache_.end())
    {
        outCount = it->second.count;
        return it->second.devicePtr;
    }

    std::vector<GpuNuEntryPacked> packed;
    packed.reserve(entries.size());
    for (const auto& entry : entries)
    {
        if (entry.thermalType != THERMAL_DIRICHLET || entry.bcId != BC_DIRICHLET)
            continue;
        const auto slotIt = regionSlotById.find(entry.regionId);
        if (slotIt == regionSlotById.end())
            continue;

        GpuNuEntryPacked item;
        item.x = int32_t(entry.x);
        item.y = int32_t(entry.y);
        item.z = int32_t(entry.z);
        item.slot = int32_t(slotIt->second);
        item.thermalValue = entry.thermalValue;
        item.fluidNeighborCount = entry.fluidNeighborCount;
        for (size_t i = size_t(0); i < size_t(6); ++i)
            item.fluidNeighborDirIndices[i] = entry.fluidNeighborDirIndices[i];
        packed.push_back(item);
    }

    Entry cacheEntry;
    cacheEntry.count = packed.size();
    if (!packed.empty())
    {
        WALBERLA_GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&cacheEntry.devicePtr), packed.size() * sizeof(GpuNuEntryPacked)));
        WALBERLA_GPU_CHECK(gpuMemcpy(
            cacheEntry.devicePtr,
            packed.data(),
            packed.size() * sizeof(GpuNuEntryPacked),
            gpuMemcpyHostToDevice));
    }

    const auto insertResult = cache_.emplace(&entries, cacheEntry);
    const auto insertedIt = insertResult.first;
    outCount = insertedIt->second.count;
    return insertedIt->second.devicePtr;
}

bool reduceThetaRefLocal(
    const std::vector<walberla::Block*>& fullFluidBlocks,
    const std::vector<walberla::Block*>& mixedBlocks,
    SparseCellIndexList& fluidCellIndexList,
    walberla::BlockDataID thetaRuntimeID,
    double thetaRefCellVolume,
    double& thetaSumLocal,
    double& volumeLocal)
{
    logKernelLaunchConfigOnce();
    constexpr size_t kNumAccums = size_t(2);
    static thread_local CachedPinnedHostBuffer<double> hostAccumCache;
    static thread_local CachedDeviceBuffer<double> devAccumCache;
    double* hostAccum = hostAccumCache.ensure(kNumAccums);
    double* devAccum = devAccumCache.ensure(kNumAccums);
    WALBERLA_GPU_CHECK(gpuMemset(devAccum, 0, kNumAccums * sizeof(double)));

    for (auto* block : fullFluidBlocks)
    {
        auto* theta = block->getData<RuntimeField>(thetaRuntimeID);
        WALBERLA_CHECK_NOT_NULLPTR(theta);
        const DeviceFieldView thetaView = makeFieldView(*theta);
        const int64_t count = thetaView.nx * thetaView.ny * thetaView.nz;
        const int grid = launchBlockCount(count, kThetaFullThreads, kGridCapX);
        if (grid > 0)
        {
            reduceThetaRefFullKernel<<<grid, kThetaFullThreads>>>(thetaView, thetaRefCellVolume, devAccum + 0, devAccum + 1);
#ifndef NDEBUG
            WALBERLA_GPU_CHECK_LAST_ERROR();
#endif
        }
    }

    for (auto* block : mixedBlocks)
    {
        auto* theta = block->getData<RuntimeField>(thetaRuntimeID);
        WALBERLA_CHECK_NOT_NULLPTR(theta);
        const DeviceFieldView thetaView = makeFieldView(*theta);
        auto& fluidIndices = fluidCellIndexList.getVector(*block);
        const auto* indicesPtr = fluidIndices.data();
        const size_t indexCount = fluidIndices.size();
        const int grid = launchBlockCount(int64_t(indexCount), kThetaMixedThreads, kGridCapX);
        if (grid > 0)
        {
            reduceThetaRefMixedKernel<<<grid, kThetaMixedThreads>>>(
                thetaView,
                indicesPtr,
                indexCount,
                thetaRefCellVolume,
                devAccum + 0,
                devAccum + 1);
#ifndef NDEBUG
            WALBERLA_GPU_CHECK_LAST_ERROR();
#endif
        }
    }

    WALBERLA_GPU_CHECK_LAST_ERROR();
    WALBERLA_GPU_CHECK(gpuMemcpy(hostAccum, devAccum, kNumAccums * sizeof(double), gpuMemcpyDeviceToHost));

    thetaSumLocal = hostAccum[size_t(0)];
    volumeLocal = hostAccum[size_t(1)];
    return true;
}

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
    double& massLocal)
{
    logKernelLaunchConfigOnce();
    constexpr size_t kNumAccums = size_t(5);
    static thread_local CachedPinnedHostBuffer<double> hostAccumCache;
    static thread_local CachedDeviceBuffer<double> devAccumCache;
    static thread_local CachedPinnedHostBuffer<unsigned int> hostNonFiniteCache;
    static thread_local CachedDeviceBuffer<unsigned int> devNonFiniteCache;
    double* hostAccum = hostAccumCache.ensure(kNumAccums);
    double* devAccum = devAccumCache.ensure(kNumAccums);
    unsigned int* hostNonFinite = hostNonFiniteCache.ensure(size_t(1));
    unsigned int* devNonFinite = devNonFiniteCache.ensure(size_t(1));
    WALBERLA_GPU_CHECK(gpuMemset(devAccum, 0, kNumAccums * sizeof(double)));
    WALBERLA_GPU_CHECK(gpuMemset(devNonFinite, 0, sizeof(unsigned int)));

    for (auto* block : fullFluidBlocks)
    {
        auto* velocity = block->getData<RuntimeField>(velocityRuntimeID);
        auto* theta = block->getData<RuntimeField>(thetaRuntimeID);
        auto* rho = block->getData<RuntimeField>(densityRuntimeID);
        WALBERLA_CHECK_NOT_NULLPTR(velocity);
        WALBERLA_CHECK_NOT_NULLPTR(theta);
        WALBERLA_CHECK_NOT_NULLPTR(rho);

        const DeviceFieldView velocityView = makeFieldView(*velocity);
        const DeviceFieldView thetaView = makeFieldView(*theta);
        const DeviceFieldView rhoView = makeFieldView(*rho);
        const int64_t count = velocityView.nx * velocityView.ny * velocityView.nz;
        const int grid = launchBlockCount(count, kMinimalFullThreads, kGridCapX);
        if (grid > 0)
        {
            reduceMinimalFullKernel<<<grid, kMinimalFullThreads>>>(
                velocityView,
                thetaView,
                rhoView,
                cellVolume,
                devAccum + 0,
                devAccum + 1,
                devAccum + 2,
                devAccum + 3,
                devAccum + 4,
                devNonFinite);
#ifndef NDEBUG
            WALBERLA_GPU_CHECK_LAST_ERROR();
#endif
        }
    }

    for (auto* block : mixedBlocks)
    {
        auto* velocity = block->getData<RuntimeField>(velocityRuntimeID);
        auto* theta = block->getData<RuntimeField>(thetaRuntimeID);
        auto* rho = block->getData<RuntimeField>(densityRuntimeID);
        WALBERLA_CHECK_NOT_NULLPTR(velocity);
        WALBERLA_CHECK_NOT_NULLPTR(theta);
        WALBERLA_CHECK_NOT_NULLPTR(rho);

        const DeviceFieldView velocityView = makeFieldView(*velocity);
        const DeviceFieldView thetaView = makeFieldView(*theta);
        const DeviceFieldView rhoView = makeFieldView(*rho);
        auto& fluidIndices = fluidCellIndexList.getVector(*block);
        const auto* indicesPtr = fluidIndices.data();
        const size_t indexCount = fluidIndices.size();
        const int grid = launchBlockCount(int64_t(indexCount), kMinimalMixedThreads, kGridCapX);
        if (grid > 0)
        {
            reduceMinimalMixedKernel<<<grid, kMinimalMixedThreads>>>(
                velocityView,
                thetaView,
                rhoView,
                indicesPtr,
                indexCount,
                cellVolume,
                devAccum + 0,
                devAccum + 1,
                devAccum + 2,
                devAccum + 3,
                devAccum + 4,
                devNonFinite);
#ifndef NDEBUG
            WALBERLA_GPU_CHECK_LAST_ERROR();
#endif
        }
    }

    WALBERLA_GPU_CHECK_LAST_ERROR();
    WALBERLA_GPU_CHECK(gpuMemcpy(hostAccum, devAccum, kNumAccums * sizeof(double), gpuMemcpyDeviceToHost));
    WALBERLA_GPU_CHECK(gpuMemcpy(hostNonFinite, devNonFinite, sizeof(unsigned int), gpuMemcpyDeviceToHost));
    if (hostNonFinite[0] != 0u)
        return false;

    uMaxSqLocal = hostAccum[size_t(0)];
    uySqVolumeLocal = hostAccum[size_t(1)];
    uSqVolumeLocal = hostAccum[size_t(2)];
    energyLocal = hostAccum[size_t(3)];
    massLocal = hostAccum[size_t(4)];
    return true;
}

bool reduceNuLocal(
    const std::vector<ThermalBCBlockEntry>& thermalBCBlocks,
    walberla::BlockDataID thetaRuntimeID,
    const std::unordered_map<walberla::uint16_t, size_t>& regionSlotById,
    double invDx,
    double faceArea,
    NuBoundaryCacheCuda& cache,
    std::vector<double>& fluxAreaLocal,
    std::vector<double>& areaLocal)
{
    logKernelLaunchConfigOnce();
    size_t regionCount = size_t(0);
    for (const auto& slotEntry : regionSlotById)
        regionCount = std::max(regionCount, slotEntry.second + size_t(1));
    fluxAreaLocal.resize(regionCount);
    areaLocal.resize(regionCount);
    if (regionCount == size_t(0))
        return true;

    static thread_local CachedDeviceBuffer<double> devFluxAreaCache;
    static thread_local CachedDeviceBuffer<double> devAreaCache;
    static thread_local CachedPinnedHostBuffer<double> hostFluxAreaCache;
    static thread_local CachedPinnedHostBuffer<double> hostAreaCache;
    double* devFluxArea = devFluxAreaCache.ensure(regionCount);
    double* devArea = devAreaCache.ensure(regionCount);
    double* hostFluxArea = hostFluxAreaCache.ensure(regionCount);
    double* hostArea = hostAreaCache.ensure(regionCount);
    WALBERLA_GPU_CHECK(gpuMemset(devFluxArea, 0, regionCount * sizeof(double)));
    WALBERLA_GPU_CHECK(gpuMemset(devArea, 0, regionCount * sizeof(double)));

    for (const auto& blockEntry : thermalBCBlocks)
    {
        size_t entryCount = size_t(0);
        const GpuNuEntryPacked* devEntries =
            cache.getOrCreateDeviceEntries(*blockEntry.entries, regionSlotById, entryCount);
        if (entryCount == size_t(0))
            continue;

        auto* theta = blockEntry.block->getData<RuntimeField>(thetaRuntimeID);
        WALBERLA_CHECK_NOT_NULLPTR(theta);
        const DeviceFieldView thetaView = makeFieldView(*theta);

        const int grid = launchBlockCount(int64_t(entryCount), kNuThreads, kGridCapX);
        if (grid > 0)
        {
            reduceNuKernel<<<grid, kNuThreads>>>(
                thetaView,
                devEntries,
                entryCount,
                invDx,
                faceArea,
                devFluxArea,
                devArea,
                regionCount);
#ifndef NDEBUG
            WALBERLA_GPU_CHECK_LAST_ERROR();
#endif
        }
    }

    WALBERLA_GPU_CHECK_LAST_ERROR();
    WALBERLA_GPU_CHECK(gpuMemcpy(hostFluxArea, devFluxArea, regionCount * sizeof(double), gpuMemcpyDeviceToHost));
    WALBERLA_GPU_CHECK(gpuMemcpy(hostArea, devArea, regionCount * sizeof(double), gpuMemcpyDeviceToHost));
    std::copy(hostFluxArea, hostFluxArea + regionCount, fluxAreaLocal.begin());
    std::copy(hostArea, hostArea + regionCount, areaLocal.begin());
    return true;
}

} // namespace fluidsim::gpureduce

#endif
