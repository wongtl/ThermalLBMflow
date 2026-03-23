// SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "core/all.h"
#include "blockforest/all.h"
#include "field/all.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fluidsim
{
using walberla::real_t;
using walberla::uint_t;

// Shared model constants and output naming.
constexpr walberla::uint8_t CELL_FLUID = walberla::uint8_t(0);
constexpr walberla::uint8_t CELL_SOLID = walberla::uint8_t(1);

constexpr walberla::uint16_t BC_NONE = walberla::uint16_t(0);
constexpr walberla::uint16_t BC_DIRICHLET = walberla::uint16_t(1);
constexpr walberla::uint16_t BC_ADIABATIC = walberla::uint16_t(2);
constexpr walberla::uint16_t BC_HEATLOAD = walberla::uint16_t(3);
constexpr walberla::uint16_t BC_INLET = walberla::uint16_t(4);
constexpr walberla::uint16_t BC_OUTLET = walberla::uint16_t(5);
constexpr walberla::uint16_t BC_PRESSURE = walberla::uint16_t(6);
constexpr walberla::uint8_t PRESSURE_FLOW_INVALID = walberla::uint8_t(0);
constexpr walberla::uint8_t PRESSURE_FLOW_IN = walberla::uint8_t(1);
constexpr walberla::uint8_t PRESSURE_FLOW_OUT = walberla::uint8_t(2);
constexpr walberla::uint8_t THERMAL_NONE = walberla::uint8_t(0);
constexpr walberla::uint8_t THERMAL_DIRICHLET = walberla::uint8_t(1);
constexpr walberla::uint8_t THERMAL_ADIABATIC = walberla::uint8_t(2);
constexpr walberla::uint8_t THERMAL_HEATLOAD = walberla::uint8_t(3);
inline constexpr char kOutputBaseDir[] = "output";
inline constexpr char kCheckpointDirName[] = "checkpoint";
inline constexpr char kVtkDirName[] = "vtk";
inline constexpr char kCheckpointStatePrefix[] = "fluidsim_state";
constexpr std::array<std::array<int, 3>, 6> kFaceNbrDirs = {
    std::array<int, 3>{-1, 0, 0}, std::array<int, 3>{1, 0, 0}, std::array<int, 3>{0, -1, 0},
    std::array<int, 3>{0, 1, 0},  std::array<int, 3>{0, 0, -1}, std::array<int, 3>{0, 0, 1}};

// CPU-only threading mode selection.
enum class ParallelMode
{
    Outer,
    Inner,
    Serial
};

// Runtime command-line options.
struct CmdOptions
{
    uint_t timesteps = 0;
    uint_t minimalLogsEvery = 0;
    uint_t thermalLogsEvery = 0;
    double initPerturb = 0.0;
    uint_t checkpointEvery = 0;
    uint_t vtkEvery = 0;
    bool vtkInit = false;
    bool vtkMeshOnly = false;
    ParallelMode parallelMode = ParallelMode::Outer;
};

// Geometry, checkpoint, and region metadata.
struct MeshGeometryConfig
{
    std::string checkpointFolder;
    real_t scale = real_t(1);
    walberla::Vector3<real_t> translateFraction = walberla::Vector3<real_t>(real_t(0));
};

enum class GeometryRole
{
    FluidContainer,
    SolidObstacle
};

struct GeometryRegionConfig
{
    std::string name;
    std::string meshFile;
    GeometryRole role = GeometryRole::SolidObstacle;
    real_t scale = real_t(1);
    walberla::Vector3<real_t> translateFraction = walberla::Vector3<real_t>(real_t(0));
};

struct CheckpointPaths
{
    std::filesystem::path forestFile;
    std::filesystem::path pdfFile;
    std::filesystem::path densityFile;
    std::filesystem::path velocityFile;
    std::filesystem::path thetaFile;
    std::filesystem::path metaFile;
};

struct ColorRegionConfig
{
    std::string uidName;
    walberla::uint16_t bcId = BC_NONE;
    walberla::uint16_t regionIndex = walberla::uint16_t(0);
    walberla::uint8_t thermalType = THERMAL_ADIABATIC;
    bool nuOutput = false;
    real_t theta = real_t(0);
    real_t nuLCharPhys = real_t(0);
    bool hasNuDeltaThetaOverride = false;
    real_t nuDeltaThetaOverride = real_t(0);
    real_t heatload = real_t(0);
    real_t flowRho = real_t(1);
    real_t flowTheta = real_t(0);
    walberla::Vector3<real_t> flowVelocity = walberla::Vector3<real_t>(real_t(0));
    walberla::uint8_t pressureFlowMode = PRESSURE_FLOW_INVALID;
    int r = 0;
    int g = 0;
    int b = 0;
};

// Common utility helpers.
#ifndef FLUIDSIM_RUNTIME_ONLY
static CheckpointPaths makeCheckpointPaths(const std::string& prefix)
{
    CheckpointPaths paths;
    paths.forestFile = std::filesystem::path(prefix + "_forest.sbf");
    paths.pdfFile = std::filesystem::path(prefix + "_pdf.dat");
    paths.densityFile = std::filesystem::path(prefix + "_density.dat");
    paths.velocityFile = std::filesystem::path(prefix + "_velocity.dat");
    paths.thetaFile = std::filesystem::path(prefix + "_theta.dat");
    paths.metaFile = std::filesystem::path(prefix + "_meta.txt");
    return paths;
}
#endif // FLUIDSIM_RUNTIME_ONLY
static void ensureDirectory(const std::filesystem::path& dir, const std::string& context)
{
    if (dir.empty())
        return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
    {
        WALBERLA_ABORT("Failed to " << context << ": " << dir.string()
                       << " (" << ec.message() << ")");
    }
}

template <typename T>
static std::string vec3ToCsv(const walberla::Vector3<T>& v)
{
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << v[0] << "," << v[1] << "," << v[2];
    return oss.str();
}

// Shared string helpers.
static std::string toUpper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::toupper(c)); });
    return s;
}

enum class StrictUnsignedParseStatus
{
    Ok,
    Invalid,
    Overflow
};

[[maybe_unused]] static StrictUnsignedParseStatus tryParseStrictUnsignedLongLong(
    const std::string& rawValue,
    unsigned long long& valueOut)
{
    if (rawValue.empty())
        return StrictUnsignedParseStatus::Invalid;
    if (rawValue.front() == '+' || rawValue.front() == '-')
        return StrictUnsignedParseStatus::Invalid;
    if (!std::all_of(rawValue.begin(), rawValue.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
        return StrictUnsignedParseStatus::Invalid;

    unsigned long long parsed = 0ULL;
    const auto* begin = rawValue.data();
    const auto* end = begin + rawValue.size();
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec == std::errc::result_out_of_range)
        return StrictUnsignedParseStatus::Overflow;
    if (result.ec != std::errc() || result.ptr != end)
        return StrictUnsignedParseStatus::Invalid;

    valueOut = parsed;
    return StrictUnsignedParseStatus::Ok;
}

[[maybe_unused]] static uint_t parseStrictUintOrAbort(const std::string& context, const std::string& rawValue)
{
    unsigned long long parsed = 0ULL;
    const auto status = tryParseStrictUnsignedLongLong(rawValue, parsed);
    if (status == StrictUnsignedParseStatus::Overflow)
        WALBERLA_ABORT(context << " is too large for an unsigned integer: '" << rawValue << "'");
    if (status != StrictUnsignedParseStatus::Ok)
        WALBERLA_ABORT(context << " expects an unsigned integer, got '" << rawValue << "'");
    if (parsed > static_cast<unsigned long long>(std::numeric_limits<uint_t>::max()))
        WALBERLA_ABORT(context << " is too large for uint_t: '" << rawValue << "'");
    return static_cast<uint_t>(parsed);
}

[[maybe_unused]] static void validatePrmArgOrAbort(int argc, char** argv)
{
    if (argc <= 1 || argv[1] == nullptr)
        WALBERLA_ABORT("Expected parameter file path as argv[1] before runtime flags.");

    const std::string prmArg(argv[1]);
    if (prmArg.empty() || prmArg.rfind("--", 0) == 0)
    {
        WALBERLA_ABORT("Expected parameter file path as argv[1] before runtime flags, got '"
                       << prmArg << "'");
    }
}

// Setup-only parsing and metadata helpers.
#ifndef FLUIDSIM_RUNTIME_ONLY
template <typename T>
static walberla::Vector3<T> parseVec3Csv(const std::string& text)
{
    if constexpr (std::is_same_v<T, uint_t>)
    {
        const size_t comma1 = text.find(',');
        const size_t comma2 = (comma1 == std::string::npos) ? std::string::npos : text.find(',', comma1 + size_t(1));
        if (comma1 == std::string::npos || comma2 == std::string::npos || text.find(',', comma2 + size_t(1)) != std::string::npos)
            WALBERLA_ABORT("Invalid Vec3 CSV format in checkpoint metadata: '" << text << "'");

        const std::string xText = text.substr(size_t(0), comma1);
        const std::string yText = text.substr(comma1 + size_t(1), comma2 - comma1 - size_t(1));
        const std::string zText = text.substr(comma2 + size_t(1));
        return walberla::Vector3<T>(
            parseStrictUintOrAbort("Invalid Vec3 CSV format in checkpoint metadata", xText),
            parseStrictUintOrAbort("Invalid Vec3 CSV format in checkpoint metadata", yText),
            parseStrictUintOrAbort("Invalid Vec3 CSV format in checkpoint metadata", zText));
    }
    else
    {
        std::istringstream iss(text);
        T x{};
        T y{};
        T z{};
        char c1 = '\0';
        char c2 = '\0';
        if (!(iss >> x >> c1 >> y >> c2 >> z) || c1 != ',' || c2 != ',')
            WALBERLA_ABORT("Invalid Vec3 CSV format in checkpoint metadata: '" << text << "'");
        iss >> std::ws;
        if (!iss.eof())
            WALBERLA_ABORT("Invalid Vec3 CSV format in checkpoint metadata: '" << text << "'");
        return walberla::Vector3<T>(x, y, z);
    }
}

static std::map<std::string, std::string> readCheckpointMetadata(const std::filesystem::path& file)
{
    std::ifstream in(file);
    if (!in)
        WALBERLA_ABORT("Failed to open checkpoint metadata file: " << file.string());

    std::map<std::string, std::string> data;
    std::string line;
    size_t lineNumber = size_t(0);
    while (std::getline(in, line))
    {
        ++lineNumber;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        if (line.front() == '#')
            continue;
        const auto pos = line.find('=');
        if (pos == std::string::npos)
        {
            WALBERLA_ABORT("Malformed checkpoint metadata line " << lineNumber << " in "
                           << file.string() << ": '" << line << "'");
        }

        const std::string key = line.substr(size_t(0), pos);
        if (key.empty())
        {
            WALBERLA_ABORT("Malformed checkpoint metadata line " << lineNumber << " in "
                           << file.string() << ": empty key");
        }

        const auto inserted = data.emplace(key, line.substr(pos + size_t(1)));
        if (!inserted.second)
        {
            WALBERLA_ABORT("Duplicate checkpoint metadata key '" << key << "' in "
                           << file.string());
        }
    }
    return data;
}

static std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

// CPU-only parallel-mode parser helpers.
static ParallelMode parseParallelMode(const std::string& arg)
{
    const std::string value = toUpper(arg);
    if (value == "OUTER")
        return ParallelMode::Outer;
    if (value == "INNER")
        return ParallelMode::Inner;
    if (value == "SERIAL")
        return ParallelMode::Serial;
    WALBERLA_ABORT("Unsupported --parallelMode value '" << arg << "'. Expected one of: outer, inner, serial.");
}

static const char* parallelModeToString(ParallelMode mode)
{
    switch (mode)
    {
    case ParallelMode::Outer:
        return "outer";
    case ParallelMode::Inner:
        return "inner";
    case ParallelMode::Serial:
        return "serial";
    }
    return "outer";
}

static std::string stripQuotes(std::string s)
{
    if (s.size() >= size_t(2))
    {
        const char first = s.front();
        const char last = s.back();
        if ((first == '\"' && last == '\"') || (first == '\'' && last == '\''))
            return s.substr(size_t(1), s.size() - size_t(2));
    }
    return s;
}

static void validateCheckpointMetadataStringFieldOrAbort(
    const std::string& fieldName,
    const std::string& value)
{
    if (value.find('|') != std::string::npos ||
        value.find('\n') != std::string::npos ||
        value.find('\r') != std::string::npos)
    {
        WALBERLA_ABORT(fieldName << " must not contain '|', '\\n', or '\\r' because it is written "
                       << "verbatim into checkpoint metadata. Got '" << value << "'");
    }
}

// CPU-only OpenMP sweep-thread query.
static uint_t getSweepThreadCount()
{
#ifdef _OPENMP
    return uint_t(std::max(1, omp_get_max_threads()));
#else
    return uint_t(1);
#endif
}

static std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct FileFingerprint
{
    std::uintmax_t bytes = std::uintmax_t(0);
    std::string hashHex;
};

static FileFingerprint computeFileFingerprint(const std::filesystem::path& filePath)
{
    std::ifstream in(filePath, std::ios::binary);
    if (!in)
        WALBERLA_ABORT("Failed to open file for fingerprinting: " << filePath.string());

    constexpr std::uint64_t fnvOffset = 1469598103934665603ULL;
    constexpr std::uint64_t fnvPrime = 1099511628211ULL;
    std::uint64_t h1 = fnvOffset;
    std::uint64_t h2 = 0x9e3779b97f4a7c15ULL;
    std::uintmax_t bytes = std::uintmax_t(0);

    constexpr size_t fingerprintChunkBytes = size_t(1u) << 20;
    std::vector<unsigned char> buffer(fingerprintChunkBytes, static_cast<unsigned char>(0));
    while (in)
    {
        in.read(reinterpret_cast<char*>(buffer.data()), std::streamsize(buffer.size()));
        const std::streamsize n = in.gcount();
        if (n <= std::streamsize(0))
            break;
        bytes += std::uintmax_t(n);
        for (std::streamsize i = std::streamsize(0); i < n; ++i)
        {
            const std::uint64_t b = std::uint64_t(buffer[size_t(i)]);
            h1 ^= b;
            h1 *= fnvPrime;
            h2 = splitmix64(h2 ^ (b + 0x9e3779b97f4a7c15ULL));
        }
    }
    if (!in.eof() && in.fail())
        WALBERLA_ABORT("Failed while fingerprinting file: " << filePath.string());

    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(16) << h1
        << std::setw(16) << h2;
    return FileFingerprint{bytes, oss.str()};
}

static double deterministicCenteredNoise(int gx, int gy, int gz)
{
    std::uint64_t key = 0xcbf29ce484222325ULL;
    key ^= std::uint64_t(std::uint32_t(gx));
    key *= 0x100000001b3ULL;
    key ^= std::uint64_t(std::uint32_t(gy));
    key *= 0x100000001b3ULL;
    key ^= std::uint64_t(std::uint32_t(gz));
    key *= 0x100000001b3ULL;
    const std::uint64_t h = splitmix64(key);
    constexpr double inv53 = 1.0 / 9007199254740992.0; // 2^53
    const double u01 = double(h >> 11) * inv53;
    return 2.0 * u01 - 1.0;
}

static CmdOptions parseArgs(int argc, char** argv)
{
    // Parse runtime flags supplied by launcher scripts (run_sim_*.sbatch / run_sim_local.sh).
    validatePrmArgOrAbort(argc, argv);
    CmdOptions cmd;
    const auto parseDoubleOrAbort = [](const std::string& flag, const char* rawValue) -> double {
        try
        {
            const std::string value(rawValue);
            size_t consumed = size_t(0);
            const double parsed = std::stod(value, &consumed);
            if (consumed != value.size())
                WALBERLA_ABORT(flag << " expects a numeric value, got '" << value << "'");
            if (!std::isfinite(parsed))
                WALBERLA_ABORT(flag << " expects a finite numeric value, got '" << value << "'");
            return parsed;
        }
        catch (const std::exception& ex)
        {
            WALBERLA_ABORT(flag << " expects a numeric value, got '" << rawValue
                               << "' (" << ex.what() << ")");
            return 0.0;
        }
    };
    const auto parseBoolOrAbort = [](const std::string& flag, const char* rawValue) -> bool {
        const std::string value = toLower(std::string(rawValue));
        if (value == "1" || value == "true")
            return true;
        if (value == "0" || value == "false")
            return false;
        WALBERLA_ABORT(flag << " expects one of: 0, 1, false, true (got '" << rawValue << "')");
        return false;
    };
    for (int i = 2; i < argc; ++i)
    {
        const std::string a(argv[i]);
        if (a.rfind("--", 0) != 0)
        {
            WALBERLA_ABORT("Unexpected positional argument: '" << a << "'");
        }
        if (a == "--timesteps")
        {
            if (i + 1 >= argc) WALBERLA_ABORT("--timesteps requires a positive integer");
            const auto parsed = parseStrictUintOrAbort("--timesteps", std::string(argv[++i]));
            if (parsed == uint_t(0))
                WALBERLA_ABORT("--timesteps must be > 0");
            cmd.timesteps = parsed;
        }
        else if (a == "--minimalLogs")
        {
            if (i + 1 >= argc) WALBERLA_ABORT("--minimalLogs requires an integer cadence");
            cmd.minimalLogsEvery = parseStrictUintOrAbort("--minimalLogs", std::string(argv[++i]));
        }
        else if (a == "--thermalLogs")
        {
            if (i + 1 >= argc) WALBERLA_ABORT("--thermalLogs requires an integer cadence");
            cmd.thermalLogsEvery = parseStrictUintOrAbort("--thermalLogs", std::string(argv[++i]));
        }
        else if (a == "--initPerturb")
        {
            if (i + 1 >= argc) WALBERLA_ABORT("--initPerturb requires a numeric amplitude");
            cmd.initPerturb = parseDoubleOrAbort("--initPerturb", argv[++i]);
            if (cmd.initPerturb < 0.0)
                WALBERLA_ABORT("--initPerturb amplitude must be >= 0");
        }
        else if (a == "--checkpointEvery")
        {
            if (i + 1 >= argc) WALBERLA_ABORT("--checkpointEvery requires an integer cadence");
            cmd.checkpointEvery = parseStrictUintOrAbort("--checkpointEvery", std::string(argv[++i]));
        }
        else if (a == "--vtkevery")
        {
            if (i + 1 >= argc) WALBERLA_ABORT("--vtkevery requires an integer cadence");
            cmd.vtkEvery = parseStrictUintOrAbort("--vtkevery", std::string(argv[++i]));
        }
        else if (a == "--vtkinit")
        {
            if (i + 1 >= argc) WALBERLA_ABORT("--vtkinit requires a boolean value (0/1/false/true)");
            cmd.vtkInit = parseBoolOrAbort("--vtkinit", argv[++i]);
        }
        else if (a == "--vtkmeshonly")
        {
            if (i + 1 >= argc) WALBERLA_ABORT("--vtkmeshonly requires a boolean value (0/1/false/true)");
            cmd.vtkMeshOnly = parseBoolOrAbort("--vtkmeshonly", argv[++i]);
        }
        else if (a == "--parallelMode")
        {
            if (i + 1 >= argc) WALBERLA_ABORT("--parallelMode requires one of: outer, inner, serial");
            cmd.parallelMode = parseParallelMode(argv[++i]);
        }
        else if (a.rfind("--", 0) == 0)
        {
            WALBERLA_ABORT("Unknown command-line option: '" << a << "'");
        }
    }
    if (cmd.timesteps == 0)
        WALBERLA_ABORT("--timesteps is required and must be > 0");

    return cmd;
}

static bool inDomainCell(const walberla::CellInterval& domainBB, int gx, int gy, int gz)
{
    return gx >= int(domainBB.xMin()) && gx <= int(domainBB.xMax()) && gy >= int(domainBB.yMin()) && gy <= int(domainBB.yMax()) &&
           gz >= int(domainBB.zMin()) && gz <= int(domainBB.zMax());
}

static bool isNeighborReachable(
    const walberla::CellInterval& domainBB,
    int ngx, int ngy, int ngz,
    bool periodicX, bool periodicY, bool periodicZ)
{
    const bool insideX = (ngx >= int(domainBB.xMin()) && ngx <= int(domainBB.xMax()));
    const bool insideY = (ngy >= int(domainBB.yMin()) && ngy <= int(domainBB.yMax()));
    const bool insideZ = (ngz >= int(domainBB.zMin()) && ngz <= int(domainBB.zMax()));
    return (insideX || periodicX) && (insideY || periodicY) && (insideZ || periodicZ);
}

template <typename CellTypeField_T>
static bool hasFaceNeighborTypeInDomain(
    const CellTypeField_T* cellType,
    const walberla::CellInterval& domainBB,
    int gx, int gy, int gz,
    int x, int y, int z,
    bool periodicX, bool periodicY, bool periodicZ,
    walberla::uint8_t neighborType)
{
    for (const auto& d : kFaceNbrDirs)
    {
        const int ngx = gx + d[0];
        const int ngy = gy + d[1];
        const int ngz = gz + d[2];
        if (!isNeighborReachable(domainBB, ngx, ngy, ngz, periodicX, periodicY, periodicZ))
            continue;
        if ((*cellType)(x + d[0], y + d[1], z + d[2], 0) == neighborType)
            return true;
    }
    return false;
}

template <typename CellTypeField_T, typename Stencil_T>
static bool hasStencilNeighborTypeInDomain(
    const CellTypeField_T* cellType,
    const walberla::CellInterval& domainBB,
    int gx, int gy, int gz,
    int x, int y, int z,
    bool periodicX, bool periodicY, bool periodicZ,
    walberla::uint8_t neighborType)
{
    for (uint_t qi = uint_t(0); qi < Stencil_T::Q; ++qi)
    {
        const auto dir = Stencil_T::dir[qi];
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
        if ((*cellType)(x + dx, y + dy, z + dz, 0) == neighborType)
            return true;
    }
    return false;
}

template <typename CellTypeField_T>
static walberla::uint8_t collectFluidNeighborDirIndicesInDomain(
    const CellTypeField_T* cellType,
    const walberla::CellInterval& domainBB,
    int gx, int gy, int gz,
    int x, int y, int z,
    bool periodicX, bool periodicY, bool periodicZ,
    std::array<walberla::uint8_t, kFaceNbrDirs.size()>& neighborDirIndices)
{
    walberla::uint8_t neighborCount = walberla::uint8_t(0);
    for (size_t dirIdx = size_t(0); dirIdx < kFaceNbrDirs.size(); ++dirIdx)
    {
        const auto& d = kFaceNbrDirs[dirIdx];
        const int ngx = gx + d[0];
        const int ngy = gy + d[1];
        const int ngz = gz + d[2];
        if (!isNeighborReachable(domainBB, ngx, ngy, ngz, periodicX, periodicY, periodicZ))
            continue;
        if ((*cellType)(x + d[0], y + d[1], z + d[2], 0) == CELL_FLUID)
            neighborDirIndices[size_t(neighborCount++)] = walberla::uint8_t(dirIdx);
    }
    return neighborCount;
}
#endif // FLUIDSIM_RUNTIME_ONLY

struct ThermalBoundaryCell
{
    int x = 0;
    int y = 0;
    int z = 0;
    walberla::uint16_t bcId = BC_NONE;
    walberla::uint16_t regionId = walberla::uint16_t(0);
    walberla::uint8_t thermalType = THERMAL_NONE;
    real_t thermalValue = real_t(0);
    std::array<walberla::uint8_t, kFaceNbrDirs.size()> fluidNeighborDirIndices{};
    walberla::uint8_t fluidNeighborCount = walberla::uint8_t(0);
};

struct ThermalBCBlockEntry
{
    walberla::IBlock* block = nullptr;
    const std::vector<ThermalBoundaryCell>* entries = nullptr;
};

using ThermalBoundaryCache = std::unordered_map<const walberla::IBlock*, std::vector<ThermalBoundaryCell>>;

#ifndef FLUIDSIM_RUNTIME_ONLY
template <typename CellTypeField_T, typename BcField_T, typename RegionIdField_T, typename ThermalTypeField_T, typename ThermalValueField_T>
static std::shared_ptr<ThermalBoundaryCache> buildThermalBoundaryCache(
    const std::shared_ptr<walberla::StructuredBlockForest>& blocks,
    walberla::BlockDataID cellTypeID,
    walberla::BlockDataID bcIdID,
    walberla::BlockDataID regionIdID,
    walberla::BlockDataID thermalTypeID,
    walberla::BlockDataID thermalValueID,
    bool periodicX, bool periodicY, bool periodicZ)
{
    auto cache = std::make_shared<ThermalBoundaryCache>();
    const auto domainBBL0 = blocks->getDomainCellBB(uint_t(0));
    for (auto& block : *blocks)
    {
        auto* cellType = block.getData<CellTypeField_T>(cellTypeID);
        auto* bcId = block.getData<BcField_T>(bcIdID);
        auto* regionId = block.getData<RegionIdField_T>(regionIdID);
        auto* thermalType = block.getData<ThermalTypeField_T>(thermalTypeID);
        auto* thermalValue = block.getData<ThermalValueField_T>(thermalValueID);
        const auto& domainBB = domainBBL0;
        const auto bb = blocks->getBlockCellBB(block);

        const int gx0 = int(bb.xMin());
        const int gy0 = int(bb.yMin());
        const int gz0 = int(bb.zMin());
        const int nx = int(bb.xSize());
        const int ny = int(bb.ySize());
        const int nz = int(bb.zSize());

        std::vector<ThermalBoundaryCell> entries;
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                {
                    if ((*cellType)(x, y, z, 0) != CELL_SOLID)
                        continue;

                    const auto localBcId = (*bcId)(x, y, z, 0);
                    const auto localThermalType = (*thermalType)(x, y, z, 0);
                    if (localBcId == BC_NONE || localThermalType == THERMAL_NONE)
                        continue;

                    const int gx = gx0 + x;
                    const int gy = gy0 + y;
                    const int gz = gz0 + z;
                    std::array<walberla::uint8_t, kFaceNbrDirs.size()> fluidNeighborDirIndices{};
                    const walberla::uint8_t fluidNeighborCount = collectFluidNeighborDirIndicesInDomain(
                        cellType, domainBB, gx, gy, gz, x, y, z, periodicX, periodicY, periodicZ, fluidNeighborDirIndices);

                    if (fluidNeighborCount == walberla::uint8_t(0))
                        continue;

                    entries.push_back(ThermalBoundaryCell{
                        x, y, z, localBcId, (*regionId)(x, y, z, 0), localThermalType, (*thermalValue)(x, y, z, 0), fluidNeighborDirIndices, fluidNeighborCount});
                }

        if (!entries.empty())
            cache->emplace(&block, std::move(entries));
    }
    return cache;
}
#endif // FLUIDSIM_RUNTIME_ONLY

} // namespace fluidsim
#ifndef FLUIDSIM_RUNTIME_ONLY
namespace fluidsim
{
struct PlyPropertyDecl
{
    bool isList = false;
    std::string name;
    std::string valueType;
    std::string listCountType;
    std::string listEntryType;
    std::string line;
};

struct PlyHeaderDecl
{
    bool isPly = false;
    bool binaryLittleEndian = false;
    bool sawEndHeader = false;
    bool vertexHasRed = false;
    bool vertexHasGreen = false;
    bool vertexHasBlue = false;
    bool faceHasRed = false;
    bool faceHasGreen = false;
    bool faceHasBlue = false;
    std::uint64_t vertexCount = 0;
    std::uint64_t faceCount = 0;
    std::vector<PlyPropertyDecl> vertexProperties;
    std::vector<PlyPropertyDecl> faceProperties;
    std::vector<std::string> headerLines;
    std::streamoff bodyOffset = std::streamoff(0);
    size_t facePropertyLineBegin = std::numeric_limits<size_t>::max();
    size_t facePropertyLineEnd = std::numeric_limits<size_t>::max();
    size_t vertexIndicesProperty = std::numeric_limits<size_t>::max();

    bool hasVertexRgb() const { return vertexHasRed && vertexHasGreen && vertexHasBlue; }
    bool hasFaceRgb() const { return faceHasRed && faceHasGreen && faceHasBlue; }
};

static void trimCarriageReturn(std::string& s)
{
    if (!s.empty() && s.back() == '\r')
        s.pop_back();
}

static size_t plyScalarByteWidth(const std::string& typeName)
{
    const std::string t = toLower(typeName);
    if (t == "char" || t == "int8" || t == "uchar" || t == "uint8")
        return size_t(1);
    if (t == "short" || t == "int16" || t == "ushort" || t == "uint16")
        return size_t(2);
    if (t == "int" || t == "int32" || t == "uint" || t == "uint32" || t == "float" || t == "float32")
        return size_t(4);
    if (t == "double" || t == "float64")
        return size_t(8);
    return size_t(0);
}

static bool decodeNonNegativeLittleEndianInteger(
    const char* rawData, size_t rawSize, const std::string& typeName, std::uint64_t& valueOut)
{
    const std::string t = toLower(typeName);
    valueOut = 0;
    if (rawData == nullptr || rawSize == size_t(0))
        return false;

    auto readUnsigned = [&]() {
        std::uint64_t v = 0;
        for (size_t i = size_t(0); i < rawSize; ++i)
            v |= (std::uint64_t(std::uint8_t(rawData[i])) << (8 * i));
        return v;
    };

    if (t == "uchar" || t == "uint8")
    {
        valueOut = std::uint64_t(std::uint8_t(rawData[0]));
        return true;
    }
    if (t == "char" || t == "int8")
    {
        const auto v = std::int8_t(rawData[0]);
        if (v < 0) return false;
        valueOut = std::uint64_t(v);
        return true;
    }
    if (t == "ushort" || t == "uint16")
    {
        if (rawSize != size_t(2)) return false;
        valueOut = readUnsigned();
        return true;
    }
    if (t == "short" || t == "int16")
    {
        if (rawSize != size_t(2)) return false;
        const std::uint16_t u = std::uint16_t(readUnsigned());
        std::int16_t v = 0;
        std::memcpy(&v, &u, sizeof(v));
        if (v < 0) return false;
        valueOut = std::uint64_t(v);
        return true;
    }
    if (t == "uint" || t == "uint32")
    {
        if (rawSize != size_t(4)) return false;
        valueOut = readUnsigned();
        return true;
    }
    if (t == "int" || t == "int32")
    {
        if (rawSize != size_t(4)) return false;
        const std::uint32_t u = std::uint32_t(readUnsigned());
        std::int32_t v = 0;
        std::memcpy(&v, &u, sizeof(v));
        if (v < 0) return false;
        valueOut = std::uint64_t(v);
        return true;
    }
    return false;
}

static bool parsePlyPropertyLine(const std::string& line, PlyPropertyDecl& property)
{
    std::istringstream iss(line);
    std::string keyword;
    if (!(iss >> keyword))
        return false;
    if (toLower(keyword) != "property")
        return false;

    std::string token;
    if (!(iss >> token))
        return false;

    property = PlyPropertyDecl{};
    property.line = line;
    if (toLower(token) == "list")
    {
        property.isList = true;
        if (!(iss >> property.listCountType >> property.listEntryType >> property.name))
            return false;
    }
    else
    {
        property.isList = false;
        property.valueType = token;
        if (!(iss >> property.name))
            return false;
    }
    return true;
}

static bool parsePlyHeaderDecl(const std::filesystem::path& path, PlyHeaderDecl& headerOut)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;

    enum class HeaderSection
    {
        Other,
        Vertex,
        Face
    };

    HeaderSection section = HeaderSection::Other;
    headerOut = PlyHeaderDecl{};

    std::string line;
    size_t lineIndex = size_t(0);
    while (std::getline(in, line))
    {
        trimCarriageReturn(line);
        headerOut.headerLines.push_back(line);

        if (lineIndex == size_t(0))
            headerOut.isPly = (toLower(line) == "ply");

        std::istringstream iss(line);
        std::string keyword;
        if (!(iss >> keyword))
        {
            ++lineIndex;
            continue;
        }

        const std::string keywordLower = toLower(keyword);
        if (keywordLower == "format")
        {
            std::string formatName;
            if (iss >> formatName)
                headerOut.binaryLittleEndian = (toLower(formatName) == "binary_little_endian");
        }
        else if (keywordLower == "element")
        {
            std::string elementName;
            std::uint64_t count = 0;
            if (!(iss >> elementName >> count))
                return false;
            const std::string e = toLower(elementName);
            if (e == "vertex")
            {
                section = HeaderSection::Vertex;
                headerOut.vertexCount = count;
            }
            else if (e == "face")
            {
                section = HeaderSection::Face;
                headerOut.faceCount = count;
            }
            else
            {
                section = HeaderSection::Other;
            }
        }
        else if (keywordLower == "property")
        {
            PlyPropertyDecl property;
            if (!parsePlyPropertyLine(line, property))
                return false;

            if (section == HeaderSection::Vertex)
            {
                if (property.isList)
                    return false;
                headerOut.vertexProperties.push_back(property);
                const std::string propName = toLower(property.name);
                if (propName == "red") headerOut.vertexHasRed = true;
                if (propName == "green") headerOut.vertexHasGreen = true;
                if (propName == "blue") headerOut.vertexHasBlue = true;
            }
            else if (section == HeaderSection::Face)
            {
                if (headerOut.facePropertyLineBegin == std::numeric_limits<size_t>::max())
                    headerOut.facePropertyLineBegin = lineIndex;
                headerOut.facePropertyLineEnd = lineIndex + size_t(1);
                const std::string propName = toLower(property.name);
                if (propName == "red") headerOut.faceHasRed = true;
                if (propName == "green") headerOut.faceHasGreen = true;
                if (propName == "blue") headerOut.faceHasBlue = true;
                if (property.isList)
                {
                    if ((propName == "vertex_indices" || propName == "vertex_index") &&
                        headerOut.vertexIndicesProperty == std::numeric_limits<size_t>::max())
                    {
                        headerOut.vertexIndicesProperty = headerOut.faceProperties.size();
                    }
                }
                headerOut.faceProperties.push_back(property);
            }
        }
        else if (keywordLower == "end_header")
        {
            headerOut.sawEndHeader = true;
            headerOut.bodyOffset = in.tellg();
            break;
        }

        ++lineIndex;
    }

    if (!headerOut.sawEndHeader)
        return false;
    if (!headerOut.isPly)
        return false;
    if (!headerOut.binaryLittleEndian)
        return false;
    if (headerOut.vertexCount == 0 || headerOut.faceCount == 0)
        return false;
    if (headerOut.vertexIndicesProperty == std::numeric_limits<size_t>::max())
        return false;

    return true;
}

static bool copyRawBytes(std::istream& in, std::ostream& out, std::uint64_t bytesToCopy)
{
    constexpr size_t kChunkSize = size_t(1) << 20;
    std::vector<char> buffer(kChunkSize);
    std::uint64_t remaining = bytesToCopy;
    while (remaining > 0)
    {
        const size_t chunk = size_t(std::min<std::uint64_t>(remaining, std::uint64_t(buffer.size())));
        if (!in.read(buffer.data(), std::streamsize(chunk)))
            return false;
        out.write(buffer.data(), std::streamsize(chunk));
        if (!out)
            return false;
        remaining -= std::uint64_t(chunk);
    }
    return true;
}

static std::filesystem::path compatMeshDirectoryForSource(const std::filesystem::path& sourcePath)
{
    const auto sourceDir = sourcePath.parent_path();
    return sourceDir / "geometry_compat";
}

static std::filesystem::path maybeCreateOpenMeshCompatiblePly(
    const std::filesystem::path& sourcePath,
    const std::filesystem::path& compatDir,
    bool& convertedOut)
{
    convertedOut = false;
    const std::string ext = toLower(sourcePath.extension().string());
    if (ext != ".ply")
        return sourcePath;

    PlyHeaderDecl header;
    if (!parsePlyHeaderDecl(sourcePath, header))
        return sourcePath;

    const size_t listIndex = header.vertexIndicesProperty;
    if (listIndex == size_t(0))
        return sourcePath;

    ensureDirectory(compatDir, "create mesh compatibility directory");

    const std::filesystem::path convertedPath =
        compatDir / (sourcePath.stem().string() + "_openmesh_compat.ply");

    auto convertedPathIsUsable = [&]() -> bool {
        std::error_code ec;
        if (!std::filesystem::exists(convertedPath, ec) || ec)
            return false;

        PlyHeaderDecl convertedHeader;
        if (!parsePlyHeaderDecl(convertedPath, convertedHeader))
            return false;
        if (convertedHeader.vertexCount != header.vertexCount || convertedHeader.faceCount != header.faceCount)
            return false;

        std::error_code srcTimeEc;
        const auto srcTime = std::filesystem::last_write_time(sourcePath, srcTimeEc);
        std::error_code convTimeEc;
        const auto convTime = std::filesystem::last_write_time(convertedPath, convTimeEc);
        if (srcTimeEc || convTimeEc)
            return false;

        return convTime >= srcTime;
    };

    if (convertedPathIsUsable())
    {
        convertedOut = true;
        return convertedPath;
    }

    const std::uint64_t tmpTag = std::uint64_t(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const std::filesystem::path tempPath =
        compatDir / (sourcePath.stem().string() + "_openmesh_compat.ply.tmp." + std::to_string(tmpTag));

    std::ifstream in(sourcePath, std::ios::binary);
    if (!in)
        WALBERLA_ABORT("Failed to open source PLY for conversion: " << sourcePath.string());
    in.seekg(header.bodyOffset);

    std::ofstream out(tempPath, std::ios::binary | std::ios::trunc);
    if (!out)
        WALBERLA_ABORT("Failed to create converted PLY temp file: " << tempPath.string());

    for (size_t i = size_t(0); i < header.headerLines.size();)
    {
        if (i == header.facePropertyLineBegin &&
            header.facePropertyLineBegin != std::numeric_limits<size_t>::max())
        {
            out << header.faceProperties[listIndex].line << '\n';
            for (size_t p = size_t(0); p < header.faceProperties.size(); ++p)
            {
                if (p == listIndex) continue;
                out << header.faceProperties[p].line << '\n';
            }
            i = header.facePropertyLineEnd;
            continue;
        }
        out << header.headerLines[i] << '\n';
        ++i;
    }

    std::uint64_t vertexStride = 0;
    for (const auto& prop : header.vertexProperties)
    {
        const size_t width = plyScalarByteWidth(prop.valueType);
        if (width == size_t(0))
            WALBERLA_ABORT("Unsupported vertex property type in PLY conversion: " << prop.valueType);
        vertexStride += std::uint64_t(width);
    }

    if (!copyRawBytes(in, out, header.vertexCount * vertexStride))
        WALBERLA_ABORT("Failed while copying vertex payload during PLY conversion: " << sourcePath.string());

    std::vector<std::vector<char>> rawFields(header.faceProperties.size());
    for (std::uint64_t face = std::uint64_t(0); face < header.faceCount; ++face)
    {
        for (size_t p = size_t(0); p < header.faceProperties.size(); ++p)
        {
            const auto& prop = header.faceProperties[p];
            if (!prop.isList)
            {
                const size_t width = plyScalarByteWidth(prop.valueType);
                if (width == size_t(0))
                    WALBERLA_ABORT("Unsupported face scalar property type in PLY conversion: " << prop.valueType);
                rawFields[p].resize(width);
                if (!in.read(rawFields[p].data(), std::streamsize(width)))
                    WALBERLA_ABORT("Unexpected end of file while reading face scalar property during PLY conversion: "
                                   << sourcePath.string());
            }
            else
            {
                const size_t countWidth = plyScalarByteWidth(prop.listCountType);
                const size_t entryWidth = plyScalarByteWidth(prop.listEntryType);
                if (countWidth == size_t(0) || entryWidth == size_t(0))
                    WALBERLA_ABORT("Unsupported list property type in PLY conversion: count="
                                   << prop.listCountType << " entry=" << prop.listEntryType);

                std::array<char, size_t(8)> countRawBuf{};
                WALBERLA_CHECK_LESS_EQUAL(countWidth, countRawBuf.size());
                if (!in.read(countRawBuf.data(), std::streamsize(countWidth)))
                    WALBERLA_ABORT("Unexpected end of file while reading face list count during PLY conversion: "
                                   << sourcePath.string());

                std::uint64_t entryCount = 0;
                if (!decodeNonNegativeLittleEndianInteger(countRawBuf.data(), countWidth, prop.listCountType, entryCount))
                    WALBERLA_ABORT("Invalid negative or unsupported face list count type in PLY conversion: "
                                   << prop.listCountType);
                if (p == listIndex && entryCount != std::uint64_t(3))
                {
                    WALBERLA_ABORT("TriangleMesh requires faces with 3 vertex indices; got "
                                   << entryCount << " at face " << face
                                   << " in " << sourcePath.string());
                }

                if (entryWidth != 0 &&
                    entryCount > (std::numeric_limits<std::uint64_t>::max() / std::uint64_t(entryWidth)))
                {
                    WALBERLA_ABORT("Face list payload size overflow during PLY conversion.");
                }
                const std::uint64_t payloadBytes64 = entryCount * std::uint64_t(entryWidth);
                if (payloadBytes64 > std::uint64_t(std::numeric_limits<size_t>::max()))
                    WALBERLA_ABORT("Face list payload too large during PLY conversion.");
                const size_t payloadBytes = size_t(payloadBytes64);

                rawFields[p].resize(countWidth + payloadBytes);
                std::memcpy(rawFields[p].data(), countRawBuf.data(), countWidth);
                if (payloadBytes > size_t(0))
                {
                    if (!in.read(rawFields[p].data() + countWidth, std::streamsize(payloadBytes)))
                        WALBERLA_ABORT("Unexpected end of file while reading face list payload during PLY conversion: "
                                       << sourcePath.string());
                }
            }
        }

        out.write(rawFields[listIndex].data(), std::streamsize(rawFields[listIndex].size()));
        for (size_t p = size_t(0); p < rawFields.size(); ++p)
        {
            if (p == listIndex) continue;
            out.write(rawFields[p].data(), std::streamsize(rawFields[p].size()));
        }
    }

    if (in.peek() != std::char_traits<char>::eof())
    {
        out << in.rdbuf();
        if (!out)
            WALBERLA_ABORT("Failed while writing converted PLY temp file: " << tempPath.string());
    }
    out.flush();
    if (!out)
        WALBERLA_ABORT("Failed while finalizing converted PLY temp file: " << tempPath.string());

    out.close();
    if (!out)
        WALBERLA_ABORT("Failed while closing converted PLY temp file: " << tempPath.string());

    std::error_code renameEc;
    std::filesystem::rename(tempPath, convertedPath, renameEc);
    if (renameEc)
    {
        if (convertedPathIsUsable())
        {
            std::error_code removeEc;
            std::filesystem::remove(tempPath, removeEc);
            convertedOut = true;
            return convertedPath;
        }
        WALBERLA_ABORT("Failed to atomically finalize converted PLY file: "
                       << tempPath.string() << " -> " << convertedPath.string()
                       << " (" << renameEc.message() << ")");
    }

    convertedOut = true;
    return convertedPath;
}

static walberla::uint16_t bcIdFromRegionName(const std::string& regionName)
{
    // Region-name contract is PREFIX + positive numeric suffix, e.g. DIRICHLET1.
    // Bare prefixes (e.g. DIRICHLET) and zero-only suffixes (e.g. DIRICHLET0)
    // are rejected to keep region identifiers explicit.
    const std::string name = toUpper(regionName);
    auto hasPositiveNumericSuffix = [&](const std::string& prefix) {
        if (name.rfind(prefix, size_t(0)) != size_t(0))
            return false;
        if (name.size() <= prefix.size())
            return false;
        const std::string suffix = name.substr(prefix.size());
        if (!std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
            return false;
        return std::any_of(suffix.begin(), suffix.end(), [](char c) { return c != '0'; });
    };

    if (hasPositiveNumericSuffix("DIRICHLET")) return BC_DIRICHLET;
    if (hasPositiveNumericSuffix("ADIABATIC")) return BC_ADIABATIC;
    if (hasPositiveNumericSuffix("HEATLOAD")) return BC_HEATLOAD;
    if (hasPositiveNumericSuffix("INLET")) return BC_INLET;
    if (hasPositiveNumericSuffix("OUTLET")) return BC_OUTLET;
    if (hasPositiveNumericSuffix("PRESSURE")) return BC_PRESSURE;
    return BC_NONE;
}

static walberla::uint8_t thermalTypeFromString(const std::string& thermalString)
{
    const std::string thermal = toUpper(stripQuotes(thermalString));
    if (thermal == "DIRICHLET")
        return THERMAL_DIRICHLET;
    if (thermal == "ADIABATIC" || thermal == "ADIAB")
        return THERMAL_ADIABATIC;
    if (thermal == "HEATLOAD")
        return THERMAL_HEATLOAD;
    WALBERLA_ABORT("Unsupported ColorBC.Region thermal type '" << thermalString
                   << "'. Expected Dirichlet, Adiabatic, or Heatload.");
}

static walberla::uint8_t pressureFlowModeFromString(const std::string& flowString)
{
    const std::string flow = toUpper(stripQuotes(flowString));
    if (flow == "IN")
        return PRESSURE_FLOW_IN;
    if (flow == "OUT")
        return PRESSURE_FLOW_OUT;
    WALBERLA_ABORT("Unsupported ColorBC.Region flow '" << flowString
                   << "'. Expected in or out.");
}

static GeometryRole geometryRoleFromString(const std::string& roleString)
{
    const std::string role = toUpper(stripQuotes(roleString));
    if (role == "FLUID_CONTAINER")
        return GeometryRole::FluidContainer;
    if (role == "SOLID_OBSTACLE")
        return GeometryRole::SolidObstacle;
    WALBERLA_ABORT("Unsupported MeshGeometry.Region role '" << roleString
                   << "'. Expected FLUID_CONTAINER or SOLID_OBSTACLE.");
}

// Resolve the parent of `params` from argv[1], typically `<apps>/shared/params/<prm-file>`.
// This is a path contract helper (not a generic "current working tree" resolver).
static std::filesystem::path resolveAppRootPath(int argc, char** argv)
{
    validatePrmArgOrAbort(argc, argv);

    const std::filesystem::path prmPath = std::filesystem::absolute(std::filesystem::path(argv[1]));
    if (!prmPath.has_parent_path())
        WALBERLA_ABORT("Could not resolve parameter file parent directory: " << prmPath.string());

    const auto prmParent = prmPath.parent_path();
    if (prmParent.filename() != "params" || !prmParent.has_parent_path())
    {
        WALBERLA_ABORT("Expected parameter file inside a 'params' directory "
                       << "(recommended: '<apps>/shared/params/'). Got: " << prmPath.string());
    }
    return prmParent.parent_path();
}

static std::filesystem::path resolveMeshPath(const std::string& meshFile, int argc, char** argv)
{
    std::filesystem::path meshPath(stripQuotes(meshFile));
    if (meshPath.is_relative())
    {
        meshPath = resolveAppRootPath(argc, argv) / meshPath;
    }
    return std::filesystem::absolute(meshPath);
}

template <typename MeshType>
static walberla::math::AABB computeMeshAabb(const MeshType& mesh)
{
    WALBERLA_CHECK_GREATER(mesh.n_vertices(), size_t(0));

    auto vIt = mesh.vertices_begin();
    auto p = mesh.point(*vIt);

    real_t xMin = real_t(p[0]);
    real_t yMin = real_t(p[1]);
    real_t zMin = real_t(p[2]);
    real_t xMax = xMin;
    real_t yMax = yMin;
    real_t zMax = zMin;

    ++vIt;
    for (; vIt != mesh.vertices_end(); ++vIt)
    {
        p = mesh.point(*vIt);
        const real_t x = real_t(p[0]);
        const real_t y = real_t(p[1]);
        const real_t z = real_t(p[2]);

        xMin = std::min(xMin, x);
        yMin = std::min(yMin, y);
        zMin = std::min(zMin, z);
        xMax = std::max(xMax, x);
        yMax = std::max(yMax, y);
        zMax = std::max(zMax, z);
    }

    return walberla::math::AABB(xMin, yMin, zMin, xMax, yMax, zMax);
}

template <typename MeshType>
static void vertexToFaceColor(MeshType& mesh, const typename MeshType::Color& defaultColor)
{
    WALBERLA_CHECK(mesh.has_vertex_colors());
    mesh.request_face_colors();

    for (auto faceIt = mesh.faces_begin(); faceIt != mesh.faces_end(); ++faceIt)
    {
        typename MeshType::Color vertexColor;
        bool useVertexColor = true;

        auto vertexIt = mesh.fv_iter(*faceIt);
        if (!vertexIt.is_valid())
        {
            WALBERLA_ABORT("Mesh face has no vertices while converting vertex colors to face colors.");
        }

        vertexColor = mesh.color(*vertexIt);
        ++vertexIt;
        while (vertexIt.is_valid() && useVertexColor)
        {
            if (vertexColor != mesh.color(*vertexIt)) useVertexColor = false;
            ++vertexIt;
        }

        mesh.set_color(*faceIt, useVertexColor ? vertexColor : defaultColor);
    }
}

} // namespace fluidsim
#endif // FLUIDSIM_RUNTIME_ONLY



namespace fluidsim
{
template <typename ScalarField_T>
struct SwapTheta
{
    walberla::BlockDataID thetaID;
    walberla::BlockDataID thetaTmpID;

    void operator()(walberla::IBlock* block)
    {
        auto* theta = block->getData<ScalarField_T>(thetaID);
        auto* thetaTmp = block->getData<ScalarField_T>(thetaTmpID);
        theta->swapDataPointers(thetaTmp);
    }
};

template <typename ScalarField_T, typename CellTypeField_T, typename ThermalTypeField_T, typename ThermalValueField_T>
static void initializeThetaFromCellTypes(
    const std::shared_ptr<walberla::StructuredBlockForest>& blocks,
    walberla::BlockDataID thetaID,
    walberla::BlockDataID cellTypeID,
    walberla::BlockDataID thermalTypeID,
    walberla::BlockDataID thermalValueID,
    real_t thetaInit)
{
    for (auto& b : *blocks)
    {
        auto* theta = b.getData<ScalarField_T>(thetaID);
        auto* cellType = b.getData<CellTypeField_T>(cellTypeID);
        auto* thermalType = b.getData<ThermalTypeField_T>(thermalTypeID);
        auto* thermalValue = b.getData<ThermalValueField_T>(thermalValueID);
        const auto ci = theta->xyzSize();

        for (auto cell = ci.begin(); cell != ci.end(); ++cell)
        {
            const int x = cell->x();
            const int y = cell->y();
            const int z = cell->z();

            if ((*cellType)(x, y, z, 0) == CELL_FLUID)
            {
                (*theta)(x, y, z, 0) = thetaInit;
                continue;
            }

            if ((*thermalType)(x, y, z, 0) == THERMAL_DIRICHLET)
                (*theta)(x, y, z, 0) = (*thermalValue)(x, y, z, 0);
            else
                (*theta)(x, y, z, 0) = thetaInit;
        }
    }
}
} // namespace fluidsim
