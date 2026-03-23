// SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "core/Abort.h"
#include "core/mpi/Broadcast.h"
#include "mesh_common/MeshIO.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace fluidsim::appsupport
{

template <typename MeshType>
inline void readMeshFromRootAndBroadcast(
    const std::filesystem::path& filename,
    MeshType& mesh,
    bool binaryFile = false)
{
    const std::string filenameStr = filename.string();
    const std::string extension = filename.extension().string();
    std::string payload;

    WALBERLA_ROOT_SECTION()
    {
        if (!std::filesystem::exists(filename))
            WALBERLA_ABORT("The mesh file \"" << filenameStr << "\" does not exist!");

        std::error_code fileSizeEc;
        const auto fileSize = std::filesystem::file_size(filename, fileSizeEc);
        if (fileSizeEc)
            WALBERLA_ABORT("Error while reading file \"" << filenameStr << "\": " << fileSizeEc.message());

        if (fileSize > std::uintmax_t(std::numeric_limits<std::string::size_type>::max()))
            WALBERLA_ABORT("The mesh file \"" << filenameStr << "\" is too large to read into memory.");

        if (fileSize > std::uintmax_t(std::numeric_limits<std::streamsize>::max()))
            WALBERLA_ABORT("The mesh file \"" << filenameStr << "\" is too large to read through std::ifstream.");

        std::ifstream stream(filenameStr.c_str(), std::ifstream::in | std::ifstream::binary);

        if (!stream)
            WALBERLA_ABORT("Error while reading file \"" << filenameStr << "\"!");

        payload.resize(static_cast<std::string::size_type>(fileSize));

        if (!payload.empty())
        {
            stream.read(payload.data(), static_cast<std::streamsize>(payload.size()));
            if (!stream || stream.gcount() != static_cast<std::streamsize>(payload.size()))
                WALBERLA_ABORT("Error while reading file \"" << filenameStr << "\"!");
        }
    }

    walberla::mpi::broadcastObject(payload);

    std::istringstream iss;
    if (binaryFile)
        iss = std::istringstream(payload, std::ifstream::in | std::ifstream::binary);
    else
        iss = std::istringstream(payload, std::ifstream::in);

    if (!walberla::mesh::readFromStream<MeshType>(iss, mesh, extension, binaryFile))
        WALBERLA_ABORT("Error while reading file \"" << filenameStr << "\"!");
}

} // namespace fluidsim::appsupport
