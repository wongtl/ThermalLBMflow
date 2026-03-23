// SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "core/Abort.h"
#include "core/mpi/Broadcast.h"
#include "mesh_common/MeshIO.h"

#include <filesystem>
#include <fstream>
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

        std::ifstream stream;
        if (binaryFile)
            stream.open(filenameStr.c_str(), std::ifstream::in | std::ifstream::binary);
        else
            stream.open(filenameStr.c_str(), std::ifstream::in);

        if (!stream)
            WALBERLA_ABORT("Error while reading file \"" << filenameStr << "\"!");

        stream.seekg(0, std::ios::end);
        payload.reserve(static_cast<std::string::size_type>(stream.tellg()));
        stream.seekg(0, std::ios::beg);
        payload.assign((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
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
