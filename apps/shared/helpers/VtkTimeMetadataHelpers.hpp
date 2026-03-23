// SPDX-FileCopyrightText: 2026 David Wong, University of Oxford
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "core/Abort.h"

#include <charconv>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fluidsim::appsupport
{

inline std::optional<unsigned long long> parseCollectorFromPathToken(const std::string& pathToken)
{
    const size_t underscore = pathToken.rfind('_');
    const size_t dot = pathToken.rfind('.');
    if (underscore == std::string::npos || dot == std::string::npos || dot <= underscore + size_t(1))
        return std::nullopt;

    const std::string numberToken = pathToken.substr(underscore + size_t(1), dot - underscore - size_t(1));
    if (numberToken.empty())
        return std::nullopt;
    for (char c : numberToken)
    {
        if (c < '0' || c > '9')
            return std::nullopt;
    }

    unsigned long long collector = 0;
    const char* begin = numberToken.data();
    const char* end = begin + numberToken.size();
    const auto parseResult = std::from_chars(begin, end, collector);
    if (parseResult.ec != std::errc() || parseResult.ptr != end)
        return std::nullopt;

    return collector;
}

inline std::string formatRealForVtkSeries(double value)
{
    std::ostringstream os;
    os << std::setprecision(17) << value;
    return os.str();
}

inline void rescalePvdTimes(const std::filesystem::path& pvdPath, double scale, double offset)
{
    std::ifstream in(pvdPath);
    if (!in)
        return;

    std::vector<std::string> lines;
    std::string line;
    bool changed = false;
    while (std::getline(in, line))
    {
        if (line.find("<DataSet") != std::string::npos)
        {
            const size_t fileKeyPos = line.find("file=\"");
            const size_t timeKeyPos = line.find("timestep=\"");
            if (fileKeyPos != std::string::npos && timeKeyPos != std::string::npos)
            {
                const size_t fileValueStart = fileKeyPos + size_t(6);
                const size_t fileValueEnd = line.find('"', fileValueStart);
                if (fileValueEnd != std::string::npos)
                {
                    const std::string fileToken = line.substr(fileValueStart, fileValueEnd - fileValueStart);
                    const auto collector = parseCollectorFromPathToken(fileToken);
                    if (collector.has_value())
                    {
                        const size_t timeValueStart = timeKeyPos + size_t(10);
                        const size_t timeValueEnd = line.find('"', timeValueStart);
                        if (timeValueEnd != std::string::npos)
                        {
                            const double physicalTime = scale * double(*collector) + offset;
                            line.replace(
                                timeValueStart,
                                timeValueEnd - timeValueStart,
                                formatRealForVtkSeries(physicalTime));
                            changed = true;
                        }
                    }
                }
            }
        }
        lines.push_back(std::move(line));
    }
    in.close();

    if (!changed)
        return;

    std::ofstream out(pvdPath, std::ios::trunc);
    for (const auto& l : lines)
        out << l << '\n';
}

inline void rescaleVthbSeriesTimes(const std::filesystem::path& seriesPath, double scale, double offset)
{
    std::ifstream in(seriesPath);
    if (!in)
        return;

    std::vector<std::string> lines;
    std::string line;
    bool changed = false;
    while (std::getline(in, line))
    {
        const size_t nameKeyPos = line.find("\"name\"");
        const size_t timeKeyPos = line.find("\"time\"");
        if (nameKeyPos != std::string::npos && timeKeyPos != std::string::npos)
        {
            const size_t nameColon = line.find(':', nameKeyPos);
            if (nameColon != std::string::npos)
            {
                const size_t nameValueStartQuote = line.find('"', nameColon + size_t(1));
                if (nameValueStartQuote != std::string::npos)
                {
                    const size_t nameValueEndQuote = line.find('"', nameValueStartQuote + size_t(1));
                    if (nameValueEndQuote != std::string::npos)
                    {
                        const std::string nameToken =
                            line.substr(nameValueStartQuote + size_t(1), nameValueEndQuote - nameValueStartQuote - size_t(1));
                        const auto collector = parseCollectorFromPathToken(nameToken);
                        if (collector.has_value())
                        {
                            const size_t timeColon = line.find(':', timeKeyPos);
                            if (timeColon != std::string::npos)
                            {
                                size_t timeValueStart = timeColon + size_t(1);
                                while (timeValueStart < line.size() &&
                                       std::isspace(static_cast<unsigned char>(line[timeValueStart])) != 0)
                                    ++timeValueStart;
                                const size_t timeValueEnd = line.find_first_of(",}", timeValueStart);
                                if (timeValueEnd != std::string::npos)
                                {
                                    const double physicalTime = scale * double(*collector) + offset;
                                    line.replace(
                                        timeValueStart,
                                        timeValueEnd - timeValueStart,
                                        formatRealForVtkSeries(physicalTime));
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        lines.push_back(std::move(line));
    }
    in.close();

    if (!changed)
        return;

    std::ofstream out(seriesPath, std::ios::trunc);
    for (const auto& l : lines)
        out << l << '\n';
}

inline void rescaleVtkTimeMetadata(const std::string& baseFolder, const std::string& identifier, double scale, double offset = 0.0)
{
    if (!std::isfinite(scale) || !std::isfinite(offset))
        WALBERLA_ABORT("rescaleVtkTimeMetadata requires finite scale and offset.");

    rescalePvdTimes(std::filesystem::path(baseFolder) / (identifier + ".pvd"), scale, offset);
    rescaleVthbSeriesTimes(std::filesystem::path(baseFolder) / (identifier + ".vthb.series"), scale, offset);
}

} // namespace fluidsim::appsupport
