// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "nau/nau_editor_config.hpp"

#include <fmt/base.h>

#include <QString>

#include <string>


// ** NauObjectGUID

class NAU_EDITOR_API NauObjectGUID
{
public:
    NauObjectGUID();
    NauObjectGUID(uint64_t guid);

    static NauObjectGUID invalid() { return NauObjectGUID(0); }

    operator uint64_t() const { return m_id; }
    operator QString() const { return QString::number(static_cast<qulonglong>(m_id), 16); };
    operator std::string() const { return static_cast<QString>(*this).toUtf8().data(); };

private:
    uint64_t m_id;
};

namespace std
{
    template<>
    struct std::hash<NauObjectGUID>
    {
        std::size_t operator()(const NauObjectGUID& guid) const
        {
            return static_cast<uint64_t>(guid);
        }
    };
}

template<>
struct fmt::formatter<NauObjectGUID, char8_t> : fmt::formatter<const char8_t*, char8_t>
{
    template <typename FormatContext>
    auto format(const NauObjectGUID& input, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "'{}'", static_cast<uint64_t>(input));
    }
};
