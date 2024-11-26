// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// String wrapper


#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include <QString>

// ** NauString

class NAU_EDITOR_ENGINE_API NauString : public std::string
{
public:
    NauString() noexcept = default;
    ~NauString() noexcept = default;

    // Copy constructors
    NauString(const std::string& rhs);
    NauString(const NauString& rhs);
    NauString(const QString& rhs);
    NauString(const char* rhs);

    // Move constructor
    NauString(NauString&& rhs) noexcept;
    NauString(std::string&& rhs) noexcept;

    // Copy-assignment operators
    NauString& operator = (const std::string& rhs);
    NauString& operator = (const NauString& rhs);
    NauString& operator = (const QString& rhs);
    NauString& operator = (const char* rhs);

    // Move-assignment operators
    NauString& operator = (NauString&& rhs) noexcept;
    NauString& operator = (std::string&& rhs) noexcept;

    operator QString() const;
};

NAU_EDITOR_ENGINE_API NauString operator + ( const NauString& lhs, const NauString& rhs );
NAU_EDITOR_ENGINE_API NauString operator + ( const NauString& lhs, const std::string& rhs );
NAU_EDITOR_ENGINE_API NauString operator + ( const std::string& lhs, const NauString& rhs );
NAU_EDITOR_ENGINE_API NauString operator + ( const NauString& lhs, const char* rhs );
NAU_EDITOR_ENGINE_API NauString operator + ( const char* lhs, const NauString& rhs );
