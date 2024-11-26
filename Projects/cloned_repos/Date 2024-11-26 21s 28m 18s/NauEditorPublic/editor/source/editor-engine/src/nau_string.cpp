// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/nau_string.hpp"

// ** NauString

NauString::NauString(const std::string& rhs)
    : std::string(rhs)
{}

NauString::NauString(const NauString& rhs)
    : std::string(rhs)
{}

NauString::NauString(const QString& rhs)
    : std::string(rhs.toUtf8().constData())
{}

NauString::NauString(const char* rhs)
    : std::string(rhs)
{}

NauString::NauString(NauString&& rhs) noexcept
    : std::string(static_cast<std::string&&>(rhs))
{}

NauString::NauString(std::string&& rhs) noexcept
    : std::string(std::move(rhs))
{}

NauString& NauString::operator = (const std::string& rhs)
{
    if (this != &rhs) {
        std::string& string = *this;
        string = rhs;
    }
    return *this;
}

NauString& NauString::operator = (const NauString& rhs)
{
    if (this != &rhs) {
        std::string& string = *this;
        string = rhs;
    }
    return *this;
}

NauString& NauString::operator = (const QString& rhs)
{
    std::string& string = *this;
    string = rhs.toUtf8().constData();
    return *this;
}

NauString& NauString::operator = (const char* rhs)
{
    std::string& string = *this;
    string = rhs;
    return *this;
}

NauString& NauString::operator = (NauString&& rhs) noexcept
{
    if (this != &rhs) {
        std::string& string = *this;
        string = static_cast<std::string&&>(rhs);
    }
    return *this;
}

NauString& NauString::operator = (std::string&& rhs) noexcept
{
    if (this != &rhs) {
        std::string& string = *this;
        string = std::move(rhs);
    }
    return *this;
}

NauString::operator QString() const
{
    return { data() };
}

NauString operator + (const NauString& lhs, const NauString& rhs)
{
    return static_cast< const std::string& >( lhs ) + static_cast< const std::string& >( rhs );
}

NauString operator + (const NauString& lhs, const std::string& rhs)
{
    return static_cast< const std::string& >( lhs ) + rhs;
}

NauString operator + (const std::string& lhs, const NauString& rhs)
{
    return lhs + static_cast< const std::string& >( rhs );
}

NauString operator + (const NauString& lhs, const char* rhs)
{
    return static_cast< const std::string& >( lhs ) + rhs;
}

NauString operator + (const char* lhs, const NauString& rhs)
{
    return lhs + static_cast< const std::string& >( rhs );
}
