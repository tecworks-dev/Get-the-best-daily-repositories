// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Convenient cast of typed enums to underlying type.

#pragma once


template <typename T>
constexpr std::enable_if_t<std::is_enum<T>::value, typename std::underlying_type_t<T>>
operator+(T val)
{
    return static_cast<std::underlying_type_t<T>>(val);
}
