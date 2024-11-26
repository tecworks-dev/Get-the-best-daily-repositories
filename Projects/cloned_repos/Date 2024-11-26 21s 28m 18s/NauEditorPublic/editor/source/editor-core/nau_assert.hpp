// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Wrappers for assert

#pragma once

#include <cassert>
#include <utility>


// TODO: use __builtin_expect to build our own assert
// Use it to show backtraces in log files
#define NED_ASSERT(...) assert(__VA_ARGS__)
#define NED_PASSTHROUGH_ASSERT(ARG) ned_passthrough_assert(ARG)


template <typename T>
decltype(auto) ned_passthrough_assert(T&& expression)
{
    NED_ASSERT(expression);
    return std::forward<T>(expression);
}
