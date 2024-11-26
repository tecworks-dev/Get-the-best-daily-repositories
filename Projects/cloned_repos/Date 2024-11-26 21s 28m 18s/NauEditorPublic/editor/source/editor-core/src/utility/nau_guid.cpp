// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_guid.hpp"

#include <random>

// ** NauObjectGUID

static std::random_device radomDevice;
static std::mt19937_64 RNG(radomDevice());
static std::uniform_int_distribution<std::mt19937_64::result_type> uniformDistribution;

NauObjectGUID::NauObjectGUID()
    : m_id(uniformDistribution(RNG))
{
}

NauObjectGUID::NauObjectGUID(uint64_t guid)
    : m_id(guid)
{
}
