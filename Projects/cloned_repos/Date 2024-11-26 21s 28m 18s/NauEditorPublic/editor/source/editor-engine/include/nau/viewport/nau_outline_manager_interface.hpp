// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport outline manager
#pragma once

#include "nau/rtti/rtti_impl.h"
#include "nau/rtti/rtti_object.h"

#include "nau/nau_editor_engine_api.hpp"


// ** NauOutlineManagerInterface

class NAU_EDITOR_ENGINE_API NauOutlineManagerInterface : virtual public nau::IRefCounted
{
    NAU_INTERFACE(NauOutlineManagerInterface, nau::IRefCounted)

public:
    NauOutlineManagerInterface() = default;
    virtual ~NauOutlineManagerInterface() noexcept = default;

    NauOutlineManagerInterface(const NauOutlineManagerInterface&) = delete;
    NauOutlineManagerInterface(NauOutlineManagerInterface&&) noexcept = delete;

    NauOutlineManagerInterface& operator = (const NauOutlineManagerInterface&) = delete;
    NauOutlineManagerInterface& operator = (NauOutlineManagerInterface&&) noexcept = delete;

    virtual void enableOutline(bool flag) = 0;
};