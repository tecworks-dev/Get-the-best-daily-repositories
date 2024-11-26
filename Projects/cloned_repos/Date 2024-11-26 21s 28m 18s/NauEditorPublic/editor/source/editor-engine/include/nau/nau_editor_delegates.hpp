// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Global editor events.

#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include "nau/nau_delegate.hpp"
#include "nau/nau_editor_modes.hpp"


// ** NauEditorDelegates
//
// Provides access to editor global events

class NAU_EDITOR_ENGINE_API NauEditorEngineDelegates
{
public:
    NauEditorEngineDelegates() = delete;

public:
    static NauDelegate<NauEditorMode> onEditorModeChanged;
    static NauDelegate<> onRenderDebug;
};