// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport content editor interfaces

#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include "nau/nau_editor_modes.hpp"

#include <QMouseEvent>


// ** NauViewportContentToolsInterface

class NAU_EDITOR_ENGINE_API NauViewportContentToolsInterface
{
public:
    virtual ~NauViewportContentToolsInterface() = default;

    virtual void handleMouseInput(QMouseEvent* mouseEvent, float dpi) = 0;
    virtual bool isUsing() const = 0;
    
    virtual void setEditMode(NauEditingMode mode) = 0;
    virtual NauEditingMode editingMode() const = 0;
};

using NauViewportContentToolsInterfacePtr = std::shared_ptr<NauViewportContentToolsInterface>;