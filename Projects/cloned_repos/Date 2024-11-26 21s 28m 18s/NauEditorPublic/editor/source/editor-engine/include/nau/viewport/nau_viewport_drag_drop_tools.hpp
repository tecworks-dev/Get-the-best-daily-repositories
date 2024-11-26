// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Drag & drop tools for viewport interface

#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include "nau/nau_editor_modes.hpp"

#include <QMouseEvent>


// ** NauViewportDragDropToolsInterface

class NAU_EDITOR_ENGINE_API NauViewportDragDropToolsInterface
{
public:
    virtual ~NauViewportDragDropToolsInterface() = default;

    virtual void onDragEnterEvent(QDragEnterEvent* event) = 0;
    virtual void onDragLeaveEvent(QDragLeaveEvent* event) = 0;
    virtual void onDragMoveEvent(QDragMoveEvent* event) = 0;
    virtual void onDropEvent(QDropEvent* event) = 0;
};

using NauViewportDragDropToolsInterfacePtr = std::shared_ptr<NauViewportDragDropToolsInterface>;