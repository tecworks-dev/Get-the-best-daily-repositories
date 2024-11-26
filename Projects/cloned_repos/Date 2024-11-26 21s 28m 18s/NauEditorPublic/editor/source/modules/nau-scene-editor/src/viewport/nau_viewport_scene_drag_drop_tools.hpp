// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Drag&drops tools for scene editor

#pragma once

#include "nau/viewport/nau_viewport_drag_drop_tools.hpp"


// ** NauSceneDragDropTools

class NauSceneDragDropTools : public NauViewportDragDropToolsInterface
{
public:
    void onDragEnterEvent(QDragEnterEvent* event) override {};
    void onDragLeaveEvent(QDragLeaveEvent* event) override {};
    void onDragMoveEvent(QDragMoveEvent* event) override {};
    void onDropEvent(QDropEvent* event) override {};
};