// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Scene editor viewport tools

#pragma once

#include "nau/viewport/nau_viewport_content_editor_tools.hpp"
#include "nau/selection/nau_usd_selection_container.hpp"
#include "nau_transform_tools.hpp"
#include "nau_object_tools.hpp"


// ** NauSceneEditorViewportTools

class NauSceneEditorViewportTools : public NauViewportContentToolsInterface
{
public:
    NauSceneEditorViewportTools(NauUsdSelectionContainerPtr container);
    ~NauSceneEditorViewportTools() = default;

    void handleMouseInput(QMouseEvent* mouseEvent, float dpi) override;
    bool isUsing() const override;

    NauEditingMode editingMode() const override;
    void setEditMode(NauEditingMode mode) override;

    GizmoCoordinateSpace sceneToolsCoordinateSpace() const;
    void setSceneToolsCoordinateSpace(GizmoCoordinateSpace space);

private:
    // Tools settings
    NauEditingMode m_editMode;
    GizmoCoordinateSpace m_coordinateSpace;

    NauUsdSelectionContainerPtr m_selectionContainer;
    std::unique_ptr<NauTransformTool> m_activeTool;
    std::unique_ptr<NauObjectTools> m_objectTools;
};