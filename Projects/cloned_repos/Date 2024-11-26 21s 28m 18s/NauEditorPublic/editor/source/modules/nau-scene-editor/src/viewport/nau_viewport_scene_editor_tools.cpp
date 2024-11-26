// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_viewport_scene_editor_tools.hpp"
#include "nau_log.hpp"


// ** NauSceneEditorViewportTools

NauSceneEditorViewportTools::NauSceneEditorViewportTools(NauUsdSelectionContainerPtr container)
    : m_selectionContainer(container)
    , m_coordinateSpace(GizmoCoordinateSpace::Local)
    , m_objectTools(std::make_unique<NauObjectTools>(m_selectionContainer))
{

}

void NauSceneEditorViewportTools::handleMouseInput(QMouseEvent* mouseEvent, float dpi)
{
    if (m_activeTool) {
        m_activeTool->handleMouseInput(mouseEvent, dpi);
    }
    if (m_objectTools) {
        m_objectTools->handleMouseInput(mouseEvent, dpi);
    }
}

bool NauSceneEditorViewportTools::isUsing() const
{
   return (m_activeTool && m_activeTool->isUsing()) || (m_objectTools && m_objectTools->isUsing());
}

NauEditingMode NauSceneEditorViewportTools::editingMode() const
{
    return m_editMode;
}

void NauSceneEditorViewportTools::setEditMode(NauEditingMode mode)
{
    if (m_editMode == mode) {
        return;
    }

    m_editMode = mode;

    std::string editModeName;

    // activate/deactivate gizmo
    // TODO: move edit mode name to transform tools
    switch (m_editMode) {
    case NauEditingMode::Select:
        m_activeTool.reset();
        editModeName = "Select";
        break;
    case NauEditingMode::Translate:
        m_activeTool = std::make_unique<NauTranslateTool>(m_selectionContainer);
        editModeName = "Transform";
        break;
    case NauEditingMode::Rotate:
        m_activeTool = std::make_unique<NauRotateTool>(m_selectionContainer);
        editModeName = "Rotate";
        break;
    case NauEditingMode::Scale:
        m_activeTool = std::make_unique<NauScaleTool>(m_selectionContainer);
        editModeName = "Scale";
        break;
    default:
        NAU_LOG("Unknown transform tool type!");
        return;
    }

    if (m_activeTool) {
        m_activeTool->setCoordinateSpace(m_coordinateSpace);
    }

    if (m_objectTools) {
        m_objectTools->updateBasis();
    }

    NAU_LOG_DEBUG("Editing mode switched to {}", editModeName.c_str());
}

GizmoCoordinateSpace NauSceneEditorViewportTools::sceneToolsCoordinateSpace() const
{
    return m_coordinateSpace;
}

void NauSceneEditorViewportTools::setSceneToolsCoordinateSpace(GizmoCoordinateSpace space)
{
    if (m_coordinateSpace == space) {
        return;
    }

    m_coordinateSpace = space;
    if (m_activeTool) {
        m_activeTool->setCoordinateSpace(m_coordinateSpace);
    }
    if (m_objectTools) {
        m_objectTools->updateBasis();
    }
}
