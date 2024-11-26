// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/undo-redo/nau_usd_scene_undo_redo.hpp"

#include "nau_log.hpp"
#include "nau_assert.hpp"


// ** NauUsdSceneUndoRedoSystem

NauUsdSceneUndoRedoSystem::NauUsdSceneUndoRedoSystem(NauUndoable* undoRedoSystem)
    : m_undoRedoSystem(undoRedoSystem)
{
}

void NauUsdSceneUndoRedoSystem::groupBegin(size_t commandsCount)
{
    if (commandsCount <= 1) {
        return;
    }

    if (m_undoRedoSystem->groupIsOpen()) {
        NED_ERROR("Undo/Redo: commands group already opened!");
        return;
    }

    m_undoRedoSystem->groupBegin(commandsCount);
}

void NauUsdSceneUndoRedoSystem::bindCurrentScene(pxr::UsdStageRefPtr currentScene)
{
    NED_ASSERT(currentScene);
    NED_ASSERT(!m_currentScene);
    m_currentScene = currentScene;
}

void NauUsdSceneUndoRedoSystem::unbindCurrentScene()
{
    m_currentScene.Reset();
}
