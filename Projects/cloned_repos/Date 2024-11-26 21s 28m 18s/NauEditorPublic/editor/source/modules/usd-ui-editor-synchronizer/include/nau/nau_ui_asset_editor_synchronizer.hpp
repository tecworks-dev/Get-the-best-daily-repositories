// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Provides synchronization between UsdStage and ui scene

#pragma once

#include "nau/nau_ui_asset_editor_synchronizer_config.hpp"
#include "nau/undo-redo/nau_usd_scene_undo_redo.hpp"
#include "nau/selection/nau_usd_selection_container.hpp"
#include "usd_translator/usd_translator.h"
#include "nau/nau_delegate.hpp"

#include "nau/input.h"

#include "nau/ui.h"
#include "nau/ui/elements/canvas.h"

#include "nau/scene/world.h"

#include "memory"


// TODO: Implement NauUISceneObjectsMap


// ** NauUISceneSynchronizer
// TODO: Make an interface for the scene synchronizer

class NAU_USD_UI_EDITOR_SYNCHRONIZER_API NauUISceneSynchronizer
{
    friend class NauUsdUIAssetEditorSynchronizer;

public:
    NauUISceneSynchronizer();
    ~NauUISceneSynchronizer();

    void onEditorSceneChanged(pxr::UsdStageRefPtr editorScene, nau::scene::IWorld::WeakRef uiWorld);

private:
    void loadUIScene(pxr::UsdStageRefPtr editorScene, nau::scene::IWorld::WeakRef uiWorld);
    void unloadUIScene(nau::scene::IWorld::WeakRef uiWorld);

private:
    pxr::UsdStageRefPtr m_editorCurrentScene;
    nau::ui::Canvas* m_uiCurrentScene = nullptr;
    nau::scene::IScene::WeakRef m_engineCurrentScene;
    std::unique_ptr<UsdTranslator::UITranslator> m_sceneTranslator;
};


// ** NauUsdUIAssetEditorSynchronizer

class NAU_USD_UI_EDITOR_SYNCHRONIZER_API NauUsdUIAssetEditorSynchronizer
{
public:
    NauUsdUIAssetEditorSynchronizer(const NauUsdSceneUndoRedoSystemPtr& undoRedoSystem, const NauUsdSelectionContainerPtr& selectionContainer);
    ~NauUsdUIAssetEditorSynchronizer() = default;

    void onEditorSceneChanged(pxr::UsdStageRefPtr editorScene, nau::scene::IWorld::WeakRef uiWorld);

    void focusOnObject(const pxr::SdfPath& primPath);

private:
    void startSelectionSync();
    void startTransformToolsSync();

private:
    NauUsdSceneUndoRedoSystemPtr m_sceneUndoRedoSystem;
    NauUsdSelectionContainerPtr m_selectionContainer;
    std::unique_ptr<NauUISceneSynchronizer> m_sceneSynchronizer;
};
