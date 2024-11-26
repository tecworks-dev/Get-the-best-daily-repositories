// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Main editor interface

#pragma once

#include "nau/nau_editor_config.hpp"
#include "nau/scene/nau_scene_manager.hpp"
#include "nau/app/nau_editor_window_interface.hpp"
#include "nau/assets/nau_asset_manager.hpp"
#include "nau/thumbnail-manager/nau_thumbnail_manger.hpp"
#include "commands/nau_commands.hpp"


// ** NauEditorInterface

class NAU_EDITOR_API NauEditorInterface
{
public:
    virtual ~NauEditorInterface() = default;

    static NauProjectPtr& currentProject();

    virtual const NauEditorWindowAbstract& mainWindow() const noexcept = 0;
    virtual NauAssetManagerInterfacePtr assetManager() const noexcept = 0;
    virtual NauThumbnailManagerInterfacePtr thumbnailManager() const noexcept = 0;
    virtual NauUndoable* undoRedoSystem() = 0;

    virtual void showMainWindow() = 0;
    virtual void switchScene(const NauSceneManager::SceneInfo& scene) = 0;
    virtual void postInit() = 0;

protected:

    // TODO: Move static project pointer from NauEditor
    inline static NauProjectPtr m_project = nullptr;
};