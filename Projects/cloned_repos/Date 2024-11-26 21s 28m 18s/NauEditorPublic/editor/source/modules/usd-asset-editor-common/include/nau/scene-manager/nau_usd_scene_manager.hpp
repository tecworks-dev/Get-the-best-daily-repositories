// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Scene editor class. Needed to edit the scene and synchronize it with engine scene.

#pragma once

#include "nau/nau_usd_asset_editor_common_config.hpp"

#include "pxr/usd/usd/stage.h"
#include "nau/scene/nau_editor_scene_manager_interface.hpp"
#include "nau/nau_delegate.hpp"

#include "usd_proxy/usd_stage_watcher.h"

#include <string>


// ** NauUsdSceneManager

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdSceneManager : public NauEditorSceneManagerInterface
{
public:
    pxr::UsdStageRefPtr currentScene();

    bool createSceneFile(const std::string& path) override;
    bool loadScene(const std::string& path) override;
    bool unloadCurrentScene() override;
    bool saveCurrentScene() override;
    bool isCurrentSceneSaved() override;
    std::string currentScenePath() const override;

public:
    NauDelegate<pxr::UsdStageRefPtr> currentSceneChanged;

private:
    pxr::UsdStageRefPtr m_currentScene;

    // TODO: Move to other system
    bool m_isSceneSaved = true;
    std::unique_ptr<UsdProxy::StageObjectChangedWatcher> m_stageWatcher;
};