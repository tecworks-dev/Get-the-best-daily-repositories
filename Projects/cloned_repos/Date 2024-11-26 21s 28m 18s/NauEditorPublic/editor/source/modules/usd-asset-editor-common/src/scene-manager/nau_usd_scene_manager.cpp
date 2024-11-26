// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/scene-manager/nau_usd_scene_manager.hpp"

#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "nau/utils/nau_usd_editor_utils.hpp"

#include <QFileInfo>


// ** NauUsdSceneManager

bool NauUsdSceneManager::createSceneFile(const std::string& path)
{
    if (QFileInfo(path.c_str()).exists()) {
        return false;
    }

    pxr::UsdStage::CreateNew(path.c_str());
    return true;
}

bool NauUsdSceneManager::loadScene(const std::string& path)
{
    m_stageWatcher.reset();

    if (QFileInfo(path.c_str()).exists()) {
        m_currentScene = pxr::UsdStage::Open(path.c_str());
    } else {
        m_currentScene = pxr::UsdStage::CreateNew(path.c_str());
    }

    //// We do not support USD transform operations
    // So when loading a scene we force clear opOrder and add one transform operation
    NauUsdSceneUtils::forceUpdatePrimsTransformOpOrder(m_currentScene);

    // Load references
    for (auto loadable : m_currentScene->FindLoadable()) {
        m_currentScene->Load(loadable);
    }

    currentSceneChanged.broadcast(m_currentScene);

    // TODO: Move to other system, use scene editor stage watcher
    m_stageWatcher = std::make_unique<UsdProxy::StageObjectChangedWatcher>(m_currentScene, [this](pxr::UsdNotice::ObjectsChanged const& notice) {
        m_isSceneSaved = false;
    });

    return true;
}

bool NauUsdSceneManager::unloadCurrentScene()
{
    m_stageWatcher.reset();
    m_currentScene.Reset();

    currentSceneChanged.broadcast(nullptr);

    return true;
}

bool NauUsdSceneManager::saveCurrentScene()
{
    m_currentScene->Save();
    m_isSceneSaved = true;

    return true;
}

bool NauUsdSceneManager::isCurrentSceneSaved()
{
    return m_isSceneSaved;
}

pxr::UsdStageRefPtr NauUsdSceneManager::currentScene()
{
    return m_currentScene;
}

std::string NauUsdSceneManager::currentScenePath() const
{
    if (!m_currentScene) {
        return "";
    }

    return m_currentScene->GetRootLayer()->GetRealPath();
}