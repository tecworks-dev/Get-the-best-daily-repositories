// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "nau/nau_ui_asset_editor_synchronizer.hpp"

#include "nau/service/service_provider.h"
#include "nau/editor-engine/nau_editor_engine_services.hpp"

#include "nau/utils/nau_usd_editor_utils.hpp"

#include "nau_log.hpp"

#include "nau/app/window_manager.h"
#include "nau/app/platform_window.h"

#include "nau/scene/scene_manager.h"
#include "nau/scene/scene_factory.h"

// TODO: Use NauEditor wrappers from:
//#include "../src/engine/nau_editor_window_manager.h"


// ** NauUISceneSynchronizer

NauUISceneSynchronizer::NauUISceneSynchronizer()
{
}

NauUISceneSynchronizer::~NauUISceneSynchronizer()
{
}

void NauUISceneSynchronizer::onEditorSceneChanged(pxr::UsdStageRefPtr scene, nau::scene::IWorld::WeakRef uiWorld)
{
    if (!scene) {
        unloadUIScene(uiWorld);
        return;
    }

    m_editorCurrentScene = scene;
    loadUIScene(m_editorCurrentScene, uiWorld);
}

void NauUISceneSynchronizer::loadUIScene(pxr::UsdStageRefPtr editorScene, nau::scene::IWorld::WeakRef uiWorld)
{
    // Creating a canvas
    if(!m_uiCurrentScene) {
        m_uiCurrentScene = nau::ui::Canvas::create("canvas");
        m_uiCurrentScene->retain();

        nau::IPlatformWindow& window = nau::getServiceProvider().get<nau::IWindowManager>().getActiveWindow();
        window.setVisible(true);

        const auto [x, y] = window.getSize();

        nau::input::setScreenResolution(x, y);
        nau::getServiceProvider().get<nau::ui::UiManager>().setScreenSize(x, y);
        nau::getServiceProvider().get<nau::ui::UiManager>().configureResourcePath();

        m_uiCurrentScene->setReferenceSize({ float(1920), float(1080) });
        m_uiCurrentScene->setRescalePolicy(nau::ui::RescalePolicy::FitToSize);

        nau::getServiceProvider().get<nau::ui::UiManager>().addCanvas(m_uiCurrentScene);

        auto sceneCreateTask = [this, &uiWorld]() -> nau::async::Task<>
        {
            auto engineScene = nau::getServiceProvider().get<nau::scene::ISceneFactory>().createEmptyScene();
            engineScene->setName("UI service scene");
            nau::getServiceProvider().get<nau::ui::UiManager>().setEngineScene(engineScene.getRef());
            m_engineCurrentScene = co_await uiWorld->addScene(std::move(engineScene));
        };
        Nau::EditorEngine().runTaskSync(sceneCreateTask().detach());
    }

    m_sceneTranslator = std::make_unique<UsdTranslator::UITranslator>();
    m_sceneTranslator->setSource(m_editorCurrentScene);
    m_sceneTranslator->setTarget(m_uiCurrentScene);
    m_sceneTranslator->initScene();
    m_sceneTranslator->follow();
}

void NauUISceneSynchronizer::unloadUIScene(nau::scene::IWorld::WeakRef uiWorld)
{
    if (!m_uiCurrentScene) {
        return;
    }

    m_uiCurrentScene->removeAllChildren();
    nau::getServiceProvider().get<nau::ui::UiManager>().removeCanvas(m_uiCurrentScene->getCanvasName());
    m_uiCurrentScene->release();
    m_uiCurrentScene = nullptr;

    uiWorld->removeScene(m_engineCurrentScene);
}


// ** NauUsdUIAssetEditorSynchronizer

NauUsdUIAssetEditorSynchronizer::NauUsdUIAssetEditorSynchronizer(const NauUsdSceneUndoRedoSystemPtr& undoRedoSystem, const NauUsdSelectionContainerPtr& selectionContainer)
    : m_sceneUndoRedoSystem(undoRedoSystem)
    , m_selectionContainer(selectionContainer)
    , m_sceneSynchronizer(std::make_unique<NauUISceneSynchronizer>())
{
    // TODO: Implement
    //startSelectionSync();
    //startTransformToolsSync();
}

void NauUsdUIAssetEditorSynchronizer::onEditorSceneChanged(pxr::UsdStageRefPtr editorScene, nau::scene::IWorld::WeakRef uiWorld)
{
    m_sceneSynchronizer->onEditorSceneChanged(editorScene, uiWorld);
}

void NauUsdUIAssetEditorSynchronizer::focusOnObject(const pxr::SdfPath& primPath)
{
    // TODO: Implement
}

void NauUsdUIAssetEditorSynchronizer::startSelectionSync()
{
    // TODO: Implement
}

void NauUsdUIAssetEditorSynchronizer::startTransformToolsSync()
{
    // TODO: Implement
}
