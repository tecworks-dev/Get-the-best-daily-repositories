// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport instances manager implementation

#include "nau_viewport_manager_impl.hpp"
#include "nau/service/service_provider.h"
#include "../engine/nau_editor_window_manager.h"
#include "nau/graphics/core_graphics.h"
#include "nau/editor-engine/nau_editor_engine_services.hpp"


// ** NauViewportManager

NauViewportManager::NauViewportManager()
{
    m_mainViewportInfo.viewportWidget = new NauViewportWidget(nullptr);
    m_mainViewportInfo.viewportWidget->setViewportName("Main");

    nau::Ptr<NauEditorWindowManager> windowManager = nau::rtti::createInstanceSingleton<NauEditorWindowManager>(m_mainViewportInfo.viewportWidget);
    nau::getServiceProvider().addService(windowManager);
}

NauViewportWidget* NauViewportManager::mainViewport() const
{
    return m_mainViewportInfo.viewportWidget;
}

NauViewportWidget* NauViewportManager::createViewport(const std::string& name)
{
    if (m_viewportInfos.contains(name)) {
        return m_viewportInfos.at(name).viewportWidget;
    }

    auto windowManager = nau::getServiceProvider().find<NauEditorWindowManager>();

    auto& viewportInfo = m_viewportInfos[name];
    viewportInfo.viewportWidget = new NauViewportWidget(nullptr);
    viewportInfo.viewportWidget->setViewportName(name);
    viewportInfo.coreWindow = windowManager->createWindowFromWidget(viewportInfo.viewportWidget);

    auto* const coreGraphics = nau::getServiceProvider().find<nau::ICoreGraphics>();
    auto windowHandle = viewportInfo.viewportWidget->windowHandle();
    viewportInfo.renderWindow = *Nau::EditorEngine().runTaskSync(coreGraphics->createRenderWindow(windowHandle).detach());
    Nau::EditorEngine().runTaskSync(viewportInfo.renderWindow.acquire()->enableRenderStages(nau::render::NauRenderStage::NauGUIStage).detach());

    return viewportInfo.viewportWidget;
}

void NauViewportManager::deleteViewport(const std::string& name)
{
    if (!m_viewportInfos.contains(name)) {
        return;
    }

    auto& viewportInfo = m_viewportInfos[name];

    // Close render window
    auto* const coreGraphics = nau::getServiceProvider().find<nau::ICoreGraphics>();
    auto windowHandle = viewportInfo.viewportWidget->windowHandle();
    Nau::EditorEngine().runTaskSync(coreGraphics->closeWindow(windowHandle));

    // Destroy core window instance
    viewportInfo.coreWindow.reset();

    // Destroy widget
    viewportInfo.viewportWidget->deleteLater();

    m_viewportInfos.erase(name);
}

void NauViewportManager::setViewportRendererWorld(const std::string& name, nau::Uid worldUid)
{
    if (!m_viewportInfos.contains(name)) {
        return;
    }

    auto& viewportInfo = m_viewportInfos[name];
    viewportInfo.renderWindow.acquire()->setWorld(worldUid);
}

void NauViewportManager::resize(const std::string& name, int newWidth, int newHeight)
{
    // TODO: Refactor viewport manager
    if (name == "Main") {
        auto* coreGraphics = nau::getServiceProvider().find<nau::ICoreGraphics>();
        coreGraphics->getDefaultRenderWindow().acquire()->requestViewportResize(newWidth, newHeight).detach();
        return;
    }

    if (!m_viewportInfos.contains(name)) {
        return;
    }

    auto& viewportInfo = m_viewportInfos[name];
    viewportInfo.renderWindow.acquire()->requestViewportResize(newWidth, newHeight).detach();
}