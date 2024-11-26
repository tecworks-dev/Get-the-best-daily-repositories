// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_editor_window_manager.h"
#include "nau/runtime/internal/runtime_object_registry.h"

#include <QGuiApplication.h>


// ** NauEditorViewportWindow

NauEditorViewportWindow::NauEditorViewportWindow(NauViewportWidget* viewportWindow)
    : m_viewportWindow(viewportWindow)
{
    NAU_ASSERT(m_viewportWindow);
}

NauEditorViewportWindow::~NauEditorViewportWindow()
{
    // Destroy widget??
}

HWND NauEditorViewportWindow::getWindowHandle() const
{
    return reinterpret_cast<HWND>(m_viewportWindow->winId());
}

void NauEditorViewportWindow::setVisible(bool visible)
{
    m_viewportWindow->setVisible(visible);
}

bool NauEditorViewportWindow::isVisible() const
{
    return m_viewportWindow->isVisible();
}

eastl::pair<unsigned, unsigned> NauEditorViewportWindow::getSize() const
{
    NAU_ASSERT(m_viewportWindow);

    const unsigned xSize = m_viewportWindow->size().width() * m_viewportWindow->devicePixelRatioF();;
    const unsigned ySize = m_viewportWindow->size().height() * m_viewportWindow->devicePixelRatioF();;

    return {xSize, ySize};
}

void NauEditorViewportWindow::setSize(unsigned sizeX, unsigned sizeY)
{
    // TODO: Need implement?
    // We can set viewport size only from editor side
}

eastl::pair<unsigned, unsigned> NauEditorViewportWindow::getClientSize() const
{
    QRect rect = QGuiApplication::screens().at(0)->geometry();
    return {rect.width() * m_viewportWindow->devicePixelRatioF(), rect.height() * m_viewportWindow->devicePixelRatioF()};
}

void NauEditorViewportWindow::setPosition(unsigned positionX, unsigned positionY)
{
    // TODO: Need implement?
    // We can set viewport position only from editor side
}

eastl::pair<unsigned, unsigned> NauEditorViewportWindow::getPosition() const
{
    // TODO: Need implement?
    // We can set viewport position only from editor side
    return { 0,0 };
}

void NauEditorViewportWindow::setName(const char* name)
{
    m_viewportWindow->setObjectName(name);
}


// ** NauEditorWindowManager

NauEditorWindowManager::NauEditorWindowManager(NauViewportWidget* viewportWindow)
{
    nau::RuntimeObjectRegistration{nau::Ptr<>{this}}.setAutoRemove();

    // create window
    m_mainWindow = nau::rtti::createInstanceSingleton<NauEditorViewportWindow>(viewportWindow);
}

NauEditorWindowManager::~NauEditorWindowManager()
{
    m_mainWindow.reset();
}

nau::IPlatformWindow& NauEditorWindowManager::getActiveWindow()
{
    NAU_ASSERT(m_mainWindow);
    return *m_mainWindow;
}

nau::Ptr<nau::IPlatformWindow> NauEditorWindowManager::createWindowFromWidget(NauViewportWidget* viewportWindow)
{
    return nau::rtti::createInstance<NauEditorViewportWindow>(viewportWindow);
}