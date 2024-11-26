// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor viewport wrapper for engine

#pragma once


#include "nau/platform/windows/app/windows_window.h"
#include "nau/rtti/rtti_impl.h"
#include "nau/app/window_manager.h"

#include "nau/viewport/nau_viewport_widget.hpp"


class NauEditorViewportWindow : public nau::IWindowsWindow
{
    NAU_CLASS_(NauEditorViewportWindow, nau::IWindowsWindow)

public:
    NauEditorViewportWindow(NauViewportWidget* viewportWindow);

    ~NauEditorViewportWindow();

    void setVisible(bool) override;

    bool isVisible() const override;

    eastl::pair<unsigned, unsigned> getSize() const override;
    void setSize(unsigned sizeX, unsigned sizeY) override;

    eastl::pair<unsigned, unsigned> getClientSize() const override;

    void setPosition(unsigned positionX, unsigned positionY) override;
    eastl::pair<unsigned, unsigned> getPosition() const override;


    void setName(const char* name) override;

    HWND getWindowHandle() const override;

private:
    NauViewportWidget* m_viewportWindow = nullptr;
};


class NauEditorWindowManager : public nau::IWindowManager
{
    NAU_CLASS_(NauEditorWindowManager, nau::IWindowManager)

public:
    NauEditorWindowManager(NauViewportWidget* viewportWindow);
    ~NauEditorWindowManager();

    nau::IPlatformWindow& getActiveWindow() override;
    nau::Ptr<nau::IPlatformWindow> createWindowFromWidget(NauViewportWidget* viewportWindow);

private:
    nau::Ptr<NauEditorViewportWindow> m_mainWindow;
};
