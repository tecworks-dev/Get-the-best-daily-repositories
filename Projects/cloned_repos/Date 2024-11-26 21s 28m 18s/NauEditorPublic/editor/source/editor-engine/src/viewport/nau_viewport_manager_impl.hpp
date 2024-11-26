// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport instances manager implementation

#pragma once

#include "nau/viewport/nau_viewport_manager.hpp"
#include "nau/render/render_window.h"
#include "nau/app/platform_window.h"

#include <map>


// ** NauViewportInfo

struct NauViewportInfo
{
    NauViewportWidget* viewportWidget = nullptr;
    nau::Ptr<nau::IPlatformWindow> coreWindow = nullptr;
    nau::render::IRenderWindow::WeakRef renderWindow;
};


// ** NauViewportManager

class NauViewportManager : public NauViewportManagerInterface
{
public:
    NauViewportManager();
    ~NauViewportManager() = default;

    NauViewportWidget* mainViewport() const override;

    NauViewportWidget* createViewport(const std::string& name) override;
    void deleteViewport(const std::string& name) override;
    void setViewportRendererWorld(const std::string& name, nau::Uid worldUid) override;
    void resize(const std::string& name, int newWidth, int newHeight) override;

private:
    NauViewportInfo m_mainViewportInfo;
    std::map<std::string, NauViewportInfo> m_viewportInfos;
};