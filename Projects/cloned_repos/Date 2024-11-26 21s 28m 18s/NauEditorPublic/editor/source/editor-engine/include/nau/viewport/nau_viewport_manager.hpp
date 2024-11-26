// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport instances manager interface

#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include "nau/viewport/nau_viewport_widget.hpp"


// ** NauViewportManagerInterface

class NAU_EDITOR_ENGINE_API NauViewportManagerInterface
{
public:
    virtual ~NauViewportManagerInterface() = default;
    virtual NauViewportWidget* mainViewport() const = 0;

    virtual NauViewportWidget* createViewport(const std::string& name) = 0;
    virtual void deleteViewport(const std::string& name) = 0;
    virtual void setViewportRendererWorld(const std::string& name, nau::Uid worldUid) = 0;

    virtual void resize(const std::string& name, int newWidth, int newHeight) = 0;
};