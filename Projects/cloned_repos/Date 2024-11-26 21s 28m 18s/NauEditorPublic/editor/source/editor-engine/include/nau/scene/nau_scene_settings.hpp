// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor scene settings

#pragma once

#include "nau/nau_editor_engine_api.hpp"

#include <vector>
#include <string>


// ** NauSceneSettings
//
// Class with scene settings
// Temporary classes for scene settings storage and display
// TODO: Use NauDataModel for scene settings and maybe create a fake scene node and use settings as components

class NAU_EDITOR_ENGINE_API NauSceneSettings
{
public:
    NauSceneSettings() = default;
    ~NauSceneSettings() = default;

    void clear();

    const std::vector<std::string>& initialScripts() const { return m_initialScripts; }
    void addInitialScript(const std::string& script);
    void removeInitialScript(const std::string& script);

private:
    std::vector<std::string> m_initialScripts;
};
