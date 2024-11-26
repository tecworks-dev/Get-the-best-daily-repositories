// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/scene/nau_scene_settings.hpp"


// ** NauSceneSettings

void NauSceneSettings::clear()
{
    m_initialScripts.clear();
}

void NauSceneSettings::addInitialScript(const std::string& script)
{
    m_initialScripts.push_back(script);
}

void NauSceneSettings::removeInitialScript(const std::string& script)
{
    m_initialScripts.erase(std::remove(m_initialScripts.begin(), m_initialScripts.end(), script), m_initialScripts.end());
}