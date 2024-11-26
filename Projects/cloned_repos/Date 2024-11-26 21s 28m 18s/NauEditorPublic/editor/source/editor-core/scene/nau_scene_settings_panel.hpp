// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor scene settings widget.

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include <nau/scene/nau_scene_settings.hpp>

#include <QListWidget>


// ** NauSceneSettingsPanel
//
// Widget for displaying scene settings
// TODO: Change to create an interface through NauProperty (as in the inspector panel)

class NAU_EDITOR_API NauSceneSettingsPanel : public NauWidget
{
    Q_OBJECT

public:
    NauSceneSettingsPanel(NauWidget* parent);

    void loadSettings(NauSceneSettings* sceneSettings);

private:
    QString getRelativeFilePathFromFileExplorer();

private:
    QListWidget* m_initailScriptsProperty;
    NauLayoutVertical* m_layout;
};
