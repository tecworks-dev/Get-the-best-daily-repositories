// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport toolbar implementation

#pragma once

#include "nau_scene_camera_settings.hpp"
#include "nau_shortcut_hub.hpp"
#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_toolbar.hpp"
#include "nau/viewport/nau_viewport_widget.hpp"

#include <QTimer>


// ** NauSceneEditorViewportToolbar
//
// Viewport toolbar. Provides camera UI settings, gizmo selection

class NauSceneEditorViewportToolbar : public NauToolbarBase
{
    Q_OBJECT

public:
    explicit NauSceneEditorViewportToolbar(NauViewportWidget& viewport, NauShortcutHub* shortcutHub,
        NauWidget* parent = nullptr);

    NauSceneCameraSettingsWidget* cameraSettings() { return m_cameraSettingsView; }

signals:
    void eventCoordinateSpaceChanged(bool isLocal);

public slots:
    void handlePlaymodeOn();
    void handlePlaymodeOff();

private slots:
    void handleMenu();
    void handleCameraSettings();

private:
    void updateFromViewport();

private:
    NauViewportWidget& m_linkedViewport;
    NauSceneCameraSettingsWidget* m_cameraSettingsView;

    // Transform tool mode
    NauToolButton* m_selectActionButton = nullptr;
    NauToolButton* m_translateActionButton = nullptr;
    NauToolButton* m_rotateActionButton = nullptr;
    NauToolButton* m_scaleActionButton = nullptr;

    // Transform tool settings
    NauBorderlessButton* m_coordinateSpaceButton = nullptr;
    NauMenu* m_coordinateSpaceMenu = nullptr;

    // Camera
    NauToolButton* m_cameraButton = nullptr;
};