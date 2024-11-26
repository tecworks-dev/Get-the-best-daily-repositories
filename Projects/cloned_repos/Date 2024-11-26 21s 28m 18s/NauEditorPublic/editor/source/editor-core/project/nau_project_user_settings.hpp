// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// User-specific project settings stored in a .nauproject.user file

#pragma once

#include "nau_project_path.hpp"
#include "nau/scene/nau_scene_manager.hpp"

#include <optional>


// ** NauProjectUserSettings

class NAU_EDITOR_API NauProjectUserSettings : public QObject
{
    Q_OBJECT

public:
    // Main window appearance settings.
    struct MainWindowAppearance
    {
        bool isMaximized{};

        // Serialized value of geometry of main window.
        // See QWidget::saveGeometry.
        QByteArray geometryState;

        // Serialized state of tabs in docking system.
        // See ads::DockingManager::saveState.
        QByteArray dockingState;
    };


    NauProjectUserSettings(const NauProjectPath& path, NauSceneManager& manager);

    bool save();
    bool load();

    void addRecentScene(const NauSceneManager::SceneInfo& scene);
    void fillRecentScenesMenu(QMenu& menu);  // TODO: simply return a vector of scenes once we have a proper main menu

    void setWorldOutlineVisibleColumns(const std::vector<int>& visibleColumnsIds);
    std::vector<int> worldOutlineVisibleColumns() const;

    void setMainWindowAppearance(const MainWindowAppearance& appearance);
    const std::optional<MainWindowAppearance>& mainWindowAppearance() const;

    void setCameraSettings(const QJsonObject& data);
    const std::optional<QJsonObject> cameraSettings() const;

private:
    static std::optional<MainWindowAppearance> readMainWindowAppearance(const QJsonObject& root);
    static QJsonObject mainWindowAppearanceToJsonObject(const MainWindowAppearance& object);

    static const char* mainWindowSettingName();

private:
    const NauProjectPath  m_path;
    NauSceneManager&      m_sceneManager;

    // Settings
    std::vector<NauSceneManager::SceneInfo> m_recentScenes;
    std::vector<int> m_worldOutlineVisibleColumns;
    std::optional<MainWindowAppearance> m_mainWindowAppearance;
    std::optional<QJsonObject> m_cameraSettings;
};
