// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_assert.hpp"
#include "nau_log.hpp"
#include "nau_project_user_settings.hpp"
#include "nau_widget_utility.hpp"
#include "nau/app/nau_qt_app.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>
#include <QMessageBox>


// ** NauProjectUserSettings

NauProjectUserSettings::NauProjectUserSettings(const NauProjectPath& path, NauSceneManager& manager)
    : m_path(path)
    , m_sceneManager(manager)
{
    QObject::connect(&manager, &NauSceneManager::eventLoadScene, [this](const NauSceneManager::SceneInfo& scene) {
        this->addRecentScene(scene);
    });
}

bool NauProjectUserSettings::save()
{
    NED_ASSERT(m_path.isValid());
    const auto [success, path] = m_path.createUserSettingsFileIfNeeded();
    if (!success) {
        NED_ERROR("Failed to save user settings!");
        return false;
    }

    // Prepare data
    QJsonObject root;

    // Recent scenes
    QJsonArray scenes;
    for (const auto& scene : m_recentScenes) {
        scenes.append(NauSceneManager::serializeScene(scene));
    }
    root["recentScenes"] = scenes;

    // Outline columns
    QJsonArray columns;
    for (const int column : m_worldOutlineVisibleColumns) {
        columns.append(column);
    }
    root["outlineColumns"] = columns;

    if (m_mainWindowAppearance) {
        root[mainWindowSettingName()] = mainWindowAppearanceToJsonObject(*m_mainWindowAppearance);
    }

    if (m_cameraSettings) {
        root["cameraSettings"] = m_cameraSettings.value();
    }

    // Save
    NauFile result(path);
    if (!result.open(QIODevice::WriteOnly)) {
        NED_ERROR("Failed to open user settings file for saving at {}", path.toUtf8().constData());
        return false;
    }

    result.write(QJsonDocument(root).toJson());
    result.close();

    NED_DEBUG("Project user settings saved successfully");
    return true;
}

bool NauProjectUserSettings::load()
{
    const auto [success, path] = m_path.createUserSettingsFileIfNeeded();
    if (!success) {
        NED_ERROR("Failed to load user settings!");
        return false;
    }

    // Load project file
    NauFile projectFile(path);
    if (!projectFile.open(QIODevice::ReadOnly)) {
        NED_ERROR("Failed to load user settings at {}!", path.toUtf8().constData());
        return false;
    }

    // Load JSON
    QJsonObject data = QJsonDocument::fromJson(projectFile.readAll()).object();
    if (data.isEmpty()) {
        NED_WARNING("Empty user settings data, might be just empty or invalid JSON - {}!", path.toUtf8().constData());
        return false;
    }

    // Recent scenes
    m_recentScenes.clear();
    QJsonArray recentScenes = data["recentScenes"].toArray();
    for (const auto& scene : recentScenes) {
        m_recentScenes.push_back(NauSceneManager::deserializeScene(scene.toObject()));
    }

    // Outline columns
    m_worldOutlineVisibleColumns.clear();
    QJsonArray columns = data["outlineColumns"].toArray();
    for (const auto& column : columns) {
        m_worldOutlineVisibleColumns.push_back(column.toInt());
    }

#ifdef NDEBUG
    // Temporary workaround: do not read appearance settings in Debug.
    // We've got a warning while restoring Ads state: "QWindowsWindow::setGeometry: Unable to set geometry"
    // which leads to assertion fail.
    m_mainWindowAppearance = readMainWindowAppearance(data);
#endif

    if (data.contains("cameraSettings")) {
        m_cameraSettings = data["cameraSettings"].toObject();
    }

    NED_DEBUG("Project user settings loaded successfully");
    return true;
}

void NauProjectUserSettings::addRecentScene(const NauSceneManager::SceneInfo& scene)
{
    if (auto itScene = std::find(m_recentScenes.begin(), m_recentScenes.end(), scene); itScene != m_recentScenes.end()) {
        m_recentScenes.erase(itScene);
    }

    m_recentScenes.push_back(scene);

    // For now always save unconditionally
    save();
}

void NauProjectUserSettings::fillRecentScenesMenu(QMenu& menu)
{
    NED_DEBUG("Filling recent scenes menu");

    menu.clear();
    for (auto itScene = m_recentScenes.rbegin(); itScene != m_recentScenes.rend(); itScene++) {
        const auto& scene = *itScene;
        const auto relativePath = NauSceneManager::sceneToPath(scene);
        menu.addAction(NauSceneManager::sceneToPath(scene), [this, scene, relativePath] {
            if (m_sceneManager.sceneExists(scene)) {
                m_sceneManager.switchScene(scene);
            } else {  // Scene no longer exists, thats okay, just warn the user and update the set
                QMessageBox::warning(nullptr, NauApp::name(), QObject::tr("Scene %1 no longer exists!").arg(relativePath), QMessageBox::Ok);
                std::erase(m_recentScenes, scene);
            }
        });
    }
}

void NauProjectUserSettings::setWorldOutlineVisibleColumns(const std::vector<int>& visibleColumnsIds)
{
    m_worldOutlineVisibleColumns = visibleColumnsIds;
    save();
}

std::vector<int> NauProjectUserSettings::worldOutlineVisibleColumns() const
{
    return m_worldOutlineVisibleColumns;
}

void NauProjectUserSettings::setMainWindowAppearance(const MainWindowAppearance& appearance)
{
    m_mainWindowAppearance = appearance;
    save();
}

const std::optional<NauProjectUserSettings::MainWindowAppearance>& NauProjectUserSettings::mainWindowAppearance() const
{
    return m_mainWindowAppearance;
}

void NauProjectUserSettings::setCameraSettings(const QJsonObject& data)
{
    m_cameraSettings = data;
    save();
}

const std::optional<QJsonObject> NauProjectUserSettings::cameraSettings() const
{
    return m_cameraSettings;
}

std::optional<NauProjectUserSettings::MainWindowAppearance>
    NauProjectUserSettings::readMainWindowAppearance(const QJsonObject& root)
{
    if (!root.contains(mainWindowSettingName())) {
        return std::nullopt;
    }

    auto const& object = root[mainWindowSettingName()];

    return MainWindowAppearance{
        object["maximized"].toBool(),
        QByteArray::fromHex(object["geometry"].toVariant().toByteArray()),
        QByteArray::fromHex(object["docking"].toVariant().toByteArray()),
    };
}

QJsonObject NauProjectUserSettings::mainWindowAppearanceToJsonObject(const MainWindowAppearance& appearance)
{
    return QJsonObject {
        {"maximized", appearance.isMaximized },
        {"geometry", QJsonValue::fromVariant(appearance.geometryState.toHex()) },
        {"docking", QJsonValue::fromVariant(appearance.dockingState.toHex()) }
    };
}

const char* NauProjectUserSettings::mainWindowSettingName()
{
    return "mainWindowAppearance";
}
