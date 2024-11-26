// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Main window of the editor

#pragma once

#include "nau/app/nau_editor_window_interface.hpp"

#include "nau_dock_manager.hpp"
#include "nau_shortcut_hub.hpp"

#include "project/nau_project.hpp"
#
#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_widget_utility.hpp"
#include "inspector/nau_object_inspector.hpp"

#include "scene/nau_world.hpp"
#include "scene/nau_scene_settings_panel.hpp"
#include "nau/scene/nau_editor_scene_manager_interface.hpp"

#include "nau/viewport/nau_viewport_widget.hpp"

#include "commands/nau_commands.hpp"

#include "fileAccessor/nau_file_access.hpp"


// ** NauEditorWindow

class NAU_EDITOR_API NauEditorWindow : public NauEditorWindowAbstract
{
    Q_OBJECT

public:
    NauEditorWindow();
    ~NauEditorWindow();

    // Widgets getters
    NauDockManager* dockManager() const noexcept override;
    NauMainToolbar* toolbar() const noexcept override;

    NauShortcutHub* shortcutHub() const noexcept override;

protected:
    NauWorldOutlinerWidget* outliner() const noexcept override;
    NauInspectorPage* inspector() const noexcept override;
    NauViewportContainerWidget* viewport() const noexcept override;
    NauTitleBar* titleBar() const noexcept override;

    NauTabbedLoggerWidget* logger() const noexcept override;
    NauProjectBrowser* projectBrowser() const noexcept override;

    void applyUserSettings(const NauProject& project) override;
    void saveUserSettings(NauProjectPtr project) override;

    void closeWindow(QCloseEvent* event) override;

    void setUpdateWindowTitleCallback(const std::function<void()>& cb) override;

private:
    void closeEvent(QCloseEvent* event) override;
    void showEvent(QShowEvent* event) override;
    #ifdef Q_OS_WIN
    bool nativeEvent(const QByteArray& eventType, void* message, qintptr* result) override;
    #endif
    void mouseMoveEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

    // Loads default layout of docking widgets.
    // Used as alternative of user defined perspectives in case of there are no such one.
    void loadDefaultUiPerspective();

    // Tries to load main window appearance settings from specified project.
    void maybeLoadMainWindowAppearance();

private:
    NauShortcutHub* m_shortcutHub;
    NauDockManager* m_dockManager;
    
    // Main Widgets
    NauProjectBrowser* m_projectBrowser;
    NauEntityInspectorPage* m_entityInspector;
    NauSceneSettingsPanel* m_sceneSettingsPanel;
    NauTabbedLoggerWidget* m_logger;
    NauTitleBar* m_titleBar;
    NauMainToolbar* m_toolbar;
    NauViewportContainerWidget* m_viewport;
    NauWorldOutlinerWidget* m_worldOutline;
    NauInspectorPage* m_inspector;

    // User settings
    std::optional<NauProjectUserSettings::MainWindowAppearance> m_mainWindowAppearance;
};
