// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Abstract editor window

#pragma once

#include "nau/nau_editor_config.hpp"

#include "nau/inspector/nau_inspector.hpp"
#include "nau/outliner/nau_world_outline_panel.hpp"
#include "viewport/nau_viewport_container_widget.hpp"
#include "nau_dock_manager.hpp"
#include "nau_logger.hpp"
#include "nau_title_bar.hpp"
#include "nau_main_toolbar.hpp"
#include "browser/nau_project_browser.hpp"
#include "nau_shortcut_hub.hpp"

#include "project/nau_project.hpp"
#include "baseWidgets/nau_widget.hpp"


// ** NauEditorWindowAbstract

class NAU_EDITOR_API NauEditorWindowAbstract : public NauMainWindow
{
    Q_OBJECT
    friend class NauEditor;
    friend class NauUsdSceneEditor;

public:
    virtual ~NauEditorWindowAbstract() = default;
    
    virtual NauDockManager* dockManager() const noexcept = 0;
    virtual NauMainToolbar* toolbar() const noexcept = 0;
    virtual NauShortcutHub* shortcutHub() const noexcept = 0;

protected:
    virtual NauWorldOutlinerWidget* outliner() const noexcept = 0;
    virtual NauInspectorPage* inspector() const noexcept = 0;
    virtual NauViewportContainerWidget* viewport() const noexcept = 0;

    // Private widgets getters. Can be used only in NauEditor
    virtual NauTitleBar* titleBar() const noexcept = 0;

    virtual NauTabbedLoggerWidget* logger() const noexcept = 0;
    virtual NauProjectBrowser* projectBrowser() const noexcept = 0;

    // User settings
    virtual void applyUserSettings(const NauProject& project) = 0;
    virtual void saveUserSettings(NauProjectPtr project) = 0;

    virtual void closeWindow(QCloseEvent* event) = 0;

    virtual void setUpdateWindowTitleCallback(const std::function<void()>& callback) = 0;

protected:
    Q_SIGNAL void eventCloseWindow(QCloseEvent* event);
};