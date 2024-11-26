// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Main menu of the project.

#pragma once

#include "nau_shortcut_hub.hpp"
#include "baseWidgets/nau_menu.hpp"
#include "project/nau_project.hpp"

#include <unordered_map>


// ** NauMainMenu

class NAU_EDITOR_API NauMainMenu : public NauMenuBar
{
    Q_OBJECT

    friend class NauTitleBarMenu;
    friend class NauEditor;

public:
    enum class MenuItem
    {
        Project,
        Scene,
        Edit,
        View,
        Help
    };

    NauMainMenu(NauShortcutHub* shortcutHub, NauWidget* parent);

    NauMenu* getMenu(MenuItem item) const;
    void setEditorLanguage(const QString& lang);

signals:
    void saveProjectRequested();
    void aboutDialogRequested();
    void feedbackDialogRequested();
    void eventReveal();
    void eventNewScene();
    void eventLoadScene();
    void eventSaveScene();
    void eventRecentScenes(QMenu* recent);  // TODO: NauMenu
    void eventUndo();
    void eventRedo();
    void eventSwitchLanguageRequested(const QString& lang);

private:
    void buildProjectSection();
    void buildSceneSection();
    void buildEditSection();
    void buildViewSection();
    void buildHelpSection();
    void buildLanguageSection();


    NauShortcutHub* m_shortcutHub;
    std::unordered_map<MenuItem, NauMenu*> m_menuByItem;
    std::unordered_map<QString, QAction*> m_langActionByLang;
};
