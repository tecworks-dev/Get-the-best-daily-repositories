// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Object inspector main widget

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_spoiler.hpp"
#include "nau_entity_creation_panel.hpp"


// ** NauComponentSpoiler
//
// A component spoiler serves as a visual encapsulation of properties

class NAU_EDITOR_API NauComponentSpoiler : public NauSpoiler
{
    Q_OBJECT

public:
    NauComponentSpoiler(const QString& title, const int animationDuration = 0, NauWidget* parent = 0);

signals:
    void eventComponentRemoved();

private:
    void createButtons();
    void updateMenus();

private:
    // Context menu of this area. Also attached to button with 3dots.
    std::unique_ptr<NauMenu> m_areaMenu;
};


// The inspector panel is responsible for displaying
// the properties of the components of the game entity selected by the user on the stage.
// Only this widget needs to know which entity is currently running, which file is selected, and so on.

#pragma region INSPECTOR PANEL LEVEL

// ** NauEntityInspectorPageHeader

class NauStaticTextLabel;

class NAU_EDITOR_API NauInspectorPageHeader : public NauWidget
{
    Q_OBJECT

public:
    NauInspectorPageHeader(const std::string& title, const std::string& subtitle);

    void changeIcon(const std::string& resource);
    void changeTitle(const std::string& title);
    void changeSubtitle(const std::string& subtitle);

    NauObjectCreationList* creationList() const;

private:
    inline static constexpr int Height = 80;
    inline static constexpr int OuterMargin = 16;
    inline static constexpr int HorizontalSpacer = 26;

    QLabel*             m_icon;
    NauStaticTextLabel* m_title;
    NauStaticTextLabel* m_subtitle;

    NauObjectCreationList* m_objectCreationList;
};


// ** NauInspectorPage

class NAU_EDITOR_API NauInspectorPage : public NauWidget
{
    Q_OBJECT

public:
    NauInspectorPage(NauWidget* parent);

    NauInspectorPageHeader* addHeader(const std::string& objectName, const std::string& objectKind);
    void addSpoiler(NauComponentSpoiler* spoiler);

    NauComponentSpoiler* addComponent(const std::string& componentName);
    void clear();

    NauLayoutVertical* mainLayout();

private:
    NauLayoutVertical* m_layout;
    std::vector<NauComponentSpoiler*> m_componentSpoilers;

    inline static constexpr int ErrorLabelOuterMargin = 16;
};