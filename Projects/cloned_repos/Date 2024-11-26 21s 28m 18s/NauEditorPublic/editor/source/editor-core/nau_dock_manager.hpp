// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Wrapper around CDockManager from Advanced Docking System(ads).

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "nau_dock_widget.hpp"
#include "DockManager.h"
#include "DockAreaTitleBar.h"
#include "DockAreaTabBar.h"
#include "DockAreaWidget.h"
#include "DockWidgetTab.h"
#include "DockComponentsFactory.h"

#include <QVariantAnimation>
#include <set>


// ** NauDockManager

class NAU_EDITOR_API NauDockManager : public ads::CDockManager
{
    Q_OBJECT
public:
    explicit NauDockManager(QWidget* parent);

    const std::set<ads::CDockWidget*>& dockWidgets() const;

    ads::CDockWidget* projectBrowser() const;
    void setProjectBrowser(ads::CDockWidget* projectBrowser);

    ads::CDockWidget* inspector() const;
    void setInspector(ads::CDockWidget* inspector);

private:
    // It's a duplicate of ads::CDockManager::dockWidgetsMap.
    // But at this moment we can't use it for we have various code generation mode for Qt, Editor, ADS.
    // ToDo: Remove this member after we have consistent building of editor-engine-qt-ads.
    std::set<ads::CDockWidget*> m_dockWidgets;

    ads::CDockWidget* m_projectBrowser = nullptr;
    ads::CDockWidget* m_inspector = nullptr;
};


// ** NauDockAreaTitleBar

class NAU_EDITOR_API NauDockAreaTitleBar : public ads::CDockAreaTitleBar
{
    Q_OBJECT
public:
    explicit NauDockAreaTitleBar(ads::CDockAreaWidget* parent);

protected:
    void contextMenuEvent(QContextMenuEvent* event) override;

private:
    void createButtons();
    void updateMenus();

private:
    // Context menu of this area. Also attached to button with 3dots.
    std::unique_ptr<NauMenu> m_areaMenu;

    // Sub-menu of m_areaMenu that lists tabs not belong to this area.
    std::unique_ptr<NauMenu> m_addTabMenu;
    NauDockManager* const m_dockManager;
    ads::CDockAreaWidget* const m_areaWidget;
};


// ** NauDockAreaTitleBar

class NAU_EDITOR_API NauDockAreaTabBar : public ads::CDockAreaTabBar
{
    Q_OBJECT

public:
    explicit NauDockAreaTabBar(ads::CDockAreaWidget* parent);

};


// ** NauDockWidgetTab

class NAU_EDITOR_API NauDockWidgetTab : public ads::CDockWidgetTab
{
    Q_OBJECT
public:
    explicit NauDockWidgetTab(ads::CDockWidget* parent);

    void setFlashingEnabled(bool enabled);
    void setBadgeTextVisible(bool visible);

protected:
    void contextMenuEvent(QContextMenuEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    void updateUi();
    void resetFlashing();
    std::unique_ptr<QVariantAnimation> engageColorAnimator(const NauColor& from, const NauColor& to);

private:
    std::unique_ptr<QVariantAnimation> m_backgroundFlasher;
    std::unique_ptr<QVariantAnimation> m_foregroundFlasher;

    NauPalette m_palette;
    NauBrush m_backgroundBrush;
    QColor m_foregroundColor;

    int m_durationMsecs = 300;
    bool m_badgeVisible = false;
    QSize m_badgeSize{4, 4};

    static const int m_gap = 12;
};


// ** NauAdsComponentsFactory

class NAU_EDITOR_API NauAdsComponentsFactory : public ads::CDockComponentsFactory
{
public:
    ads::CDockAreaTitleBar* createDockAreaTitleBar(ads::CDockAreaWidget* DockArea) const override;

    ads::CDockAreaTabBar* createDockAreaTabBar(ads::CDockAreaWidget* DockArea) const override;

    ads::CDockWidgetTab* createDockWidgetTab(ads::CDockWidget* DockWidget) const override;
};


