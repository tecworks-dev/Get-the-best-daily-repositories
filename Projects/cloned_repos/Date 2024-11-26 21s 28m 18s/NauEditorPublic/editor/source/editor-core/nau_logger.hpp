// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Logger window

#pragma once

#include "nau_dock_manager.hpp"
#include "nau_dock_widget.hpp"
#include "nau_log.hpp"
#include "nau_shortcut_hub.hpp"
#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_widget_utility.hpp"

#include "log/nau_log_status_bar.hpp"

#include "log/nau_log_source_model.hpp"
#include "log/nau_log_tree_view_item_delegate.hpp"
#include "log/nau_log_widget.hpp"
#include "log/nau_log_toolbar.hpp"
#include "log/nau_logger_details_panel.hpp"


// ** NauLoggerControlPanel
//
// Logger toolbar wrapper

class NAU_EDITOR_API NauLoggerControlPanel : public NauWidget
{
    Q_OBJECT
public:
    NauLoggerControlPanel(NauWidget* parent);

    void setAutoScroll(bool set = true);

    QAction* detailsPanelVisibilityAction() const;

signals:
    void eventClear();
    void eventFilterByLevelChangeRequested(const std::vector<NauLogLevel>& preferredLevels);
    void eventToggleDetailVisibilityRequested(bool visible);

private:
    NauLogToolBar* m_toolbar;
    NauLayoutHorizontal* m_layout;
};


class NAU_EDITOR_API NauLoggerSourceTreeContainer : public NauFrame
{
    Q_OBJECT
public:
    NauLoggerSourceTreeContainer(QWidget* parent = nullptr);

signals:
    void eventSourceToggleRequested(const QStringList& sources);

private:
    NauTreeView* m_sourceTree = nullptr;
    NauLogSourceModel* sourceModel = nullptr;
};


// ** NauLoggerOutputPanel
// 
// Logger panel wrapper

class NAU_EDITOR_API NauLoggerOutputPanel : public NauWidget
{
    Q_OBJECT
public:
    NauLoggerOutputPanel(NauShortcutHub* shortcutHub, NauWidget* parent);

    void registerControlPanel(NauLogToolBar* toolBar);

    void setLogSourceFilter(const QStringList& sourceName);
    void setLogLevelFilter(const std::vector<NauLogLevel>& preferredLevels);

    void clear();
    void copyMessagesToClipboard() const;
    void copySelectedMessageToClipboard() const;

    bool detailsPanelVisible() const;

    std::size_t selectedItemsCount() const;

signals:
    void eventCurrentMessageChanged(const QModelIndex& current);
    void eventMessageSelectionChanged(const QModelIndexList& selected);
    void eventToggleDetailVisibilityRequested(bool visible);
    void eventOutputCleared();

private slots:
    void showContextMenu(const QPoint& position);
    void handleSaveLogAsRequest();

private:
    NauLogWidget* m_logger;
    NauLogStatusPanel* m_statusBar;
};


// ** NauLoggerWidget
//
// Combines logger output panel with a toolbar

class NAU_EDITOR_API NauLoggerWidget : public NauWidget
{
    Q_OBJECT

public:
    NauLoggerWidget(NauShortcutHub* shortcutHub, NauWidget* parent = nullptr);

private:
    NauLoggerSourceTreeContainer* m_sourceTreeContainer;
    NauLogToolBar* m_controlPanel;
    NauLoggerOutputPanel* m_loggerPanel;
    NauLoggerDetailsPanel* m_detailsPanel;
};



// ** NauTabbedLoggerWidget
// Combines multiple logger widgets into tabs

class NAU_EDITOR_API NauTabbedLoggerWidget : public NauDockWidget
{
    Q_OBJECT

public:
    NauTabbedLoggerWidget(NauShortcutHub* shortcutHub, NauDockManager* manager);
    
    NauLoggerWidget* addTab(const QString& name);
    NauDockWidget* getTab(const QString& name);
    void switchTab(const QString& name);

private:
    NauShortcutHub* m_shortcutHub;
    NauDockManager* m_manager;
    std::map<QString, NauDockWidget*> m_tabs;
};
