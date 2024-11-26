// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// File that contains everything related to World Outline

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_header_view.hpp"
#include "baseWidgets/nau_label.hpp"
#include "baseWidgets/nau_icon.hpp"
#include "nau_color.hpp"
#include "nau_shortcut_hub.hpp"
#include "nau_plus_enum.hpp"
#include "scene/nau_world.hpp"
#include "filter/nau_search_widget.hpp"
#include "nau_entity_creation_panel.hpp"
#include "filter/nau_filter_widget.hpp"
#include "nau_tree_view_item_delegate.hpp"

#include <set>


// ** NauWorldOulineItemContextMenu
//
// Context menu, allows the user to manipulate objects of the scene hierarchy

class NAU_EDITOR_API NauWorldOulineItemContextMenu : public NauMenu
{
    Q_OBJECT

public:
    NauWorldOulineItemContextMenu(const QString& title, NauWidget* parent = nullptr);
};


// ** NauWorldOutlineHeaderView
//
// View-model for World Outline table headers.

class NAU_EDITOR_API NauWorldOutlineHeaderView : public NauHeaderView
{
public:
    NauWorldOutlineHeaderView(Qt::Orientation orientation, QWidget* parent = nullptr);

    QSize sortIndicatorSize() const override;
};


// ** NauWorldOutlineTableWidget
//
// Widget representing a table of game objects that are placed in the game world.

class NAU_EDITOR_API NauWorldOutlineTableWidget : public NauTreeWidget
{
    Q_OBJECT

public:
    enum class Columns
    {
        // TODO: put back once implemented:
        // Visibility = 0,
        Disabled   = 0,
        Name       = 1,
        Type       = 2,
        // Modified   = 2,
        // Tags       = 3,
        // Layer      = 4,
        Guid       = 3,
    };

    NauWorldOutlineTableWidget(NauShortcutHub* shortcutHub, NauWidget* parent = nullptr);

    void changeAllChildredCheckState(QTreeWidgetItem* item, int logicalIndex, Qt::CheckState state);

    void setColumnVisibility(std::vector<int> visibleColumns);

    void clearTable();

    const int getCurrentRowIndex() const;
    const QModelIndexList getSelectedRowsWithGuids() const;

    // Calling a custom context menu
    void customMenuRequested(QPoint pos);

    // Custom event handlers
    // TODO: Extend the functionality of these functions to work with multi-select objects
    bool cutItems();
    bool copyItems();
    bool pasteItems();
    bool duplicateItems();
    bool deleteItems();
    bool renameItems(); // TODO: Multi-renaming
    bool focusOnItem(); // TODO: Multi-selection

    // Called when a tree object is modified, such as a rename
    void onItemChanged(QTreeWidgetItem* item);
 
    NauIcon* availabilityIcon() { return &m_availabilityIcon; }
    NauIcon* visibilityIcon() { return &m_visibilityIcon; }

// Signals to which the NauEngine will subscribe
signals:
    
    void eventRename(QTreeWidgetItem* item, const QString& newName);
    void eventColumnsVisibilityChanged(const std::vector<int>& visibleColums);
    void eventFocus(QTreeWidgetItem* item);

    // Signals for client subscribing
    // These operations should be in the context menu and not use the outliner in any way
    // All operations use object selection first rather than line selection in the outliner
    void eventCut();
    void eventCopy();
    void eventPaste();
    void eventDuplicate();
    void eventDelete();
    void eventMove(const QModelIndex& destination, const std::vector<QString>& guids);

    void eventTabIsNowCurrent();

protected:
    // Mouse click handlers
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dragMoveEvent(QDragMoveEvent *event) override;
    void dropEvent(QDropEvent *event) override;
    QMimeData* mimeData(const QList<QTreeWidgetItem *> &items) const override;

private:
    // TODO: Use in future
    // It's not working at the moment
    std::string generateNumericalDisplayName(const std::string& entityName);

private:
    NauShortcutHub* m_shortcutHub;

    NauWorldOutlineHeaderView* m_header;

    QString m_nameBeforeRenaming;

    NauIcon m_availabilityIcon;
    NauIcon m_visibilityIcon;

    NauTreeViewItemDelegate* m_delegate;
};


// ** NauWorldOutlinerWidget
//
// Widget that allows you to view objects located in the game world.

class NAU_EDITOR_API NauWorldOutlinerWidgetHeader : public NauWidget
{
    Q_OBJECT

public:
    NauWorldOutlinerWidgetHeader(NauWidget* parent);

    const std::vector<NauWorldOutlineTableWidget::Columns>& currentFilters();
    NauObjectCreationList* creationList() const;
    void singleObjectCreation(const std::string& objectName);

signals:
    void eventSearchFilterChanged(const QString&);
    void eventCreateObject(const std::string& typeName);
    void eventChangeOutlinerTab(const std::string& modeName);

private:
    NauObjectCreationList* m_objectCreationList;
    NauPrimaryButton* m_addButton;

private:
    inline static constexpr int Height = 64;
    inline static constexpr int OuterMargin = 16;

    std::vector<NauWorldOutlineTableWidget::Columns> m_currentFilters;
};


// ** NauWorldOutlinerWidget
//
// Widget that allows you to view objects located in the game world.
// It consists of a search block with filtering and a table of game objects.

class NAU_EDITOR_API NauWorldOutlinerWidget : public NauWidget
{
    Q_OBJECT

public:
    NauWorldOutlinerWidget(NauShortcutHub* shortcutHub, QWidget* parent = nullptr);

    NauWorldOutlinerWidgetHeader& getHeaderWidget() const;

    NauWorldOutlineTableWidget& outlinerTab();

private:
    void updateFilterData(const QString& filter);
    void createOutlinerTab();

private:
    NauShortcutHub* m_shortcutHub;

    NauLayoutVertical* m_mainLayout;
    NauLayoutStacked* m_stackedLayout;

    NauWorldOutlinerWidgetHeader* m_header;
    NauWorldOutlineTableWidget* m_tableTab;
};
