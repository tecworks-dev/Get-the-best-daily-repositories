// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Custom view delegate for item view.

#pragma once

#include "nau_palette.hpp"
#include "baseWidgets/nau_widget.hpp"

#include <QStyledItemDelegate>
#include <QTreeView>
#include <set>


// ** NauTreeViewItemDelegate

class NAU_EDITOR_API NauTreeViewItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    explicit NauTreeViewItemDelegate(QObject* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;

    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const override;

    bool editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option, const QModelIndex& index) override;

    void setRowHeight(int rowHeight);

    // Gap between icon/text/sort indicator inside of a cell
    void setSpacing(int spacing);
    void setBorderThickness(float border);
    void setCellContentsMargins(int left, int top, int right, int bottom);
    void setRowContentsMargins(int left, int top, int right, int bottom);

    // Set palette for drawing.
    void setPalette(const NauPalette& palette);

    // Draw column with logicalIndex via foreground color specified in the palette(NauPalette::ForegroundBrightText)
    void setColumnHighlighted(int logicalIndex, bool highlighted = true);

    // Temporary hack. Need to display parent buttons correctly. 
    void setInteractiveColumnVisible(const QModelIndex& logicalIndex, bool highlighted = true);

    // Set indentation in pixels from parent component
    void setIndentation(int indentation);

    // Using this function, you can specify the columns on which the user overridden click event will be triggered
    void setInteractableColumn(int logicalIndex);

    // Set the column to be the parent column.
    // Controllers will be added to the cells of this column. 
    void setRootColumn(int logicalIndex);

    // Marks the column whose cells are editable
    void setEditableColumn(int logicalIndex);

    // Whether to include the parent in the calculation of the indentation
    void setRootAffect(bool affected);

signals:
    // Sent when clicked on an object in the interactive column
    void buttonEventPressed(const QModelIndex& index, int logicalIndex);

    // Sent when you click on an object whose data can be changed
    void buttonRenamePressed(const QModelIndex& index, int logicalIndex);

private:
    // Returns client area for drawing of cell, with taking into account cell and row content margins.
    // See setCellContentsMargins(), setRowContentsMargins()
    QRect calculateContentRect(const QStyleOptionViewItem& option) const;

    // Returns an indentation for specified index. Only first column can have an indentation.
    int calcIndentation(const QModelIndex& index) const;
    bool hasIndentation() const;

private:
    int m_rowHeight = 32;
    int m_spacing = 4;
    int m_topArrowPadding = 4;
    int m_indentation = 0;

    int m_rootColumn = 0;

    float m_penWidth = 1.0f;
    QMargins m_cellContentMargin = {4, 8, 4, 8};
    QMargins m_rowContentMargin = {12, 0, 12, 0};

    std::set<int> m_highlightedColumns;
    std::set<int> m_interactableColumns;
    std::set<QModelIndex> m_interactableColumnsVisible;
    std::set<int> m_editableColumns;


    NauPalette m_palette;
    NauIcon m_arrowRight;
    NauIcon m_arrowDown;

    bool m_rootAffected;
};
