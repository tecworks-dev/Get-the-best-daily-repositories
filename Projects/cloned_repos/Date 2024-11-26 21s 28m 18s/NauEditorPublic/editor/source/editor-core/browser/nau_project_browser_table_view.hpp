// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Enhances NauTreeView for the project browser.

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "nau_project_browser_file_operations_menu.hpp"

#include <QContextMenuEvent>
#include <QStaticText>


// ** NauProjectBrowserTableView

class NAU_EDITOR_API NauProjectBrowserTableView : public NauTreeView
{
    Q_OBJECT

public:
    NauProjectBrowserTableView(std::shared_ptr<NauProjectBrowserFileOperationsMenu> fileMenu, NauSplitter* parent);

    void setDataAndSelectionModel(QAbstractItemModel& model, QItemSelectionModel& selectionModel);

protected:
    virtual void contextMenuEvent(QContextMenuEvent* event) override;
    virtual void focusInEvent(QFocusEvent *event) override;
    virtual void mousePressEvent(QMouseEvent *event) override;
    virtual void startDrag(Qt::DropActions supportedActions) override;
    virtual void paintEvent(QPaintEvent* event) override;

private:
    std::shared_ptr<NauProjectBrowserFileOperationsMenu> m_fileOperationMenu;
    const QStaticText m_noFilterResultMessage;
};
