// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Enhances QT's QTreeView for the project browser.

#pragma once

#include "nau_project_browser_file_operations_menu.hpp"
#include "baseWidgets/nau_widget.hpp"

#include <QContextMenuEvent>


// ** NauProjectBrowserTreeView

class NAU_EDITOR_API NauProjectBrowserTreeView : public NauTreeView
{
    Q_OBJECT
public:
    NauProjectBrowserTreeView(std::shared_ptr<NauProjectBrowserFileOperationsMenu> fileMenu, NauSplitter* parent);

    virtual void setModel(QAbstractItemModel* model) override;
    virtual void contextMenuEvent(QContextMenuEvent* event) override;
    virtual void paintEvent(QPaintEvent* event) override;

signals:
    void eventContextMenuRequestedOnEmptySpace();

private:
    std::shared_ptr<NauProjectBrowserFileOperationsMenu> m_fileOperationMenu;

};
