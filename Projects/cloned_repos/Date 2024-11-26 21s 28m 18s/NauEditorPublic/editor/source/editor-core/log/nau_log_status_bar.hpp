// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Status bar in a log panel.

#pragma once

#include "baseWidgets/nau_buttons.hpp"
#include "baseWidgets/nau_label.hpp"
#include "baseWidgets/nau_widget.hpp"


// ** NauLogStatusPanel

class NauLogStatusPanel : public NauFrame
{
    Q_OBJECT
public:
    explicit NauLogStatusPanel(NauWidget* parent);

    void handleSelectionChanged(const QModelIndexList& selected);
    void handleMessageCountChanged(std::size_t count);
    bool detailsPanelVisible() const;
    
    void setDetailsPanelVisibilityAction(QAction* action);

signals:
    void eventToggleDetailVisibilityRequested(bool visible);

private:
    void updateUi();

private:
    NauLabel* m_label = nullptr;
    NauMiscButton* m_detailsToggle = nullptr;

    std::size_t m_count = {};
    QModelIndexList m_selected;
};
