// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A widget to display info about selected resources in the project browser.

#pragma once

#include "baseWidgets/nau_label.hpp"

#include <QItemSelection>
#include <QModelIndexList>


// ** NauProjectBrowserSummaryWidget

class NAU_EDITOR_API NauProjectBrowserSummaryWidget : public NauLabel
{
    Q_DECLARE_TR_FUNCTIONS(NauProjectBrowserSummaryWidget)
public:
    explicit NauProjectBrowserSummaryWidget(NauWidget* parent);

    void setRootIndex(const QModelIndex& index);
    void onSelectionChange(const QModelIndexList& selected);

private:
    void updateUI();
    QString getFormattedSelectionSize() const;

private:
    int m_itemsCount{};
    QModelIndexList m_selection;
};