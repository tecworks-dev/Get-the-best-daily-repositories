// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A widget to display info about selected resources in the project browser.

#pragma once

#include "baseWidgets/nau_buttons.hpp"
#include "baseWidgets/nau_label.hpp"
#include "baseWidgets/nau_static_text_label.hpp"
#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_widget_utility.hpp"

#include <QItemSelection>


// ** NauProjectBrowserInfoWidget

class NAU_EDITOR_API NauProjectBrowserInfoWidget : public NauWidget
{
    Q_DECLARE_TR_FUNCTIONS(NauProjectBrowserInfoWidget)
public:
    explicit NauProjectBrowserInfoWidget(NauWidget* parent);

    void setRootIndex(const QModelIndex& rootIndex);
    void onSelectionChange(const QModelIndexList& selected);

private:
    void updateUI();
    void copySelectionToClipboard();

private:
    NauStaticTextLabel* m_label = nullptr;
    NauMiscButton* m_copyButton = nullptr;

    NauDir m_rootDirectory;
    QModelIndexList m_selection;
};