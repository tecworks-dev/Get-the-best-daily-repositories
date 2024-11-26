// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A widget to choose a type of sorting in the content view of the project browser.

#pragma once

#include "baseWidgets/nau_label.hpp"
#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_buttons.hpp"
#include "nau_sort_type.hpp"
#include "nau_sort_order.hpp"

#include <map>


// ** NauSortTypeWidget

class NAU_EDITOR_API NauSortTypeWidget : public NauWidget
{
    Q_OBJECT
public:

    explicit NauSortTypeWidget(NauWidget* parent = nullptr);

    // Update the current state to specified type and order.
    // 
    // UB if type or order were not defined.
    // Emits eventSortTypeOrderChanged(...) if new type and order were applied.
    void setSortTypeAndOrder(NauSortType type, NauSortOrder order);

    // Handles sort type and order changed from external.
    // No signals are emitted.
    void handleSortTypeAndOrder(NauSortType type, NauSortOrder order);
signals:
    void eventSortTypeOrderChanged(NauSortType type, NauSortOrder order);

private:
    void handleSortTypeChangeRequested(bool checked);
    void updateUI();
    void emitCurrentSortTypeAndOrder();

    NauSortType currentSortType() const;
    NauSortOrder currentSortOrder() const;

    static QString typeToTranslatedString(NauSortType type);

private:
    NauLabel* m_label = nullptr;
    NauAbstractButton* m_orderButton = nullptr;
    NauPushButton* m_typeButton = nullptr;
    NauMenu* m_sortingMenu = nullptr;
    bool m_signalsBlocked = false;

    std::map<NauSortType, QAction*> m_actionsByTypes;
};