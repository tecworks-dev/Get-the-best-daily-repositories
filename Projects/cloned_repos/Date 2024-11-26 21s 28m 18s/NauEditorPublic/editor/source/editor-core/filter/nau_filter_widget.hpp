// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A widget to set filters.

#pragma once

#include "nau_flow_layout.hpp"
#include "nau_filter_checkbox.hpp"
#include "nau_filter_item_widget.hpp"
#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_buttons.hpp"

#include "nau_action_widget.hpp"

// TODO: Add support for the rest of the basic ActionWidget types

// ** NauFilterWidgetAction

class NauFilterWidgetAction : public NauWidgetAction
{
    Q_OBJECT
public:
    NauFilterWidgetAction(const NauIcon& icon, const QString& text, bool isChecked = false, NauWidget* parent = nullptr);

    NauFilterCheckBox& checkbox() const noexcept;
    QString text() const noexcept;

private:
    NauActionWidgetChecker* m_checkerActionWidget;
};

// ** NauFilterWidget

class NauFilterWidget : public NauWidget
{
    Q_OBJECT
public:
    explicit NauFilterWidget(NauWidget* parent);

    // filterItemOutput used as layout holder of filter items.
    explicit NauFilterWidget(NauFlowLayout* filterItemOutput, NauWidget* parent);

    // TODO: Currently, there is only one ActionWidget type in use.
    // But in the future, instead of this method,
    // there should be a factory that selects and connects the desired ActionWidget type to the main widget.
    void addFilterParam(NauFilterWidgetAction* action, bool isChecked = false);
    NauFilterWidgetAction* addFilterParam(const NauIcon& icon, const QString& name, bool isChecked = false);

signals:
    void eventChangeFilterRequested(const QList<NauFilterWidgetAction*>& newFilter);

private:
    void handleFilterStateChange(NauFilterWidgetAction* action, Qt::CheckState state);
    void emitChangeFilterSignal();

private:
    NauMenu* m_filterMenu = nullptr;

    NauAbstractButton* m_filterButton = nullptr;
    NauFlowLayout* m_layout = nullptr;

    std::unordered_map<NauFilterWidgetAction*, std::unique_ptr<NauFilterItemWidget>> m_filterByItems;
};
