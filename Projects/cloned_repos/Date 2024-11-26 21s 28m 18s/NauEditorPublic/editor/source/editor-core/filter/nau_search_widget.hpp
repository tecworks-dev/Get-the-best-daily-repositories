// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Search widget

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "themes/nau_widget_style.hpp"


// ** NauSearchWidget

class NauSearchWidget : public NauLineEdit
{
public:
    explicit NauSearchWidget(QWidget* widget);

    [[nodiscard]] const QAction* searchAction() const noexcept;

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QAction* m_searchAction = nullptr;

    std::unordered_map<NauWidgetState, NauWidgetStyle::NauStyle> m_styleMap;
};
