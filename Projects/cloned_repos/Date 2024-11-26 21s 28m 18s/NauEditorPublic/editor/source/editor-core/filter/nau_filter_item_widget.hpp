// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A widget represents a filter item.

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_label.hpp"
#include "nau_color.hpp"
#include "nau_palette.hpp"
#include "baseWidgets/nau_buttons.hpp"
#include "baseWidgets/nau_static_text_label.hpp"

#include <QMouseEvent>


// ** NauFilterItemWidget

class NauFilterItemWidget : public NauWidget
{
    Q_OBJECT
public:
    NauFilterItemWidget(const QString& title, QWidget* parent);

    void setEnable(bool enable);
    [[nodiscard]]
    bool isFilterEnabled() const noexcept;

signals:
    void eventToggleActivityRequested(bool on);
    void eventDeleteRequested();

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    void updateUI();

private:
    QPixmap m_closeIcon;
    NauMiscButton* m_closeButton = nullptr;
    NauStaticTextLabel* m_text = nullptr;
    NauPalette m_palette;
    bool m_hovered;
    bool m_filterEnabled;
};
