// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_search_widget.hpp"
#include "themes/nau_theme.hpp"

#include <QPainterPath>


// ** NauSearchWidget

NauSearchWidget::NauSearchWidget(QWidget* widget)
    : NauLineEdit(widget)
{
    const auto& theme = Nau::Theme::current();
    m_styleMap = theme.styleSearchWidget().styleByState;

    setObjectName("searchWidget");
    setMinimumHeight(32);
    setMaximumHeight(32);
    setClearButtonEnabled(true);
    setStyleSheet("* { background-color: rgba(0, 0, 0, 0); border: 0px; }");

    findChild<QToolButton*>()->setIcon(theme.iconSearchClear());

    m_searchAction = addAction(theme.iconSearch(), QLineEdit::LeadingPosition);
}

void NauSearchWidget::paintEvent(QPaintEvent* event)
{
    const QPen pen = m_styleMap[stateListener().state()].outlinePen;
    const double penWidth = pen.widthF();
    const double roundValue = (height() - penWidth) * 0.5;
    const double offset = 0.5 * penWidth;
    const QSizeF size{ width() - penWidth, height() - penWidth };

    QPainterPath path;
    path.addRoundedRect(offset, offset, size.width(), size.height(), roundValue, roundValue);

    QPainter searchPainter{ this };
    searchPainter.setRenderHint(QPainter::SmoothPixmapTransform);
    searchPainter.setRenderHint(QPainter::Antialiasing);
    searchPainter.setPen(pen);
    searchPainter.fillPath(path, m_styleMap[stateListener().state()].background);
    searchPainter.drawPath(path);

    QLineEdit::paintEvent(event);
}

const QAction* NauSearchWidget::searchAction() const noexcept
{
    return m_searchAction;
}
