// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_static_text_label.hpp"
#include "themes/nau_theme.hpp"

#include <QPainter>


// ** NauStaticTextLabel

NauStaticTextLabel::NauStaticTextLabel(const QString& text, QWidget* parent)
    : NauWidget(parent)
    , m_text(text)
    , m_color(Qt::white)
{
    auto textOption = m_text.textOption();
    textOption.setAlignment(Qt::AlignCenter);
    m_text.setTextOption(textOption);
    
    // TODO: default font
    setFont(Nau::Theme::current().fontStaticTextBase());
}

void NauStaticTextLabel::setText(const QString& text)
{
    m_text.setText(text);
    updateGeometry();
    update();
}

void NauStaticTextLabel::setFont(const NauFont& font)
{
    m_text.prepare({}, font);
    NauWidget::setFont(font);

    updateGeometry();
    update();
}

void NauStaticTextLabel::setColor(NauColor color)
{
    m_color = color;
    update();
}

QString NauStaticTextLabel::text() const
{
    return m_text.text();
}

QSize NauStaticTextLabel::sizeHint() const
{
    return m_text.size().toSize();
}

void NauStaticTextLabel::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::TextAntialiasing);
    painter.setPen(m_color);

    // TODO: Add elided text
    // See documentation:
    // https://doc.qt.io/qt-5/qtwidgets-widgets-elidedlabel-example.html
    painter.drawStaticText(rect().topLeft(), m_text);
}

void NauStaticTextLabel::mousePressEvent(QMouseEvent* event)
{
    NauWidget::mousePressEvent(event);
    event->accept();
}

void NauStaticTextLabel::mouseDoubleClickEvent(QMouseEvent* event)
{
    NauWidget::mouseDoubleClickEvent(event);
    event->accept();

    emit doubleClicked();
}

void NauStaticTextLabel::mouseReleaseEvent(QMouseEvent* event)
{
    NauWidget::mouseReleaseEvent(event);
    event->accept();

    emit clicked();
}
