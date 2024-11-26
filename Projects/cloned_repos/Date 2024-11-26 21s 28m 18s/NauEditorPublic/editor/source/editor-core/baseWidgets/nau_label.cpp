// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_label.hpp"

#include <QMouseEvent>


// ** NauLabel

NauLabel::NauLabel(const QString& text, QWidget* parent)
    : QLabel(text, parent)
{
}

NauLabel::NauLabel(const QPixmap& px, QWidget* parent)
    : QLabel({}, parent)
{
    setPixmap(px);
}

NauLabel::NauLabel(QWidget* parent)
    : QLabel(parent)
{
}

void NauLabel::mousePressEvent(QMouseEvent* event)
{
    QLabel::mousePressEvent(event);
}

void NauLabel::mouseReleaseEvent(QMouseEvent* event)
{
    QLabel::mouseReleaseEvent(event);

    emit clicked();
}
