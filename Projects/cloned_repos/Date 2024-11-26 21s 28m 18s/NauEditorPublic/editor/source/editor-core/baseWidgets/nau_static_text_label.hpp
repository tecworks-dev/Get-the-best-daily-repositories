// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A simple self-resizing widget drawing a single line text label

#pragma once

#include "nau_widget.hpp"
#include "nau_font.hpp"
#include "nau_color.hpp"
#include <QStaticText>


// ** NauStaticTextLabel

class NAU_EDITOR_API NauStaticTextLabel : public NauWidget
{
    Q_OBJECT

public:
    NauStaticTextLabel(const QString& text, QWidget* parent = nullptr);

    void setText(const QString& text);
    void setFont(const NauFont& font);
    void setColor(NauColor color);

    QString text() const;

    QSize sizeHint() const override;

signals:
    void clicked();
    void doubleClicked();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QStaticText m_text;
    NauColor    m_color;
};
