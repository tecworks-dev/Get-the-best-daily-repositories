// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A wrapper over QTextEdit to handle application state changes based on user input

#pragma once

#include "nau_color.hpp"
#include "themes/nau_widget_style.hpp"
#include <QTextEdit>


// ** NauTextEdit

class NauTextEdit : public QTextEdit
{
    Q_OBJECT

public:
    NauTextEdit(QWidget* parent = nullptr);

public:
    void setState(NauWidgetState state);
    void setMaxLength(int maxLength);

signals:
    void eventStateChanged(NauWidgetState state);

protected:
    void paintEvent(QPaintEvent* event) override;

    void focusInEvent(QFocusEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;

    void keyPressEvent(QKeyEvent* event) override;

private:
    void handlePasteEvent();

private:
    NauWidgetState m_currentState;
    std::unordered_map<NauWidgetState, NauWidgetStyle::NauStyle> m_styleMap;

    int m_maxLength;
};
