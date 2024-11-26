// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// File that describes the basic buttons for the entire editor.

#pragma once

#include <QPushButton>
#include <QPainter>
#include <QPainterPath>
#include <QFocusEvent>
#include <QFlags>

#include "themes/nau_widget_style.hpp"


#pragma region ABSTRACT BUTTON LEVEL


// ** NauAbstractButton
//
// Abstract class for all buttons that will be used in the editor.

// TODO: Inherit NauAbstractButton from NauWidget.
// To make the styles available for other Nau widgets as well.
// But we will have to implement the click logic.
class NAU_EDITOR_API NauAbstractButton : public QPushButton
{
    Q_OBJECT

public:
    NauAbstractButton(QWidget* parent = nullptr);

    static int standardHeight();
    static int spacing();

    QSize sizeHint() const override;

    void setStateStyle(NauWidgetState state, NauWidgetStyle::NauStyle style);

protected:
    void paintEvent(QPaintEvent* event) override = 0;

    virtual void onPressed();
    virtual void onReleased();

    virtual void enterEvent(QEnterEvent* event) override;
    virtual void leaveEvent(QEvent* event) override;

    virtual void focusInEvent(QFocusEvent* event) override;
    virtual void focusOutEvent(QFocusEvent* event) override;

    virtual void changeEvent(QEvent* event) override;

    virtual void setState(NauWidgetState state);

protected:
    NauWidgetState m_currentState = NauWidgetState::Active;
    std::unordered_map<NauWidgetState, NauWidgetStyle::NauStyle> m_styleMap;
};
#pragma endregion

#pragma region BASE BUTTON LEVEL


// ** NauPrimaryButton
//
// Button to perform basic actions (presumably only in panels and tools for now).

class NAU_EDITOR_API NauPrimaryButton : public NauAbstractButton
{
public:
    NauPrimaryButton(QWidget* parent = nullptr);

public:
    void setRound(const QSize& size) noexcept;

protected:

private:
    QPainterPath getOutlinePath(NauWidgetState state);

protected:
    void paintEvent(QPaintEvent* event) override;

protected:
    QFlags<Qt::AlignmentFlag> textAlign = Qt::AlignLeft | Qt::AlignCenter;
    QFlags<Qt::AlignmentFlag> iconAlign = Qt::AlignCenter;

private:
    QSize m_round;
};


// ** NauSecondaryButton
//
// Button for performing secondary primary actions.

class NAU_EDITOR_API NauSecondaryButton : public NauPrimaryButton
{
public:
    NauSecondaryButton(QWidget* parent = nullptr);
};


// ** NauTertiaryButton
//
// Button for panel interfaces.

class NAU_EDITOR_API NauTertiaryButton : public NauSecondaryButton
{
public:
    NauTertiaryButton(QWidget* parent = nullptr);
};


// ** NauMiscButton
//
// Button for miscellaneous actions.

class NAU_EDITOR_API NauMiscButton : public NauPrimaryButton
{
public:
    NauMiscButton(QWidget* parent);

};
#pragma endregion
