// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_buttons.hpp"
#include "themes/nau_theme.hpp"


#pragma region ABSTRACT BUTTON LEVEL


// ** NauAbstractButton

NauAbstractButton::NauAbstractButton(QWidget* parent)
    : QPushButton(parent)
{
    setFocusPolicy(Qt::StrongFocus);
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);

    connect(this, &QAbstractButton::pressed, this, &NauAbstractButton::onPressed);
    connect(this, &QAbstractButton::released, this, &NauAbstractButton::onReleased);
}

int NauAbstractButton::standardHeight()
{
    return 32;
}

int NauAbstractButton::spacing()
{ 
    return 8;
}

QSize NauAbstractButton::sizeHint() const
{
    QRect clientRect{ QPoint(0, 0), QSize(0, standardHeight()) };

    if (!icon().isNull()) {
        clientRect.setWidth(iconSize().width());
    }
    if (!text().isEmpty()) {
        QFontMetricsF metrics{font()};
        const int textWidth = static_cast<int>(std::ceil(metrics.horizontalAdvance(text())));

        clientRect.setWidth(clientRect.width() > 0
            ? clientRect.width() + spacing() + textWidth
            : textWidth);
    }

    return (clientRect + contentsMargins()).size();
}


void NauAbstractButton::setStateStyle(NauWidgetState state, NauWidgetStyle::NauStyle style)
{
    m_styleMap[state] = style;
}

void NauAbstractButton::onPressed()
{
    if (isEnabled()) {
        setState(NauWidgetState::Pressed);
    }
}

void NauAbstractButton::onReleased()
{
    if (isEnabled()) {
        setState(NauWidgetState::Hover);
    }
}

void NauAbstractButton::enterEvent([[maybe_unused]] QEnterEvent* event)
{
    if (isEnabled()) {
        setState(NauWidgetState::Hover);
    }
}

void NauAbstractButton::leaveEvent([[maybe_unused]] QEvent* event)
{
    if (isEnabled()) {
        setState(NauWidgetState::Active);
    }
}

void NauAbstractButton::focusInEvent(QFocusEvent* event)
{
    if (event->reason() == Qt::FocusReason::TabFocusReason) {
        setState(NauWidgetState::TabFocused);
    }

    QAbstractButton::focusInEvent(event);
}

void NauAbstractButton::focusOutEvent(QFocusEvent* event)
{
    setState(NauWidgetState::Active);

    QAbstractButton::focusOutEvent(event);
}

void NauAbstractButton::changeEvent(QEvent* event)
{
    if (event->type() == QEvent::EnabledChange) {
        setState(isEnabled() ? NauWidgetState::Active : NauWidgetState::Disabled);
    }

    QAbstractButton::changeEvent(event);
}

void NauAbstractButton::setState(NauWidgetState state)
{
    m_currentState = state;
    update();
}

#pragma endregion

#pragma region BASE BUTTON LEVEL


// ** NauPrimaryButton

NauPrimaryButton::NauPrimaryButton(QWidget* parent)
    : NauAbstractButton(parent)
    , m_round()
{
    const auto& theme = Nau::Theme::current();

    setFont(theme.fontPrimaryButton());
    m_styleMap = theme.stylePrimaryButton().styleByState;

    setContentsMargins(16, 8, 16, 8);
}

void NauPrimaryButton::setRound(const QSize& size) noexcept
{
    m_round = size;
}

QPainterPath NauPrimaryButton::getOutlinePath(NauWidgetState state)
{
    QPainterPath path;
    auto& style = m_styleMap[state];

    if (style.outlinePen.widthF() > 0.0) {
        const QRectF backgroundRect = rect() -  0.5 * style.outlinePen.widthF() * QMarginsF(1.0, 1.0, 1.0, 1.0);
        const auto& round = m_round.isValid() ? m_round : style.radiusSize;
        path.addRoundedRect(backgroundRect, round.width(), round.height());
    }

    return path;
}

void NauPrimaryButton::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.setRenderHint(QPainter::TextAntialiasing);

    painter.setPen(m_styleMap[m_currentState].outlinePen);

    painter.fillPath(getOutlinePath(m_currentState), m_styleMap[m_currentState].background);
    painter.drawPath(getOutlinePath(m_currentState));

    QRect drawRect = event->rect() - contentsMargins();
    QRect textRect = drawRect;
    QRect iconRect = drawRect;

    if (!text().isEmpty() && !icon().isNull()) {
        iconRect = QRect{drawRect.topLeft(), QSize(16, 16) };
        textRect.setLeft(iconRect.right() + spacing());
    }

    if (!icon().isNull()) {
        icon().paint(&painter, iconRect, iconAlign, m_styleMap[m_currentState].iconState, isChecked() ? NauIcon::On : NauIcon::Off);
    }

    painter.setPen(m_styleMap[m_currentState].textColor); // No border

    if (!text().isEmpty()) {
        // TODO: Add elided text
        // See documentation:
        // https://doc.qt.io/qt-5/qtwidgets-widgets-elidedlabel-example.html
        painter.drawText(textRect, textAlign, text());
    }
}

// ** NauSecondaryButton

NauSecondaryButton::NauSecondaryButton(QWidget* parent)
    : NauPrimaryButton(parent)
{
    const auto& theme = Nau::Theme::current();

    setFont(theme.fontPrimaryButton());
    m_styleMap = theme.styleSecondaryButton().styleByState;
}

// ** NauTertiaryButton

NauTertiaryButton::NauTertiaryButton(QWidget* parent)
    : NauSecondaryButton(parent)
{
    const auto& theme = Nau::Theme::current();

    setFont(theme.fontPrimaryButton());
    m_styleMap = theme.styleTertiaryButton().styleByState;
}

// ** NauMiscButton

NauMiscButton::NauMiscButton(QWidget* parent)
    : NauPrimaryButton(parent)
{
    setContentsMargins(0, 0, 0, 0);
    m_styleMap = Nau::Theme::current().styleMiscButton().styleByState;
}

#pragma endregion
