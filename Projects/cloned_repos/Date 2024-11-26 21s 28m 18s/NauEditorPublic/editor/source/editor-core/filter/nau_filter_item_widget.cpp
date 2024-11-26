// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_filter_item_widget.hpp"
#include "nau_label.hpp"
#include "nau_palette.hpp"
#include "themes/nau_theme.hpp"

#include <QPainterPath>


// ** NauFilterItemWidget

NauFilterItemWidget::NauFilterItemWidget(const QString& title, QWidget* parent)
    : NauWidget(parent)
    , m_closeIcon(":/UI/icons/cross.svg")
    , m_closeButton(new NauMiscButton(this))
    , m_text(new NauStaticTextLabel(title, this))
    , m_hovered(false)
    , m_filterEnabled(true)
{
    setObjectName("filterItem");
    setFixedHeight(24);

    const auto& theme = Nau::Theme::current();
    m_palette = theme.paletteFilterItemWidget();
    m_text->setFont(theme.fontFilterItem());

    m_closeButton->setMaximumSize(m_closeIcon.size());
    connect(m_closeButton, &NauToolButton::clicked, this, &NauFilterItemWidget::eventDeleteRequested);

    auto layout = new NauLayoutHorizontal(this);
    layout->setContentsMargins(8, 4, 8, 4);
    layout->setSpacing(1);
    layout->addWidget(m_closeButton);
    layout->addWidget(m_text);

    updateUI();
}

void NauFilterItemWidget::setEnable(bool enable)
{
    if (m_filterEnabled != enable) {
        m_filterEnabled = enable;
        updateUI();
    }
}

bool NauFilterItemWidget::isFilterEnabled() const noexcept
{
    return m_filterEnabled;
}

void NauFilterItemWidget::mousePressEvent(QMouseEvent* event)
{
    NauWidget::mousePressEvent(event);
}

void NauFilterItemWidget::mouseReleaseEvent(QMouseEvent* event)
{
    NauWidget::mouseReleaseEvent(event);

    emit eventToggleActivityRequested(!m_filterEnabled);
}

void NauFilterItemWidget::updateUI()
{
    const NauPalette::State state = m_hovered ? NauPalette::Hovered : NauPalette::Normal;
    const NauPalette::Category category = m_filterEnabled ? NauPalette::Category::Active : NauPalette::Category::Inactive;
    const NauColor contentColor = m_palette.color(NauPalette::Role::Text, state, category);
    Nau::paintPixmap(m_closeIcon, contentColor);
    m_closeButton->setIcon(m_closeIcon);
    m_text->setColor(contentColor);

    setToolTip(m_filterEnabled ? tr("Click to disable this filter") : tr("Click to enable this filter"));
    update();
}

void NauFilterItemWidget::enterEvent(QEnterEvent* event)
{
    m_hovered = true;
    updateUI();
    NauWidget::enterEvent(event);
}

void NauFilterItemWidget::leaveEvent(QEvent* event)
{
    m_hovered = false;
    updateUI();
    NauWidget::leaveEvent(event);
}

void NauFilterItemWidget::paintEvent(QPaintEvent* event)
{
    constexpr double PEN_WIDTH = 1.0;
    constexpr double RECT_OFFSET = PEN_WIDTH * 0.5;
    constexpr double RECT_RADIUS = 1.5;
    const NauPalette::State state = m_hovered ? NauPalette::Hovered : NauPalette::Normal;
    const NauPalette::Category category = m_filterEnabled ? NauPalette::Category::Active : NauPalette::Category::Inactive;
    const QRectF rect{ RECT_OFFSET, RECT_OFFSET, width() - PEN_WIDTH, height() - PEN_WIDTH };
    const QPen pen{ m_palette.color(NauPalette::Role::Border, state, category), PEN_WIDTH };

    QPainterPath path;
    path.addRoundedRect(rect, RECT_RADIUS, RECT_RADIUS);

    QPainter painter(this);
    painter.setPen(pen);
    painter.fillPath(path, m_palette.color(NauPalette::Role::Background, state, category));
    painter.drawPath(path);
}
