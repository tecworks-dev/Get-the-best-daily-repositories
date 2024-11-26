// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/widgets/nau_common_timeline_widgets.hpp"

#include "baseWidgets/nau_buttons.hpp"
#include "themes/nau_theme.hpp"


// ** NauTimelineScrollBar

NauTimelineScrollBar::NauTimelineScrollBar(NauWidget* parent, Qt::Orientation orientation)
    : NauWidget(parent)
    , m_position(0.f)
    , m_viewLength(1.f)
    , m_previouseMouseMovePosition(-1.f)
    , m_orientation(orientation)
    , m_moveEvent(MoveEventType::None)
    , m_useRange(false)
    , m_pressed(false)
{
    if (m_orientation == Qt::Horizontal) {
        setMinimumWidth(SCROLL_MIN_LENGTH);
        setLength(height());
    }
    else {
        setMinimumHeight(SCROLL_MIN_LENGTH);
        setLength(width());
    }
}

int NauTimelineScrollBar::length() const noexcept
{
    return m_orientation == Qt::Horizontal ? width() : height();
}

float NauTimelineScrollBar::position() const noexcept
{
    return m_position;
}

float NauTimelineScrollBar::viewLength() const noexcept
{
    return m_viewLength;
}

bool NauTimelineScrollBar::isHorizontal() const noexcept
{
    return m_orientation == Qt::Horizontal;
}

void NauTimelineScrollBar::setUseRange(bool flag) noexcept
{
    m_useRange = flag;
}

void NauTimelineScrollBar::setLength(int length)
{
    length = std::max(0, length);

    if (isHorizontal()) {
        resize(length, SCROLL_WIDTH);
    }
    else {
        resize(SCROLL_WIDTH, length);
    }
}

void NauTimelineScrollBar::setPosition(float value) noexcept
{
    // TODO: fix scroller
    //m_position = std::clamp(value, 0.f, 1.f - m_viewLength);
    m_position = 0.f;
    update();
}

void NauTimelineScrollBar::setViewLength(float length) noexcept
{
    // TODO: fix scroller
    //m_viewLength = std::clamp(length, 0.f, 1.f);
    m_viewLength = 1.f;
    setPosition(m_position);
}

void NauTimelineScrollBar::shrinkScroller(float value) noexcept
{
    const float expectedLength = std::clamp(value, m_position + m_viewLength, 1.f);

    const float shrinkCoef = 1.f / expectedLength;
    setViewLength(m_viewLength * shrinkCoef);
    setPosition(m_position * shrinkCoef);
}

void NauTimelineScrollBar::mousePressEvent(QMouseEvent* event)
{
    if (!m_pressed) {
        const float positionValue = isHorizontal() ? event->position().x() : event->position().y();
        const float scrollerLength = static_cast<float>(length());
        const float position = positionValue / scrollerLength;
        if ((position < m_position) || (position > m_position + m_viewLength)) {
            setPosition(position - m_viewLength * 0.5f);
        }
        else if (m_useRange &&
            (positionValue >= m_position * scrollerLength) &&
            (positionValue <= m_position * scrollerLength + static_cast<float>(RANGE_LENGTH))) {
            m_moveEvent = MoveEventType::LeftRange;
        }
        else if (m_useRange &&
            (positionValue >= (m_position + m_viewLength) * scrollerLength - static_cast<float>(RANGE_LENGTH)) &&
            (positionValue <= (m_position + m_viewLength) * scrollerLength)) {
            m_moveEvent = MoveEventType::RightRange;
        }
        else {
            m_moveEvent = MoveEventType::View;
        }
        m_previouseMouseMovePosition = positionValue;
        m_pressed = true;
    }
    NauWidget::mousePressEvent(event);
}

void NauTimelineScrollBar::mouseReleaseEvent(QMouseEvent* event)
{
    if (m_pressed) {
        m_previouseMouseMovePosition = -1.f;
        m_moveEvent = MoveEventType::None;
        m_pressed = false;
        emit eventScrollerChangeFinished();
    }
    NauWidget::mouseReleaseEvent(event);
}

void NauTimelineScrollBar::mouseMoveEvent(QMouseEvent* event)
{
    const float positionValue = isHorizontal() ? event->position().x() : event->position().y();
    const float scrollerLength = static_cast<float>(length());
    const float viewPosition = m_position * scrollerLength;
    const float viewLength = m_viewLength * scrollerLength;
    if (m_moveEvent == MoveEventType::LeftRange) {
        const float rightRangePosition = std::round((m_position + m_viewLength) * scrollerLength - static_cast<float>(SCROLL_MIN_LENGTH));
        const float position = std::clamp(viewPosition + (positionValue - m_previouseMouseMovePosition), 0.f, rightRangePosition) / scrollerLength;
        const float diff = m_position - position;
        m_position = position;
        setViewLength(m_viewLength + diff);
        m_previouseMouseMovePosition = positionValue;
        emit eventScrollerChanged();
    }
    else if (m_moveEvent == MoveEventType::RightRange) {
        const float leftRangePosition = std::round(viewPosition + static_cast<float>(SCROLL_MIN_LENGTH));
        const float position = std::clamp(viewPosition + viewLength + (positionValue - m_previouseMouseMovePosition), leftRangePosition, scrollerLength) / scrollerLength;
        const float diff = position - m_position - m_viewLength;
        setViewLength(m_viewLength + diff);
        m_previouseMouseMovePosition = positionValue;
        update();
        emit eventScrollerChanged();
    }
    else if (m_moveEvent == MoveEventType::View) {
        setPosition(m_position + (positionValue - m_previouseMouseMovePosition) / scrollerLength);
        m_previouseMouseMovePosition = positionValue;
        emit eventScrollerChanged();
    }
    NauWidget::mouseMoveEvent(event);
}

void NauTimelineScrollBar::paintEvent(QPaintEvent* event)
{
    constexpr float RADIUS = static_cast<float>(SCROLL_WIDTH) * 0.5f;
    constexpr int LINE_RADIUS = RANGE_LINE_WIDTH / 2;
    constexpr int RANGE_OFFSET = RANGE_LENGTH + LINE_RADIUS;

    const NauPalette palette = Nau::Theme::current().paletteTimelineScrollBar();

    const auto [width, height] = size();
    const int widgetLength = length();
    const int viewPosition = static_cast<int>(std::round(m_position * widgetLength));
    const int viewLength = static_cast<int>(std::round(m_viewLength * widgetLength));

    QPainter painter{ this };
    QPainterPath path;
    std::array<QLine, 2> lines;

    if (m_orientation == Qt::Horizontal) {
        const QPoint offset{ 0, height - RANGE_LINE_WIDTH };
        QPoint start{ viewPosition + RANGE_OFFSET, LINE_RADIUS };
        lines[0] = { start, start + offset };
        start += { viewLength - RANGE_OFFSET * 2, 0 };
        lines[1] = { start, start + offset };
        path.addRoundedRect(viewPosition, 0, viewLength, height, RADIUS, RADIUS);
    } else {
        const QPoint offset{ width - RANGE_LINE_WIDTH, 0 };
        QPoint start{ LINE_RADIUS, viewPosition + RANGE_OFFSET };
        lines[0] = { start, start + offset };
        start += { 0, viewLength - RANGE_OFFSET * 2 };
        lines[1] = { start, start + offset };
        path.addRoundedRect(0, viewPosition, width, viewLength, RADIUS, RADIUS);
    }

    painter.fillPath(path, palette.color(NauPalette::Role::Background));
    if (m_useRange) {
        painter.setPen({ palette.color(NauPalette::Role::Foreground), RANGE_LINE_WIDTH });
        painter.drawLines(lines.data(), static_cast<int>(lines.size()));
    }
}


// ** NauTimelineTreeWidget

NauTimelineTreeWidget::NauTimelineTreeWidget(NauWidget* parent)
    : NauTreeWidget(parent)
{
}

bool NauTimelineTreeWidget::viewportEvent(QEvent* event)
{
    switch (event->type()) {
    case QEvent::HoverMove:
    case QEvent::HoverEnter:
    case QEvent::HoverLeave: {
        const QPoint position = static_cast<QHoverEvent*>(event)->position().toPoint();
        const QModelIndex index = indexAt(position);
        emit eventItemHover(index);
    }   break;
    }
    return NauTreeWidget::viewportEvent(event);
}

void NauTimelineTreeWidget::mousePressEvent(QMouseEvent* event)
{
    const QPoint position = event->position().toPoint();
    const QModelIndex item = indexAt(position);
    //const bool selected = selectionModel()->isSelected(indexAt(position));

    NauTreeWidget::mousePressEvent(event);

    if (!item.isValid()) {
        clearSelection();
        const QModelIndex index;
        selectionModel()->setCurrentIndex(index, QItemSelectionModel::Select);
        emit itemPressed(nullptr, -1);
    }
}


// ** NauTimelineButton

NauTimelineButton::NauTimelineButton(NauWidget* parent)
    : NauMiscButton(parent)
    , m_backgroundColor(0, 0, 0, 0)
{
    setCheckable(true);
    setChecked(false);
}

void NauTimelineButton::setBackgroundColor(const NauColor& color) noexcept
{
    m_backgroundColor = color;
}

void NauTimelineButton::paintEvent(QPaintEvent* event)
{
    constexpr int ROUND_SIZE = 2;
    constexpr int ELLIPSE_OUTER_SIZE = 2;
    constexpr int ELLIPSE_OUTER_SIZE_2 = 2 * ELLIPSE_OUTER_SIZE;
    constexpr int ELLIPSE_INNER_SIZE = 6;

    QPainter painter{ this };
    painter.setRenderHint(QPainter::Antialiasing);

    const auto [width, height] = size();

    QPainterPath ellipseOuter;
    ellipseOuter.addEllipse(width / 4, height / 4, width / 2, height / 2);
    ellipseOuter.addEllipse(width / 4 + ELLIPSE_OUTER_SIZE, height / 4 + ELLIPSE_OUTER_SIZE, width / 2 - ELLIPSE_OUTER_SIZE_2, height / 2 - ELLIPSE_OUTER_SIZE_2);

    QPainterPath ellipseInner;
    ellipseInner.addEllipse((width - ELLIPSE_INNER_SIZE) / 2, (height - ELLIPSE_INNER_SIZE) / 2, ELLIPSE_INNER_SIZE, ELLIPSE_INNER_SIZE);

    if (isChecked()) {
        QPainterPath path;
        path.addRoundedRect(0, 0, width, height, ROUND_SIZE, ROUND_SIZE);
        painter.fillPath(path, m_backgroundColor);
        painter.fillPath(ellipseInner, Qt::white);
        painter.fillPath(ellipseOuter, Qt::white);
    } else {
        painter.fillPath(ellipseInner, m_backgroundColor);
        painter.fillPath(ellipseOuter, m_backgroundColor);
    }
    NauMiscButton::paintEvent(event);
}