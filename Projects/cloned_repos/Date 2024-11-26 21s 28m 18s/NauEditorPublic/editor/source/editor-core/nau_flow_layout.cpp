// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_flow_layout.hpp"
#include "nau_assert.hpp"
#include "nau_log.hpp"


// ** NauFlowLayout

NauFlowLayout::NauFlowLayout(Qt::LayoutDirection direction, int margin,
    int hSpacing, int vSpacing, NauWidget* parent)
    : QLayout(parent)
    , m_layoutDirection(direction)
    , m_hSpace(hSpacing)
    , m_vSpace(vSpacing)
{
    NED_ASSERT(m_layoutDirection == Qt::RightToLeft || m_layoutDirection == Qt::LeftToRight);
    setContentsMargins(margin, margin, margin, margin);
}

NauFlowLayout::~NauFlowLayout()
{
    QLayoutItem* item = nullptr;
    while (count() > 0 && (item = takeAt(0))) {
        delete item;
    }
}

void NauFlowLayout::addItem(QLayoutItem* item)
{
    m_itemList.append(item);
}

void NauFlowLayout::addWidget(int atIndex, QWidget* widget)
{
    NED_ASSERT(widget && (atIndex >= 0) && (atIndex <= m_itemList.size()));

    QLayout::addWidget(widget);
    NED_ASSERT(!m_itemList.empty());

    auto item = takeAt(m_itemList.size() - 1);
    m_itemList.insert(atIndex, item);
}

int NauFlowLayout::horizontalSpacing() const
{
    return m_hSpace >= 0 ? m_hSpace : smartSpacing(QStyle::PM_LayoutHorizontalSpacing);
}

int NauFlowLayout::verticalSpacing() const
{
    return (m_vSpace >= 0) ? m_vSpace : smartSpacing(QStyle::PM_LayoutVerticalSpacing);
}

int NauFlowLayout::count() const
{
    return m_itemList.size();
}

QLayoutItem* NauFlowLayout::itemAt(int index) const
{
    return m_itemList.value(index);
}

QLayoutItem* NauFlowLayout::takeAt(int index)
{
    if (index >= 0 && index < m_itemList.size()) {
        return m_itemList.takeAt(index);
    }

    NED_WARNING("Failed to take at {} from the flow layout", index);
    return nullptr;
}

Qt::Orientations NauFlowLayout::expandingDirections() const
{
    return {};
}

bool NauFlowLayout::hasHeightForWidth() const
{
    return true;
}

int NauFlowLayout::heightForWidth(int width) const
{
    return doLayout(QRect(0, 0, width, 0), true);
}

void NauFlowLayout::setGeometry(const QRect &rect)
{
    QLayout::setGeometry(rect);
    doLayout(rect, false);
}

QSize NauFlowLayout::sizeHint() const
{
    return minimumSize();
}

QSize NauFlowLayout::minimumSize() const
{
    QSize size;
    for (const QLayoutItem* item : std::as_const(m_itemList)) {
        size = size.expandedTo(item->minimumSize());
    }

    const QMargins margins = contentsMargins();
    size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom());
    return size;
}

int NauFlowLayout::doLayout(const QRect& rect, bool testOnly) const
{
    int left, top, right, bottom;
    getContentsMargins(&left, &top, &right, &bottom);
    const QRect effectiveRect = rect.adjusted(+left, +top, -right, -bottom);

    return m_layoutDirection == Qt::LayoutDirection::LeftToRight
        ? doLayoutLeftToRight(rect, effectiveRect, testOnly)
        : doLayoutRightToLeft(rect, effectiveRect, testOnly);
}

QSize NauFlowLayout::calcSpaces(const QWidget* widget) const
{
    QSize result { horizontalSpacing(), verticalSpacing() };
    if (result.width() == -1) {
        result.setWidth(widget->style()->layoutSpacing(
            QSizePolicy::PushButton, QSizePolicy::PushButton, Qt::Horizontal));
    }
    if (result.height() == -1) {
        result.setHeight(widget->style()->layoutSpacing(
            QSizePolicy::PushButton, QSizePolicy::PushButton, Qt::Vertical));
    }

    return result;
}

int NauFlowLayout::doLayoutLeftToRight(const QRect& rect, const QRect& effectiveRect, bool testOnly) const
{
    int x = effectiveRect.x();
    int y = effectiveRect.y();
    int lineHeight = 0;

    for (QLayoutItem* item : std::as_const(m_itemList)) {
        const QSize spaces = calcSpaces(item->widget());

        int nextX = x + item->sizeHint().width() + spaces.width();
        if ((nextX - spaces.width()) > effectiveRect.right() && lineHeight > 0) {
            x = effectiveRect.x();
            y = y + lineHeight + spaces.height();
            nextX = x + item->sizeHint().width() + spaces.height();
            lineHeight = 0;
        }

        if (!testOnly) {
            item->setGeometry(QRect(QPoint(x, y), item->sizeHint()));
        }

        x = nextX;
        lineHeight = qMax(lineHeight, item->sizeHint().height());
    }

    return y + lineHeight - rect.y() + (rect.bottom() - effectiveRect.bottom());
}

int NauFlowLayout::doLayoutRightToLeft(const QRect& rect, const QRect& effectiveRect, bool testOnly) const
{
    int x = effectiveRect.right();
    int y = effectiveRect.y();
    int lineHeight = 0;

    for (auto it = m_itemList.rbegin(); it != m_itemList.rend(); ++it) {
        QLayoutItem* item = *it;
        const QSize wSize = item->sizeHint();
        const QSize spaces = calcSpaces(item->widget());

        x -= wSize.width();

        if (x < effectiveRect.left() && lineHeight > 0) {
             x = effectiveRect.right() - wSize.width();
             y = y + lineHeight + spaces.height();

            lineHeight = 0;
        }

        if (!testOnly) {
            item->setGeometry(QRect(QPoint(x, y), item->sizeHint()));
        }

        x -= spaces.width();
        lineHeight = qMax(lineHeight, item->sizeHint().height());
    }

    return y + lineHeight - rect.y() + (rect.bottom() - effectiveRect.bottom());
}

int NauFlowLayout::smartSpacing(QStyle::PixelMetric pm) const
{
    QObject* parent = this->parent();
    if (!parent) return -1;

    if (parent->isWidgetType()) {
        QWidget* pw = static_cast<QWidget*>(parent);
        return pw->style()->pixelMetric(pm, nullptr, pw);
    }

    return static_cast<QLayout*>(parent)->spacing();
}
