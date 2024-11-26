// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Animation timeline widgets

#pragma once

#include "baseWidgets/nau_buttons.hpp"
#include "baseWidgets/nau_static_text_label.hpp"
#include "baseWidgets/nau_widget.hpp"

#include "nau_assert.hpp"


// ** NauTimelineSuffixedWidget

template< class T >
class NauTimelineSuffixedWidget : public T
{
public:
    NauTimelineSuffixedWidget(const QString& suffix, NauWidget* parent)
        : T(parent)
        , m_suffixLabel(new NauStaticTextLabel(suffix, nullptr))
    {
        m_suffixLabel->setParent(this);
        m_suffixLabel->resize(22, 16);
        m_suffixLabel->setColor(NauColor{ 91, 91, 91 });
    }

    void setSuffix(const QString& suffix)
    {
        NED_ASSERT(m_suffixLabel);
        m_suffixLabel->setText(suffix);
    }

protected:
    void resizeEvent(QResizeEvent* event) override
    {
        T::resizeEvent(event);
        if (event->size().isValid()) {
            constexpr int TOP_MARGIN = 4;
            constexpr int RIGHT_MARGIN = 8;
            const int positionX = event->size().width() - RIGHT_MARGIN - m_suffixLabel->width();
            m_suffixLabel->move(positionX, TOP_MARGIN);
        }
    }

private:
    NauStaticTextLabel* m_suffixLabel;
};


// ** NauTimelineScrollBar

class NauTimelineScrollBar : public NauWidget
{
    Q_OBJECT

public:
    NauTimelineScrollBar(NauWidget* parent, Qt::Orientation orientation);

    [[nodiscard]]
    int length() const noexcept;
    [[nodiscard]]
    float position() const noexcept;
    [[nodiscard]]
    float viewLength() const noexcept;
    [[nodiscard]]
    bool isHorizontal() const noexcept;

    void setUseRange(bool flag) noexcept;
    void setLength(int length);
    void setPosition(float value) noexcept;
    void setViewLength(float length) noexcept;

    void shrinkScroller(float value) noexcept;

signals:
    void eventScrollerChanged();
    void eventScrollerChangeFinished();

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    enum BaseConstants
    {
        RANGE_LENGTH = 16,
        RANGE_LINE_WIDTH = 2,
        SCROLL_WIDTH = 4,
        SCROLL_MIN_LENGTH = (RANGE_LENGTH + RANGE_LINE_WIDTH) * 2,
    };

    enum class MoveEventType : char
    {
        None,
        LeftRange,
        RightRange,
        View
    };

    float m_position;
    float m_viewLength;
    float m_previouseMouseMovePosition;

    Qt::Orientation m_orientation;
    MoveEventType m_moveEvent;
    bool m_useRange;
    bool m_pressed;
};


// ** NauTimelineTreeWidget

class NauTimelineTreeWidget : public NauTreeWidget
{
    Q_OBJECT

public:
    NauTimelineTreeWidget(NauWidget* parent);

signals:
    void eventItemHover(const QModelIndex& index);

protected:
    bool viewportEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
};


// ** NauTimelineRecordButton

class NauTimelineButton : public NauMiscButton
{
    Q_OBJECT

public:
    NauTimelineButton(NauWidget* parent);

    void setBackgroundColor(const NauColor& color) noexcept;

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    NauColor m_backgroundColor;
};