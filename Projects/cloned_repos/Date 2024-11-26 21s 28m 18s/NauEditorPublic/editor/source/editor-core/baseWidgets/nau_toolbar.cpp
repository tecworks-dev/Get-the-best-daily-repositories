// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_toolbar.hpp"
#include "nau_assert.hpp"
#include "nau_icon.hpp"
#include "nau_color.hpp"

#include <QPainter>


constexpr int Spacer           = 8;

constexpr int ToolbarHeight    = 48;
constexpr int ToolbarHMargin   = 16;

constexpr int ButtonSize       = 32;
constexpr int ButtonMargin     = 8;

constexpr int SeparatorWidth   = 2;
constexpr int SeparatorHeight  = ButtonSize;
constexpr int SeparatorOffset  = (ButtonSize - SeparatorWidth) * 0.5 - Spacer;

// ** NauToolbarSection

NauToolbarSection::NauToolbarSection(Alignment alignment, NauToolbarBase* parent)
    : NauWidget(parent)
    , m_layout(nullptr)
{
    auto topLayout = new NauLayoutHorizontal(this);
    auto widget = new NauWidget(this);
    m_layout = new NauLayoutHorizontal(widget);
    m_layout->setSpacing(Spacer);

    if (alignment == Left) {
        topLayout->addWidget(widget);
        topLayout->addStretch(1);
    } else if (alignment == Center) {
        topLayout->addStretch(1);
        topLayout->addWidget(widget);
        topLayout->addStretch(1);
    } else if (alignment == Right) {
        topLayout->addStretch(1);
        topLayout->addWidget(widget);
    } else {
        NED_ASSERT(false && "Invalid/unimplemented alignment");
    }
}

NauToolButton* NauToolbarSection::addButtonInternal(NauIcon icon, const QString& tooltip)
{
    auto button = new NauToolButton(this);
    button->setIcon(icon);
    button->setAutoRaise(true);
    button->setContentsMargins(ButtonMargin, ButtonMargin, ButtonMargin, ButtonMargin);
    button->setFixedSize(ButtonSize, ButtonSize);
    button->setStyleSheet("QToolButton { margin: 0px; border-radius: 2px; }"
        "QToolButton:hover { background-color: #1B1B1B; }"
        "QToolButton:checked { background-color: #3143E5; }");  // TODO: use our button to get rid of qss

    if (!tooltip.isEmpty()) {
        button->setToolTip(tooltip);
    }

    m_layout->addWidget(button);
    return button;
}

NauToolButton* NauToolbarSection::addButton(NauIcon icon, const QString& tooltip, std::function<void()> callback)
{
    auto button = addButtonInternal(icon, tooltip);
    connect(button, &NauToolButton::clicked, callback);
    return button;
}

void NauToolbarSection::addMenu()
{
    Q_UNIMPLEMENTED();
}

void NauToolbarSection::addSeparator()
{
    auto separator = new NauLineWidget(NauColor(52, 52, 52), 2, Qt::Vertical, this);
    separator->setFixedWidth(SeparatorWidth);
    separator->setFixedHeight(SeparatorHeight);
    m_layout->addSpacing(SeparatorOffset);
    m_layout->addWidget(separator);
    m_layout->addSpacing(SeparatorOffset);
}

void NauToolbarSection::addExternalWidget(QWidget* widget)
{
    m_layout->addWidget(widget);
}


// ** NauToolbarBase

NauToolbarBase::NauToolbarBase(QWidget* parent)
    : NauWidget(parent)
{
    auto layout = new NauLayoutHorizontal(this);
    layout->setContentsMargins(ToolbarHMargin, 0, ToolbarHMargin, 0);
    setFixedHeight(ToolbarHeight);
}

NauToolbarSection* NauToolbarBase::addSection(NauToolbarSection::Alignment alignment)
{
    auto section = new NauToolbarSection(alignment, this);
    layout()->addWidget(section);
    return section;
}

void NauToolbarBase::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.fillRect(rect(), NauColor(0x282828));
}
