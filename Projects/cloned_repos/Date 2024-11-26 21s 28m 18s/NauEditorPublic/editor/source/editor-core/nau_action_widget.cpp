// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_action_widget.hpp"
#include "themes/nau_theme.hpp"
#include "nau_assert.hpp"


// ** NauActionWidgetAbstract

NauActionWidgetAbstract::NauActionWidgetAbstract(NauWidget* parent)
    : NauWidget(parent)
    , m_layout(new NauLayoutGrid(this))
{
}


// ** NauActionWidgetBase

NauActionWidgetBase::NauActionWidgetBase(const NauIcon& icon, const QString& text, NauWidget* parent)
    : NauActionWidgetAbstract(parent)
    , m_iconLabel(new NauLabel())
    , m_label(new NauStaticTextLabel(text))
{
    // Text box only, takes up as much space as there is content in there
    const std::vector<int> stretch = { 0, 0, 0, 0, 1, 0, 0 };
    const std::vector<int> width = { checkBoxSize, FirstBackspacing, iconSize, SecondBackspacing, MinTextBlockSize, SecondBackspacing, ToolButtonSize };

    // Thus, we define a grid of four columns, with irregular tapers between columns:
    for (int i = 0; i < 7; ++i) {
        m_layout->setColumnStretch(i, stretch[i]);
        m_layout->setColumnMinimumWidth(i, width[i]);
    }

    m_iconLabel->setPixmap(icon.pixmap(iconSize, iconSize));

    m_layout->setContentsMargins(0, 0, 0, BottonMargin);

    // Add Icon
    m_layout->addWidget(m_iconLabel, 0, 2, 1, 1);

    // Add Text Block
    m_layout->addWidget(m_label, 0, 4, 1, 1);
}

NauStaticTextLabel& NauActionWidgetBase::label()
{
    NED_ASSERT(m_label);
    return *m_label;
}


// ** NauActionWidgetChecker

NauActionWidgetChecker::NauActionWidgetChecker(const NauIcon& icon, const QString& text, NauWidget* parent)
    : NauActionWidgetBase(icon, text, parent)
    , m_checkBox(new NauFilterCheckBox("", this))
{
    m_checkBox->setFixedSize(checkBoxSize, checkBoxSize);

    // Add check box
    m_layout->addWidget(m_checkBox, 0, 0, 1, 1);
}

NauFilterCheckBox& NauActionWidgetChecker::checkBox()
{
    NED_ASSERT(m_checkBox);
    return *m_checkBox;
}


// ** NauActionWidgetCatalog

NauActionWidgetCatalog::NauActionWidgetCatalog(const NauIcon& icon, const QString& text, NauWidget* parent)
    : NauActionWidgetBase(icon, text, parent)
    , m_toolButton(new NauToolButton(this))
{
    const auto& theme = Nau::Theme::current();
    m_toolButton->setIcon(theme.iconInspectorArrowRight());

    // Add additional menu call button
    m_layout->addWidget(m_toolButton, 0, 6, 1, 1);
}


// ** NauActionWidgetCheckerCatalog

NauActionWidgetCheckerCatalog::NauActionWidgetCheckerCatalog(const NauIcon& icon, const QString& text, NauWidget* parent)
    : NauActionWidgetBase(icon, text, parent)
    , NauActionWidgetChecker(icon, text, parent)
    , NauActionWidgetCatalog(icon, text, parent)
{
}
