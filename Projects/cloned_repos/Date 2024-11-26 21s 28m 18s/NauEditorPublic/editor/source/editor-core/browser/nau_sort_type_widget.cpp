// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_sort_type_widget.hpp"
#include "nau_assert.hpp"

#include <QActionGroup>


// ** NauNauSortTypeWidget

NauSortTypeWidget::NauSortTypeWidget(NauWidget* parent)
    : NauWidget(parent)
{
    auto layout = new NauLayoutHorizontal(this);
    layout->setSpacing(8);

    m_sortingMenu = new NauMenu(tr("Sorting menu"), this);
    auto group = new QActionGroup(this);
    group->setExclusionPolicy(QActionGroup::ExclusionPolicy::Exclusive);

    const auto registerAction = [this, group](const QString& title, NauSortType type, bool checked = false)
    {
        auto action = new NauAction(title);
        action->setCheckable(true);
        action->setChecked(checked);
        group->addAction(action);

        m_sortingMenu->addAction(action);
        m_actionsByTypes[type] = action;

        connect(action, &QAction::toggled, this, &NauSortTypeWidget::handleSortTypeChangeRequested);
    };

    registerAction(tr("Name"), NauSortType::Name, true);
    registerAction(tr("Type"), NauSortType::Type);
    registerAction(tr("Modifed"), NauSortType::ModifiedTime);
    registerAction(tr("Size"), NauSortType::Size);
    registerAction(tr("Path"), NauSortType::Path);

    m_orderButton = new NauMiscButton(this);
    m_orderButton->setCheckable(true);
    m_orderButton->setCursor(Qt::CursorShape::PointingHandCursor);
    m_orderButton->setFixedSize(16, 16);

    // To-Do: move this to the theme, after UI of this area is ready.
    QIcon icon;
    icon.addPixmap(QPixmap(":/UI/icons/sortDesc.png"), QIcon::Normal, QIcon::State::On);
    icon.addPixmap(QPixmap(":/UI/icons/sortAsc.png"), QIcon::Normal, QIcon::State::Off);
    m_orderButton->setIcon(icon);

    m_typeButton = new NauBorderlessButton(m_sortingMenu, this);
    m_typeButton->setObjectName("typeButton");
    m_typeButton->setToolTip(tr("Set sorting type"));
    m_typeButton->setFixedHeight(16);

    layout->addWidget(m_orderButton);
    layout->addWidget(m_typeButton);

    connect(m_orderButton, &NauPushButton::toggled, [this](bool checked) {
        updateUI();
        emitCurrentSortTypeAndOrder();
    });

    updateUI();
}

void NauSortTypeWidget::setSortTypeAndOrder(NauSortType type, NauSortOrder order)
{
    auto itAction = m_actionsByTypes.find(type);
    NED_ASSERT(itAction != m_actionsByTypes.end());

    m_orderButton->setChecked(order == NauSortOrder::Descending);
    itAction->second->setChecked(true);
}

void NauSortTypeWidget::handleSortTypeAndOrder(NauSortType type, NauSortOrder order)
{
    auto itAction = m_actionsByTypes.find(type);
    NED_ASSERT(itAction != m_actionsByTypes.end());

    const auto reverted = qScopeGuard([this]{
        m_signalsBlocked = false;
    });
    
    m_signalsBlocked = true;

    m_orderButton->setChecked(order == NauSortOrder::Descending);
    itAction->second->setChecked(true);
}

void NauSortTypeWidget::handleSortTypeChangeRequested(bool checked)
{
    if (!checked) return;

    updateUI();
    emitCurrentSortTypeAndOrder();
}

void NauSortTypeWidget::updateUI()
{
    m_orderButton->setToolTip(currentSortOrder() == NauSortOrder::Ascending 
        ? tr("Click to sort in descending order") 
        : tr("Click to sort in ascending order"));

    m_typeButton->setText(tr("Sort by %1").arg(typeToTranslatedString(currentSortType())));
}

void NauSortTypeWidget::emitCurrentSortTypeAndOrder()
{
    if (m_signalsBlocked) {
        return;
    }

    emit eventSortTypeOrderChanged(currentSortType(), currentSortOrder());
}

NauSortType NauSortTypeWidget::currentSortType() const
{
    for (const auto&[type, action] : m_actionsByTypes) {
        if (action->isChecked()) return type;
    }

    NED_ASSERT(!"Unchecked state");
    return NauSortType::Name;
}

NauSortOrder NauSortTypeWidget::currentSortOrder() const
{
    return m_orderButton->isChecked() ? NauSortOrder::Descending : NauSortOrder::Ascending;
}

QString NauSortTypeWidget::typeToTranslatedString(NauSortType type)
{
    switch (type)
    {
        case NauSortType::Name: return tr("name");
        case NauSortType::Type: return tr("type");
        case NauSortType::ModifiedTime: return tr("modification time");
        case NauSortType::Size: return tr("size");
        case NauSortType::Path: return tr("path");
    }

    NED_ASSERT(false && "Not implemented");
    return QString();
}
