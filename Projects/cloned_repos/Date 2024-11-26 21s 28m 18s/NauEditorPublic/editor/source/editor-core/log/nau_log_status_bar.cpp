// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_log_status_bar.hpp"
#include "themes/nau_theme.hpp"


// ** NauLogStatusPanel

NauLogStatusPanel::NauLogStatusPanel(NauWidget* parent)
    : NauFrame(parent)
{
    auto layout = new NauLayoutHorizontal(this);
    m_label = new NauLabel(this);
    m_label->setProperty("logStatusPanelSelection", true);

    m_detailsToggle = new NauMiscButton(this);
    m_detailsToggle->setObjectName("logStatusPanelDetailToggle");
    m_detailsToggle->setToolTip(tr("Toggle details panel visibility"));
    m_detailsToggle->setFixedSize(QSize(16, 16));
    m_detailsToggle->setCheckable(true);
    m_detailsToggle->setIcon(Nau::Theme::current().iconLoggerDetailsToggle());
    connect(m_detailsToggle, &NauToolButton::clicked, this, &NauLogStatusPanel::eventToggleDetailVisibilityRequested);

    layout->addWidget(m_label);
    layout->addSpacerItem(new QSpacerItem(1024, 20, QSizePolicy::Expanding, QSizePolicy::Minimum));
    layout->addWidget(m_detailsToggle);
}

void NauLogStatusPanel::handleSelectionChanged(const QModelIndexList& selected)
{
    m_selected = selected;
    updateUi();
}

void NauLogStatusPanel::handleMessageCountChanged(std::size_t count)
{
    m_count = count;
    updateUi();
}

bool NauLogStatusPanel::detailsPanelVisible() const
{
    return m_detailsToggle->isChecked();
}

void NauLogStatusPanel::setDetailsPanelVisibilityAction(QAction* action)
{
    connect(m_detailsToggle, &NauMiscButton::toggled, action, &QAction::trigger);
    connect(action, &QAction::triggered, [this](bool on) {
        QSignalBlocker blocker{m_detailsToggle};
        m_detailsToggle->setChecked(on);
        emit eventToggleDetailVisibilityRequested(on);
    });
}

void NauLogStatusPanel::updateUi()
{
    QString text;
    QTextStream str(&text);

    str << tr("%n item(s)", nullptr, static_cast<int>(m_count));

    if (!m_selected.empty()) {
        str << tr(" (%n selected)", nullptr, static_cast<int>(m_selected.size()));
    }

    m_label->setText(text);
}
