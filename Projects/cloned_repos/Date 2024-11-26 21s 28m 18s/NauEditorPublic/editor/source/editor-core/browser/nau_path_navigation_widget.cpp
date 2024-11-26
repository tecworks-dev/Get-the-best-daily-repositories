// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_path_navigation_widget.hpp"
#include "nau_static_text_label.hpp"
#include "nau_buttons.hpp"
#include "nau_flow_layout.hpp"
#include "themes/nau_theme.hpp"


// ** NauPathNavigationWidget

NauPathNavigationWidget::NauPathNavigationWidget(NauWidget* parent)
    : NauWidget(parent)
{
    auto layout = new NauFlowLayout(Qt::LeftToRight, 0, 4, 4, this);
    setLayout(layout);
    setMinimumHeight(24);
}

void NauPathNavigationWidget::setNavigationChain(const NauDir& path)
{
    clearNavigationChain();

    if (path != NauDir(QStringLiteral("."))) {
        m_pathParts << NauDir::toNativeSeparators(path.path()).split(NauDir::separator());
    }

    for (int idx = 0; idx < m_pathParts.size(); ++idx) {
        auto itemButton = new NauMiscButton(this);

        if (idx > 0) {
            itemButton->setText(m_pathParts[idx]);
            itemButton->setObjectName(QStringLiteral("NavigationLabelItem"));

        } else {
            itemButton->setIcon(Nau::Theme::current().iconBreadCrumbsHome());
            itemButton->setObjectName(QStringLiteral("NavigationLabelHome"));
            itemButton->setFixedWidth(16);
        }

        itemButton->setFixedHeight(16);
        itemButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
        itemButton->setCursor(Qt::PointingHandCursor);
        itemButton->setFont(Nau::Theme::current().fontProjectBrowserSecondary());
        layout()->addWidget(itemButton);

        connect(itemButton, &NauMiscButton::clicked,
            std::bind(&NauPathNavigationWidget::emitChangeDirectoryRequest, this, idx + 1));

        auto delimiterLabel = new NauMiscButton(this);
        delimiterLabel->setIcon(Nau::Theme::current().iconBreadCrumbsDelimiter());
        delimiterLabel->setFixedHeight(16);
        delimiterLabel->setObjectName(QStringLiteral("NavigationLabelItemDelimitter"));

        layout()->addWidget(delimiterLabel);
    }

    layout()->addItem(new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum));
}

void NauPathNavigationWidget::clearNavigationChain()
{
    QLayoutItem* item = nullptr;
    while ((item = layout()->takeAt(0)) != nullptr) {
        if (item->widget()) {
            delete item->widget();
        }
        delete item;
    }

    m_pathParts = {QStringLiteral(".")};
}

void NauPathNavigationWidget::emitChangeDirectoryRequest(int pathIdx)
{
    const QString path = QStringList(m_pathParts.begin(),
        std::next(m_pathParts.begin(), pathIdx)).join(NauDir::separator());

    emit changeDirectoryRequested(path);
}
