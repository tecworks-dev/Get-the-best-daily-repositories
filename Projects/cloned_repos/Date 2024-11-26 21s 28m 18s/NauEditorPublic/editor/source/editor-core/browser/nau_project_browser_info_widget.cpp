// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_project_browser_info_widget.hpp"
#include "nau_project_browser_file_system_model.hpp"
#include "themes/nau_theme.hpp"

#include <QClipboard>
#include <QToolTip>
#include <QTextStream>


// ** NauProjectBrowserInfoWidget

NauProjectBrowserInfoWidget::NauProjectBrowserInfoWidget(NauWidget* parent)
    : NauWidget(parent)
{
    auto layout = new NauLayoutHorizontal(this);

    m_copyButton = new NauMiscButton(this);
    m_copyButton->setIcon(Nau::Theme::current().iconCopy());
    m_copyButton->setToolTip(tr("Copy full path to selected files"));
    m_copyButton->setFixedSize(16, 16);

    m_label = new NauStaticTextLabel(QString(), this);
    m_label->setFont(Nau::Theme::current().fontProjectBrowserSecondary());
    m_label->setFixedHeight(16);

    layout->setSpacing(8);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_copyButton);
    layout->addWidget(m_label, 1);

    updateUI();

    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    connect(m_copyButton, &NauToolButton::clicked, 
        this, &NauProjectBrowserInfoWidget::copySelectionToClipboard);
}

void NauProjectBrowserInfoWidget::setRootIndex(const QModelIndex& rootIndex)
{
    m_rootDirectory = rootIndex.isValid()
        ? rootIndex.data(NauProjectBrowserFileSystemModel::FilePathRole).toString()
        : QString();

    m_selection.clear();

    updateUI();
}

void NauProjectBrowserInfoWidget::onSelectionChange(const QModelIndexList& selected)
{
    m_selection = selected;
    updateUI();
}

void NauProjectBrowserInfoWidget::updateUI()
{
    m_copyButton->setVisible(!m_selection.isEmpty());

    if (m_selection.size() == 1) {
        const QString relPath = m_selection.front()
            .data(NauProjectBrowserFileSystemModel::FileRelativePathRole).toString();

        // TODO Show the guid of this resource.
        m_label->setText(relPath);

    } else if (m_selection.size() > 1) {
        m_label->setText(QString("..."));

    } else if (m_selection.isEmpty()) {
        m_label->setText(QString());

    }
}

void NauProjectBrowserInfoWidget::copySelectionToClipboard()
{
    QString paths;
    QTextStream stream(&paths);

    for (const QModelIndex& index : m_selection) {
        stream << NauDir::toNativeSeparators(index
            .data(NauProjectBrowserFileSystemModel::FilePathRole).toString()) << Qt::endl;
    }

    QGuiApplication::clipboard()->setText(paths);
    QToolTip::showText(QCursor::pos(), tr("Copied"), m_copyButton, {}, 1000);
}
