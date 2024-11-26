// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_project_browser_summary_widget.hpp"
#include "nau_project_browser_file_system_model.hpp"

#include <QTextStream>


// ** NauProjectBrowserSummaryWidget

NauProjectBrowserSummaryWidget::NauProjectBrowserSummaryWidget(NauWidget* parent)
    : NauLabel(QString(), parent)
{
}

void NauProjectBrowserSummaryWidget::setRootIndex(const QModelIndex& index)
{
    m_itemsCount = index.isValid() && index.model() ? index.model()->rowCount(index) : 0;
    m_selection.clear();

    updateUI();
}

void NauProjectBrowserSummaryWidget::onSelectionChange(const QModelIndexList& selected)
{
    m_selection = selected;
    updateUI();
}

void NauProjectBrowserSummaryWidget::updateUI()
{
    QString text;
    QTextStream str(&text);

    str << tr("%n item(s)", nullptr, m_itemsCount);

    if (!m_selection.isEmpty()) {
        str << tr(" (%n selected, %1)", nullptr, m_selection.size())
            .arg(getFormattedSelectionSize());
    }

    setText(text);
}

QString NauProjectBrowserSummaryWidget::getFormattedSelectionSize() const
{
    long long size = 0;
    for (const auto& selection : m_selection) {
        size += selection.data(NauProjectBrowserFileSystemModel::FileSizeRole).toLongLong();
    }

    return QLocale().formattedDataSize(size, 2, QLocale::DataSizeTraditionalFormat);
}
