// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_log_proxy_model.hpp"
#include "nau_log_model.hpp"
#include "nau_log.hpp"
#include "nau_plus_enum.hpp"


// ** NauLogProxyModel

NauLogProxyModel::NauLogProxyModel(QObject* parent)
    : QSortFilterProxyModel(parent)
{
    setFilterKeyColumn(+NauLogModel::Column::Message);
}

void NauLogProxyModel::setLogLevelFilter(const std::vector<NauLogLevel>& preferredLevels)
{
    m_levelFilters = preferredLevels;
    invalidateFilter();
}

void NauLogProxyModel::setLogSourceFilter(const QStringList& logSourceNames)
{
    m_logSourceNames = logSourceNames;
    invalidateFilter();
}

bool NauLogProxyModel::filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const
{
    auto logModel = dynamic_cast<NauLogModel*>(sourceModel());
    if (!logModel) {
        return QSortFilterProxyModel::filterAcceptsRow(sourceRow, sourceParent);
    }

    const QModelIndex index = sourceModel()->index(sourceRow, 0, sourceParent);

    if (!m_levelFilters.empty()) {
        // User specified the preferred list of log levels, so
        // the log level of current message must be in the list.
        const auto level = static_cast<NauLogLevel>(index.data(NauLogModel::LevelRole).toInt());

        auto itLevel = std::find(m_levelFilters.begin(), m_levelFilters.end(), level);
        if (itLevel == m_levelFilters.end()) {
            return false;
        }
    }

    if (!m_logSourceNames.isEmpty()) {
        // User specified the preferred list of log sources, so
        // the log source of current message must be in the list.
        const auto currentSourceName = index.data(NauLogModel::SourceRole).toString();

        // Note that matching the source doesn't mean current message is acceptable.
        // Other conditions may be checked below.
        bool sourceFound = false;
        for (const auto& filterSourceName : m_logSourceNames) {
            if (filterSourceName.compare(currentSourceName, filterCaseSensitivity()) == 0) {
                sourceFound = true;
                break;
            }
        }
        if (!sourceFound) {
            return false;
        }
    }

    return QSortFilterProxyModel::filterAcceptsRow(sourceRow, sourceParent);
}
