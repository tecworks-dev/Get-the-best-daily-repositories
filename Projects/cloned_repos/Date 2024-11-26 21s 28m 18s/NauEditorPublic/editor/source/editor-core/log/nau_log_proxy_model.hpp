// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Data proxy model of the log/console widget.

#pragma once

#include "nau_log.hpp"
#include <QSortFilterProxyModel>


// ** NauLogProxyModel

class NauLogProxyModel : public QSortFilterProxyModel
{
    Q_OBJECT
public:
    NauLogProxyModel(QObject* parent = nullptr);

    // Updates current filter by the level of log messages.
    void setLogLevelFilter(const std::vector<NauLogLevel>& preferredLevels);

    // Updates current filter by the source of log messages.
    void setLogSourceFilter(const QStringList& logSourceNames);

    bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const override;

private:
    std::vector<NauLogLevel> m_levelFilters;
    QStringList m_logSourceNames;
};