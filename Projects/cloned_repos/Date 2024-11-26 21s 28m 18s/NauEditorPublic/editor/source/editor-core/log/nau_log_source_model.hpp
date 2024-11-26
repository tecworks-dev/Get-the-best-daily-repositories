// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Data model in terms of Qt Model/View requirements for
// operating sources of logging.

#pragma once

#include "baseWidgets/nau_icon.hpp"
#include "nau_log_source_model_items.hpp"

#include <QCoreApplication>
#include <QAbstractItemModel>
#include <vector>
#include <deque>
#include <memory>


// ** NauLogSourceModel
// Builds data model as follows:
// RootSourceItem
//     EditorSourceItem
//     BuildSourceItem
//     PlayModeSourceItem
//     ExternalSourceItem
//     ImportedSourceItem_1
//     ...
//     ImportedSourceItem_N

class NauLogSourceModel : public QAbstractItemModel
{
public:
    NauLogSourceModel(QObject* parent = nullptr);

    enum Roles
    {
        SourceNamesRole = Qt::UserRole + 1
    };

    enum class Column
    {
        SourceName
    };

    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex& index) const  override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;

private:
    std::unique_ptr<NauSourceTreeModelItems::RootSourceItem> m_root;
};

