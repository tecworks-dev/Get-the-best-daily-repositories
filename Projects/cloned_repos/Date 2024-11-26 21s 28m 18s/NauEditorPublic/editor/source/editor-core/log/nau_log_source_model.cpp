// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_log_source_model.hpp"

#include "magic_enum/magic_enum.hpp"


// ** NauLogSourceModel

NauLogSourceModel::NauLogSourceModel(QObject* parent)
    : QAbstractItemModel(parent)
{
    m_root = std::make_unique<NauSourceTreeModelItems::RootSourceItem>();
}

int NauLogSourceModel::columnCount(const QModelIndex& parent) const
{
    return magic_enum::enum_count<NauLogSourceModel::Column>();
}

QVariant NauLogSourceModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid()) {
        return QVariant();
    }

    return static_cast<NauSourceTreeModelItems::Node*>(index.internalPointer())->data(role);
}

Qt::ItemFlags NauLogSourceModel::flags(const QModelIndex& index) const
{
    return static_cast<NauSourceTreeModelItems::Node*>(index.internalPointer())->flags();
}

QModelIndex NauLogSourceModel::index(int row, int column, const QModelIndex& parent) const
{
    if (!hasIndex(row, column, parent)) {
        return QModelIndex();
    }

    NauSourceTreeModelItems::Node* const node = !parent.isValid()
        ? m_root.get()
        : static_cast<NauSourceTreeModelItems::Node*>(parent.internalPointer());

    NauSourceTreeModelItems::Node* const child = node->childNode(row);

    return child ? createIndex(row, column, child) : QModelIndex();
}

QModelIndex NauLogSourceModel::parent(const QModelIndex& index) const
{
    if (!index.isValid()) {
        return QModelIndex();
    }

    auto parentItem = static_cast<NauSourceTreeModelItems::Node*>(index.internalPointer())->parent();

    if (!parentItem || parentItem == m_root.get()) {
        return QModelIndex();
    }

    return createIndex(parentItem->row(), 0, parentItem);
}

int NauLogSourceModel::rowCount(const QModelIndex& parent) const
{
    NauSourceTreeModelItems::Node* const parentItem = !parent.isValid()
        ? m_root.get()
        : static_cast<NauSourceTreeModelItems::Node*>(parent.internalPointer());

    return parentItem->childCount();
}

bool NauLogSourceModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    return false;
}
