// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_log_tree_view_item_delegate.hpp" 
#include "nau_log_constants.hpp"

#include <QDateTime>


// ** NauLogTreeViewItemDelegate

NauLogTreeViewItemDelegate::NauLogTreeViewItemDelegate(QObject* parent)
    : NauTreeViewItemDelegate(parent)
{
}

QString NauLogTreeViewItemDelegate::displayText(const QVariant& value, const QLocale& locale) const
{
    if (value.metaType() == QMetaType::fromType<QDateTime>()) {
        return value.value<QDateTime>().toString(NauLogConstants::dateTimeDisplayFormat());
    }

    return QStyledItemDelegate::displayText(value, locale);
}

QSize NauLogTreeViewItemDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    static const int defaultSectionSize = 64;
    return QSize{ defaultSectionSize, NauLogConstants::messageItemHeight() };
}
