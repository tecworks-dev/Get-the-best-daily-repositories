// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Custom view delegate for logger item view.

#pragma once

#include "nau_palette.hpp"
#include "nau_tree_view_item_delegate.hpp"


// ** NauLogTreeViewItemDelegate

class NauLogTreeViewItemDelegate : public NauTreeViewItemDelegate
{
public:
    explicit NauLogTreeViewItemDelegate(QObject* parent = nullptr);

    QString displayText(const QVariant& value, const QLocale& locale) const override;

    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;
};
