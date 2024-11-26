// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include <Qt>


// ** NauSortOrder

enum class NauSortOrder
{
    Ascending = Qt::SortOrder::AscendingOrder,
    Descending = Qt::SortOrder::DescendingOrder
};

namespace Nau
{
    inline Qt::SortOrder toQtSortOrder(NauSortOrder order)
    {
        return order == NauSortOrder::Ascending ? Qt::AscendingOrder : Qt::DescendingOrder;
    }

    inline NauSortOrder fromQtSortOrder(Qt::SortOrder order)
    {
        return order == Qt::AscendingOrder ? NauSortOrder::Ascending : NauSortOrder::Descending;
    }
}