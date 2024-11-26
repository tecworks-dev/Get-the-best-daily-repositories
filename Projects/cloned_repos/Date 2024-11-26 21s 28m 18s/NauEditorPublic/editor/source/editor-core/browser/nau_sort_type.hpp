// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Types of sorting in the project browser.

#pragma once


// ** NauSortType

enum class NauSortType
{
    Name,           // Sorting by the item's display text.
    Type,           // See NauEditorFileType.
    ModifiedTime,   // Sorting by the item's last modified time.
    Size,            // Sorting by the items' size.
    Path            // Sorting by the path' size.
};