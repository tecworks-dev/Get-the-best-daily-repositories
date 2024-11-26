// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_font.hpp"


// ** NauFont

NauFont::NauFont(const QString& family, int pointSize, int weight, bool italic)
    : QFont(family, pointSize, weight, italic)
{
}
