// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_icon.hpp"


// ** NauIcon

NauIcon::NauIcon(const QPixmap& pixmap)
    : QIcon(pixmap)
{
}

NauIcon::NauIcon(const QString& fileName)
    : QIcon(fileName)
{
}
