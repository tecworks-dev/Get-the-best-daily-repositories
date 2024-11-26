// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// View attributes classes

#pragma once

#include "baseWidgets/nau_widget.hpp"

#include <QStringList>


// ** ResourceSelectorType

enum class NauStringViewAttributeType {
    DEFAULT,
    MODEL,
    MATERIAL,
    TEXTURE
};


// ** NauResourceSelector

class NauResourceSelector
{
public:
    static QStringList resourceList(NauStringViewAttributeType type);
};
