// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "nau/nau_editor_config.hpp"

#include <QFont>


// ** NauFont

class NAU_EDITOR_API NauFont : public QFont
{
public:
    NauFont() = default;
    NauFont(const QString &family, int pointSize = -1, int weight = -1, bool italic = false);
};