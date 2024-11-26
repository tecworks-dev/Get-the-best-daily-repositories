// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "nau_font.hpp"

#include <QFontMetrics>


// ** NauFontMetrics

class NAU_EDITOR_API NauFontMetrics : public QFontMetrics
{
public:
    explicit NauFontMetrics(const NauFont& font);
};