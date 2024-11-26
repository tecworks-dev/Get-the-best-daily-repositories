// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_font_metrics.hpp"
#include "nau_log.hpp"


// ** NauFontMetrics

NauFontMetrics::NauFontMetrics(const NauFont& font)
    : QFontMetrics(font)
{
    const bool fontSizeResolved = font.resolveMask() & QFont::SizeResolved;
    if (!fontSizeResolved) {
        NED_WARNING("NauFontMetrics calculates horizontal advance for a text "
            "only if its size specified explicitly. Font {}", font.family().toLocal8Bit().constData());
    }
}
