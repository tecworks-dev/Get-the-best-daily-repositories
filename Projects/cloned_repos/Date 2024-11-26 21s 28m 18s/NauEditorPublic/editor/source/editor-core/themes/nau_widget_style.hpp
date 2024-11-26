// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// This file will contain the underlying data structures that are involved in setting the style of the widget.

#pragma once

#include "nau_font.hpp"
#include "nau_color.hpp"
#include "baseWidgets/nau_icon.hpp"

#include <unordered_map>


// ** NauWidgetState
//
// Describes the main states in which the widget can be in.

enum class NauWidgetState
{
    Active,
    Hover,
    Pressed,
    TabFocused,
    Error,
    Disabled    // TODO: It is possible to generate a self style so that the developer doesn't have to fill in the full range of cases.
};


// ** NauWidgetStyle
//
// Structure that generally describes the style of the widget.

struct NauWidgetStyle
{
    // ** NauStyle
    //
    // Structure that describes the visual of a widget for its particular state.
    struct NauStyle
    {
        // Text settings
        NauColor textColor;

        // Icon status
        NauIcon::Mode iconState;

        // Widget background color
        NauBrush background;

        // Widget border corner radius
        QSize radiusSize;

        // Outline pen
        NauPen outlinePen;
    };

    std::unordered_map<NauWidgetState, NauStyle> styleByState;
};
