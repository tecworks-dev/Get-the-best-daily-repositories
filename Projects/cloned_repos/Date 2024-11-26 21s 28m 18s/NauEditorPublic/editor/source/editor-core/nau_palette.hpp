// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A container of colors and brushes associated with different roles, categories and states.

#pragma once

#include "nau/nau_editor_config.hpp"
#include "nau_color.hpp"

#include <bitset>


// ** NauPalette
// ToDo: cover this with tests.

class NAU_EDITOR_API NauPalette
{
public:
    enum class Role
    {
        Background,
        AlternateBackground,

        BackgroundHeader,
        AlternateBackgroundHeader,

        BackgroundFooter,

        Foreground,
        AlternateForeground,
        ForegroundHeader,
        ForegroundBrightHeader,
        ForegroundBrightText,

        Border,
        BorderHeader,

        Text,
        TextHeader
    };

    enum State
    {
        Normal   = 0b00000000,
        Selected = 0b00000001,
        Hovered  = 0b00000010,
        Pressed  = 0b00000100,
        Error    = 0b00001000,
        Flashing = 0b00010000,
    };

    enum class Category
    {
        // Category of colors/brushes for objects that has keyboard focus.
        Active = 0,

        // Category of colors/brushes for objects that's not active.
        Inactive = 1,

        // Category of colors/brushes for disabled state.
        Disabled = 2,
    };

    // Set specified brush for role, state and category.
    // Given brush also will be set to other categories, if for these categories haven't set a brush for this role and state.
    void setBrush(Role role, NauBrush brush, int state = State::Normal, Category category = Category::Active);

    // Set specified color for role, state and category.
    // Given color also will be set to other categories, if for these categories haven't set a color for this role and state.
    void setColor(Role role, NauColor color, int state = State::Normal, Category category = Category::Active);

    // Returns color for specified role, state and category or invalid color if not found.
    NauColor color(Role role, int state = State::Normal, Category category = Category::Active) const;

    // Returns brush for specified role, state and category or NauBrush::NoBrush if not found.
    NauBrush brush(Role role, int state = State::Normal, Category category = Category::Active) const;

    // Has this palette any color or brush specified.
    bool empty() const;

private:
    static void reportWarning(const char* msg, Role role, int stateMask, Category category);

    friend bool operator==(const NauPalette& lhs, const NauPalette& rhs);

    // Iterates circularly over known categories.
    static Category nextCategory(Category category);

    void setBrushInternal(Role role, NauBrush brush, int state, Category category, bool overwrite);
    void setColorInternal(Role role, NauColor color, int state, Category category, bool overwrite);

    template<typename Container>
    void setContainerInternalImpl(Container& container, Role role, typename Container::mapped_type value,
        int stateMask, Category category, bool overwrite);

private:
    using ItemKey = std::bitset<64>;

    // Generates a unique runtime key using specified role, state, category.
    static ItemKey makeUniqueKey(Role role, int state, Category category);

    std::unordered_map<ItemKey, NauColor> m_colors;
    std::unordered_map<ItemKey, NauBrush> m_brushes;
};
