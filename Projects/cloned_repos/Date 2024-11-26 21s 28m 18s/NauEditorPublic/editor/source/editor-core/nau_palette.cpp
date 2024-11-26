// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_palette.hpp"
#include "nau_assert.hpp"
#include "nau_log.hpp"
#include "nau_plus_enum.hpp"

#include "magic_enum/magic_enum.hpp"
#include "magic_enum/magic_enum_flags.hpp"
#include "magic_enum/magic_enum_containers.hpp"
#include "magic_enum/magic_enum_iostream.hpp"


// Required for magic_enum to output state flags. Note that it cannot be in anonymous namespace.
template <>
struct magic_enum::customize::enum_range<NauPalette::State>
{
    static constexpr bool is_flags = true;
};


// ** NauPalette

void NauPalette::setBrush(Role role, NauBrush brush, int stateMask, Category category)
{
    setBrushInternal(role, brush, stateMask, category, true);

    setBrushInternal(role, brush, stateMask, nextCategory(category), /*overwrite*/false);
    setBrushInternal(role, brush, stateMask, nextCategory(nextCategory(category)), /*overwrite*/false);
}

void NauPalette::setColor(Role role, NauColor color, int stateMask, Category category)
{
    setColorInternal(role, color, stateMask, category, /*overwrite*/true);

    setColorInternal(role, color, stateMask, nextCategory(category), /*overwrite*/false);
    setColorInternal(role, color, stateMask, nextCategory(nextCategory(category)), /*overwrite*/false);
}

NauColor NauPalette::color(Role role, int stateMask, Category category) const 
{
    const auto key = makeUniqueKey(role, stateMask, category);

    auto itColor = m_colors.find(key);
    if (itColor == m_colors.cend()) {
        return {};
    }

    return itColor->second;
}

NauBrush NauPalette::brush(Role role, int stateMask, Category category) const
{
    const auto key = makeUniqueKey(role, stateMask, category);
    auto itBrush = m_brushes.find(key);
    if (itBrush == m_brushes.cend()) {
        return {};
    }

    return itBrush->second;
}

bool NauPalette::empty() const
{
    return m_colors.empty() && m_brushes.empty();
}

void NauPalette::reportWarning(const char* msg, Role role, int stateMask, Category category)
{
#ifdef DEBUG
    using namespace magic_enum;
    NED_INFO("{}: role='{}', state='{}', category='{}'", msg, enum_name(role),
        (stateMask == State::Normal ? "Normal" : enum_flags_name(static_cast<State>(stateMask))), enum_name(category));
#endif // DEBUG
}

NauPalette::Category NauPalette::nextCategory(Category category)
{
    if (category == Category::Disabled) {
        return Category::Active;
    }

    return static_cast<Category>((+category) + 1);
}

void NauPalette::setBrushInternal(Role role, NauBrush brush, int stateMask, Category category, bool overwrite)
{
    setContainerInternalImpl(m_brushes, role, brush, stateMask, category, overwrite);
}

void NauPalette::setColorInternal(Role role, NauColor color, int stateMask, Category category, bool overwrite)
{
    setContainerInternalImpl(m_colors, role, color, stateMask, category, overwrite);
}

template<typename Container>
void NauPalette::setContainerInternalImpl(Container& container, Role role,
    typename Container::mapped_type value, int stateMask, Category category, bool overwrite)
{
    const auto key = makeUniqueKey(role, stateMask, category);
    auto itRecord = container.find(key);
    if (itRecord == container.end()) {
        container.insert({ key, value });
        return;
    }

    if (overwrite) {
        reportWarning("NauPalette overwrites already set color/brush", role, stateMask, category);
        itRecord->second = value;
    }
}

NauPalette::ItemKey NauPalette::makeUniqueKey(Role role, int state, Category category)
{
    // Pack role, state and category into 64 bitset.
    // * 16 bits are for color/brush role.
    // * 16 bits are for color/brush category.
    // * 32 bits are for color/brush state.
    static_assert(sizeof(ItemKey) == 8);

    return (ItemKey{static_cast<std::uint64_t>(role)} << 48) |
        (ItemKey{static_cast<std::uint64_t>(category)} << 32) |
        ItemKey{static_cast<std::uint64_t>(state)};
}

bool operator==(const NauPalette& lhs, const NauPalette& rhs)
{
    return lhs.m_colors == rhs.m_colors && lhs.m_brushes == rhs.m_brushes;
}
