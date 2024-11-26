// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Custom types for editor.

#pragma once


#include <cmath>
#include <limits>


template<class T>
class NauRangedValue
{
public:
    constexpr NauRangedValue() noexcept = default;
    constexpr explicit NauRangedValue(T value) noexcept
    {
        setValue(value);
    }
    constexpr NauRangedValue(T minimum, T maximum) noexcept
    {
        setRange(minimum, maximum);
    }
    constexpr NauRangedValue(T value, T minimum, T maximum) noexcept
    {
        setRange(minimum, maximum);
        setValue(value);
    }

    constexpr T value() const noexcept { return m_value; }
    constexpr T minimum() const noexcept { return m_minimum; }
    constexpr T maximum() const noexcept { return m_maximum; }

    constexpr void setValue(T value) noexcept
    {
        m_value = std::clamp(value, m_minimum, m_maximum);
    }
    constexpr void setMinimum(T minimum) noexcept
    {
        m_minimum = minimum;
        m_maximum = std::max(minimum, m_maximum);
        setValue(m_value);
    }
    constexpr void setMaximum(T maximum) noexcept
    {
        m_minimum = std::min(maximum, m_minimum);
        m_maximum = maximum;
        setValue(m_value);
    }
    constexpr void setRange(T minimum, T maximum) noexcept
    {
        setMinimum(minimum);
        setMaximum(maximum);
    }

    constexpr bool operator == (const NauRangedValue& rhs) const noexcept
    {
        return (m_minimum == rhs.m_minimum) && (m_maximum == rhs.m_maximum) && (m_value == rhs.m_value);
    }

private:
    T m_minimum = std::numeric_limits<T>::min();
    T m_maximum = std::numeric_limits<T>::max();
    T m_value = std::clamp(T(), m_minimum, m_maximum);
};


template<class T>
class NauRangedPair
{
public:
    constexpr NauRangedPair() noexcept = default;
    constexpr NauRangedPair(T left, T right) noexcept
    {
        setPair(left, right);
    }
    constexpr NauRangedPair(T left, T right, T minimum, T maximum) noexcept
    {
        setRange(minimum, maximum);
        setPair(left, right);
    }

    constexpr T left() const noexcept { return m_left; }
    constexpr T right() const noexcept { return m_right; }
    constexpr T minimum() const noexcept { return m_minimum; }
    constexpr T maximum() const noexcept { return m_maximum; }

    constexpr void setPair(T left, T right) noexcept
    {
        m_left = std::clamp(left, m_minimum, m_maximum);
        m_right = std::max(m_left, std::clamp(right, m_minimum, m_maximum));
    }
    constexpr void setMinimum(T minimum) noexcept
    {
        m_minimum = minimum;
        m_maximum = std::max(minimum, m_maximum);
        setPair(m_left, m_right);
    }
    constexpr void setMaximum(T maximum) noexcept
    {
        m_minimum = std::min(maximum, m_minimum);
        m_maximum = maximum;
        setPair(m_left, m_right);
    }
    constexpr void setRange(T minimum, T maximum) noexcept
    {
        setMinimum(minimum);
        setMaximum(maximum);
    }

    constexpr bool operator == (const NauRangedPair& rhs) const noexcept
    {
        return (m_minimum == rhs.m_minimum) && (m_maximum == rhs.m_maximum) && (m_left == rhs.m_left) && (m_right == rhs.m_right);
    }

private:
    T m_minimum = std::numeric_limits<T>::min();
    T m_maximum = std::numeric_limits<T>::max();
    T m_left = std::clamp(T(), m_minimum, m_maximum);
    T m_right = m_left;
};
