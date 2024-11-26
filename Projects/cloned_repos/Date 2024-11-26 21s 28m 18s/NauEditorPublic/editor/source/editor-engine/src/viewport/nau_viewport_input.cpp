// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/viewport/nau_viewport_input.hpp"


// ** NauViewportInput

bool NauViewportInput::isMouseButtonDown(Qt::MouseButton key) const
{
    return m_mouseBtnMap.contains(key) ? m_mouseBtnMap.at(key) : false;
}

void NauViewportInput::setMouseButtonDown(Qt::MouseButton key, bool isDown)
{
    m_mouseBtnMap[key] = isDown;
}

bool NauViewportInput::isKeyDown(Qt::Key key) const
{
    return m_keyMap.contains(key) ? m_keyMap.at(key) : false;
}

void NauViewportInput::setKeyDown(Qt::Key key, bool isDown)
{
    m_keyMap[key] = isDown;
}

float NauViewportInput::deltaWheel() const
{
    return m_deltaWheel;
}

void NauViewportInput::setDeltaWheel(float deltaWheel)
{
    m_deltaWheel = deltaWheel;
}

QPointF NauViewportInput::deltaMouse() const
{
    return m_deltaMouse;
}

void NauViewportInput::setDeltaMouse(QPointF deltaMouse)
{
    m_deltaMouse = deltaMouse;
}