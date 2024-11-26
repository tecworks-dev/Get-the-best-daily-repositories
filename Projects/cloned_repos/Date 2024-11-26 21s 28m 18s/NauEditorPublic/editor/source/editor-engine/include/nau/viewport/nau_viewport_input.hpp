// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport input container

#pragma once

#include "nau/nau_editor_engine_api.hpp"

#include <unordered_map>
#include <QPoint>
#include <QEvent>


// ** NauViewportInput

class NAU_EDITOR_ENGINE_API NauViewportInput
{
public: 
    bool isMouseButtonDown(Qt::MouseButton key) const;
    void setMouseButtonDown(Qt::MouseButton key, bool isDown);

    bool isKeyDown(Qt::Key key) const;
    void setKeyDown(Qt::Key key, bool isDown);

    float deltaWheel() const;
    void setDeltaWheel(float deltaWheel);

    QPointF deltaMouse() const;
    void setDeltaMouse(QPointF deltaMouse);

private:
    // pressed mouse buttons
    std::unordered_map<Qt::MouseButton, bool> m_mouseBtnMap;
    // pressed key buttons
    std::unordered_map<Qt::Key, bool> m_keyMap;
    // current mouse delta from previous tick
    QPointF m_deltaMouse;
    // current wheel delta from previous tick
    float m_deltaWheel = 0;
};