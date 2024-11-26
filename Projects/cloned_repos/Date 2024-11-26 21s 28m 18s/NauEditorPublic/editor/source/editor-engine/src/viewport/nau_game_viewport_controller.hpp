// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport controller implementations for playmode

#pragma once

#include "nau/viewport/nau_base_viewport_controller.hpp"

#include "nau/viewport/nau_camera_controller.hpp"

#include <QWidget>
#include <QEvent>
#include <QEasingCurve>
#include <QResizeEvent>
#include <QAbstractNativeEventFilter>

#include <EASTL/vector.h>
#include "nau/platform/windows/app/window_message_handler.h"
#include <nau/math/math.h>


// ** NauPlayModeNativeEventFilter
//
// Qt misses some native events. This is a workround

class NAU_EDITOR_ENGINE_API NauPlayModeNativeEventFilter : public QAbstractNativeEventFilter
{
public:
    NauPlayModeNativeEventFilter(NauViewportWidget* linkedViewport);

    virtual bool nativeEventFilter(const QByteArray& eventType, void* message, qintptr*) Q_DECL_OVERRIDE;

private:
    NauViewportWidget* m_linkedViewport = nullptr;
};


// ** NauGameViewportController
//
// Viewport editor controller class
// Overrides base class input events handling for game mode

class NauGameViewportController : public NauBaseViewportController
{
public:
    NauGameViewportController(NauViewportWidget* viewport);
    virtual ~NauGameViewportController();

    virtual bool processEvent(QEvent* event) override;
    virtual void processNativeEvent(void* message) override;

private:
    virtual void onMouseButtonPress(QMouseEvent* event);
    virtual void onMouseMove(QMouseEvent* event);
    virtual void onKeyDown(QKeyEvent* event);

    void moveCursorToCenter();
    void keepCursorInViewport(QMouseEvent* event);
    void onFocusOut();

private:
    bool m_isFocused = true;

    // message handlers
    eastl::vector<nau::IWindowsApplicationMessageHandler*> m_appMessageHandlers;
    NauPlayModeNativeEventFilter m_nativeEventFilter;
};
