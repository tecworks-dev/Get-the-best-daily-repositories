// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_game_viewport_controller.hpp"
#include "nau/viewport/nau_viewport_utils.hpp"

#include "nau/nau_constants.hpp"
#include "nau/viewport/nau_viewport_widget.hpp"
#include "nau/selection/nau_object_selection.hpp"

#include "nau/editor-engine/nau_editor_engine_services.hpp"
#include "nau/service/service_provider.h"

#include <nau/diag/logging.h>

#include <QApplication>


// ** NauPlayModeNativeEventFilter

NauPlayModeNativeEventFilter::NauPlayModeNativeEventFilter(NauViewportWidget* linkedViewport)
    : m_linkedViewport(linkedViewport)
{

}

bool NauPlayModeNativeEventFilter::nativeEventFilter(const QByteArray& eventType, void* message, qintptr*)
{
    if (eventType == "windows_generic_MSG" || eventType == "windows_dispatcher_MSG") {
        if (m_linkedViewport && m_linkedViewport->controller()) {
            m_linkedViewport->controller()->processNativeEvent(message);
        }
    }

    return false;
}


// ** NauGameViewportController

NauGameViewportController::NauGameViewportController(NauViewportWidget* viewport)
    : NauBaseViewportController(viewport)
    , m_nativeEventFilter(viewport)
{
    qApp->installNativeEventFilter(&m_nativeEventFilter);

    this->viewport()->setMouseTracking(true);
    this->viewport()->setFocus();
    showCursor(true);
    moveCursorToCenter();
    this->viewport()->grabMouse();

    m_appMessageHandlers = nau::getServiceProvider().getAll<nau::IWindowsApplicationMessageHandler>();
}

NauGameViewportController::~NauGameViewportController()
{
    qApp->removeNativeEventFilter(&m_nativeEventFilter);
    onFocusOut();
}

void NauGameViewportController::processNativeEvent(void* message)
{
    // Do not process native events if game viewport unfocused
    if (!m_isFocused) {
        return;
    }

    // TODO: need engine cross platform API
#ifdef Q_OS_WIN
    MSG* pMsg = reinterpret_cast<MSG*>(message);

    // handle message
    for (nau::IWindowsApplicationMessageHandler* const handler : m_appMessageHandlers) {
        const auto res = handler->preDispatchMsg(*pMsg);
    }

    for (nau::IWindowsApplicationMessageHandler* const handler : m_appMessageHandlers) {
        handler->postDispatchMsg(*pMsg);
    }

#endif
}

bool NauGameViewportController::processEvent(QEvent* event)
{
    switch (event->type()) {
    case QEvent::ShortcutOverride:
        event->accept();
        break;
    case QEvent::KeyPress:
        onKeyDown(static_cast<QKeyEvent*>(event));
        break;
    case QEvent::MouseButtonPress:
        onMouseButtonPress(static_cast<QMouseEvent*>(event));
        break;
    case QEvent::MouseMove:
        onMouseMove(static_cast<QMouseEvent*>(event));
        break;
    case QEvent::Resize: {
        const float dpi = viewport()->devicePixelRatioF();
        NauBaseViewportController::onResize(static_cast<QResizeEvent*>(event), dpi);
        break;
    }
    case QEvent::FocusOut:
        onFocusOut();
        break;
    default:
        // Event not handled in this controller
        return false;
    }

    // Event handled
    return true;
}

void NauGameViewportController::onMouseButtonPress(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_isFocused = true;
        viewport()->grabMouse();
    }
}


void NauGameViewportController::onMouseMove(QMouseEvent* event)
{
    if (!m_isFocused) {
        return;
    }

    // Keep cursor hided if viewport in game mode and focused
    keepCursorInViewport(event);
}

void NauGameViewportController::onKeyDown(QKeyEvent* event)
{
    // Unfocus game viewport
    if ((event->key() == Qt::Key_F1) && (event->modifiers() == Qt::KeyboardModifier::ShiftModifier)) {
        onFocusOut();
    }

    if (event->key() == Qt::Key_Escape) {
        emit viewport()->eventEscPressed();
    }
}

void NauGameViewportController::moveCursorToCenter()
{
    const QPoint center = viewport()->geometry().center();
    const QPoint globalCenter = viewport()->mapToGlobal(center);
    moveCursor(globalCenter, m_isFocused);
}

void NauGameViewportController::keepCursorInViewport(QMouseEvent* event)
{
    const QPoint localTopLeft(0, 0);
    const QPoint localBottomRight(viewport()->size().width(), viewport()->size().height());

    const QPoint topLeft = viewport()->mapToGlobal(localTopLeft);
    const QPoint bottomRight = viewport()->mapToGlobal(localBottomRight);

    QPoint cursorPosition = event->globalPosition().toPoint();
    
    bool needForceMove = false;
    if (cursorPosition.x() < topLeft.x()) {
        cursorPosition.setX(topLeft.x());
        needForceMove = true;
    }

    if (cursorPosition.x() > bottomRight.x()) {
        cursorPosition.setX(bottomRight.x());
        needForceMove = true;
    }

    if (cursorPosition.y() < topLeft.y()) {
        cursorPosition.setY(topLeft.y());
        needForceMove = true;
    }

    if (cursorPosition.y() > bottomRight.y()) {
        cursorPosition.setY(bottomRight.y());
        needForceMove = true;
    }

    moveCursor(cursorPosition, needForceMove);
}

void NauGameViewportController::onFocusOut()
{
    m_isFocused = false;
    viewport()->releaseMouse();
}