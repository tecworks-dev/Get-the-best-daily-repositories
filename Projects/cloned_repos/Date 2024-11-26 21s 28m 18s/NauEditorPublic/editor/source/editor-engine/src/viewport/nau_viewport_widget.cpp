// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/viewport/nau_viewport_widget.hpp"

#include <QApplication>


constexpr int FPS_LIMIT = 60.0f;
constexpr int MS_PER_FRAME = (int)((1.0f / FPS_LIMIT) * 1000.0f);


// ** NauViewportWidget

NauViewportWidget::NauViewportWidget(QWidget* parent)
    : QWidget(parent)
    , m_hWnd(reinterpret_cast<HWND>(winId()))
{
    setObjectName("viewportWidget");
    QPalette pal = palette();
    pal.setColor(QPalette::Window, Qt::black);
    setAutoFillBackground(true);
    setPalette(pal);
    setFocusPolicy(Qt::StrongFocus);
    setAttribute(Qt::WA_NativeWindow);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // Viewport as a field where the render is output implies some minimum size.
    // Collapsing to zero size(for example in result of ui docking/undocking)
    // is meaningless and leads to engine's crashing. So we have to have some minimum guaranteed size.
    setMinimumSize(192, 144);
}

NauViewportWidget::~NauViewportWidget()
{

}

void NauViewportWidget::onFrame()
{
    if (m_viewportController) {
        m_viewportController->tick();
    }
}

bool NauViewportWidget::nativeEvent(const QByteArray& eventType, void* message, qintptr* result)
{
    if (m_viewportController) {
        m_viewportController->processNativeEvent(message);
    }
    return QWidget::nativeEvent(eventType, message, result);
}

bool NauViewportWidget::event(QEvent* inputEvent)
{
    bool isHandled = false;
    if (m_viewportController) {
        isHandled = m_viewportController->processEvent(inputEvent);
    }

    return isHandled || QWidget::event(inputEvent);
}

void NauViewportWidget::changeViewportController(NauBaseViewportControllerPtr controller)
{
    m_viewportController = controller;
}

NauBaseViewportControllerPtr NauViewportWidget::controller() const
{ 
    return m_viewportController ? m_viewportController : nullptr;  
}

QSize NauViewportWidget::sizeHint() const
{
    return QSize{ 640, 480 };
}
