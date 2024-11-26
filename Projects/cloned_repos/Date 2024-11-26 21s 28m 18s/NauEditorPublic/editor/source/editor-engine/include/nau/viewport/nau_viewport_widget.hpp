// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport window widget

#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include "nau/viewport/nau_base_viewport_controller.hpp"

#include <QTimer>
#include <QWidget>
#include <QMimeData>
#include <QSize>


class NauViewportWidget;


// ** NauViewportWidget
//
// Window for displaying engine frames
// Handle raw input and redirect to viewport controller

class NAU_EDITOR_ENGINE_API NauViewportWidget : public QWidget
{
    Q_OBJECT

public:
    NauViewportWidget(QWidget* parent);
    ~NauViewportWidget();

    virtual bool nativeEvent(const QByteArray& eventType, void* message, qintptr* result) override;
    virtual bool event(QEvent* inputEvent) override;

    void changeViewportController(NauBaseViewportControllerPtr controller);

    NauBaseViewportControllerPtr controller() const;
    auto windowHandle() const { return m_hWnd; }

    virtual QSize sizeHint() const override;

    void setViewportName(const std::string& name) { m_name = name; }
    std::string viewportName() const { return m_name; }

signals:
    void eventFlyModeToggle(bool on);
    void eventEscPressed();

    void eventDropStaticMeshCreateRequested(const QPoint& widgetPos, const QStringList& assetFileNames);
    void eventDropStaticMeshMoveRequested(const QPoint& widgetPos);
    void eventDropRevertRequested();
    void eventDropFinished();

public slots:
    void onFrame();

private:
    NauBaseViewportControllerPtr m_viewportController = nullptr;
    HWND m_hWnd;

    std::string m_name;
};