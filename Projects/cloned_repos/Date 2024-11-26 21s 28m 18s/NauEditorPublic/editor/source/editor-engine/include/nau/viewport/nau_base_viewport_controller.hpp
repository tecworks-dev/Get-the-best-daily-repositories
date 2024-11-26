// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport controller public base class

#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include "nau/nau_editor_modes.hpp"
#include "nau/viewport/nau_viewport_content_editor_tools.hpp"
#include "nau/viewport/nau_camera_controller.hpp"
#include "nau/viewport/nau_viewport_drag_drop_tools.hpp"
#include "nau/viewport/nau_viewport_input.hpp"
#include "nau/scene/scene_object.h"

#include <QWidget>
#include <QEvent>
#include <QResizeEvent>
#include <QGenericMatrix>


class NauViewportWidget;

// ** NauBaseViewportController
//
// Viewport controller base class
// Contains common events handling implementations and input events functions for overriding

class NAU_EDITOR_ENGINE_API NauBaseViewportController
{
public:
    NauBaseViewportController(NauViewportWidget* viewport);
    virtual ~NauBaseViewportController() = default;

    virtual void tick();

    virtual bool processEvent(QEvent* event) { return false; }
    virtual void processNativeEvent(void* message) { };

    virtual NauEditingMode editingMode();
    bool isEditingModeActive(NauEditingMode mode);

    void enableDrawGrid(bool value);

    virtual void setEditingMode(NauEditingMode mode) { }
    virtual void focusOnObject(nau::scene::SceneObject::WeakRef object) {}

    void onResize(QResizeEvent* event, const float dpi);
    void updateEngineRenderer(int newHeight, int newWidth);

protected:
    // TODO: Move to a separate mouse helper class
    void showCursor(bool visibility);
    // forceMove flag is needed to move the cursor along a changed cursor position(cursor centering, keeping it in the viewport borders)
    // If the flag is not set, then Qt automatically moves the cursor to the position from the event.
    // TODO: Figure out why the cursor shakes a little if we always move it ourselves.
    void moveCursor(const QPoint& position, bool forceMove);
    bool isCursorVisible() const { return m_cursorVisible; }
    const QPoint cursorPosition() const { return m_cursorPosition; }

    NauViewportWidget* viewport() const { return m_viewport; }

private:
    NauViewportWidget* m_viewport = nullptr;
    // TODO: Move to a separate mouse helper class
    QPoint m_cursorPosition;
    bool m_cursorVisible = true;

    bool m_isRenderInited = false;
    bool m_needInitRenderer = false;
};

using NauBaseViewportControllerPtr = std::shared_ptr<NauBaseViewportController>;


// ** NauClickMap
//
// Class helper to detect mouse button click

class NauClickMap
{
public:
    void mousePressRegister(Qt::MouseButton button, const QPointF& position);
    bool isClick(Qt::MouseButton button, const QPointF& position);

private:
    std::unordered_map<Qt::MouseButton, QPointF> m_clickMap;
};


// ** NauBaseEditorViewportController

using NauSelectionCallback = std::function<void(QMouseEvent*, float)>;

class NAU_EDITOR_ENGINE_API NauBaseEditorViewportController : public NauBaseViewportController
{
public:
    NauBaseEditorViewportController(NauViewportWidget* viewport,
                                    NauViewportContentToolsInterfacePtr contentTools,
                                    NauCameraControllerInterfacePtr cameraController,
                                    NauViewportDragDropToolsInterfacePtr dragDropTools);

    ~NauBaseEditorViewportController();

    void tick() override;

    // Editor viewport controller skips native event processing
    void processNativeEvent(void* message) override {};
    bool processEvent(QEvent* event) override;

    void setEditingMode(NauEditingMode mode) override;
    NauEditingMode editingMode() override;

    void focusOnObject(nau::scene::SceneObject::WeakRef object) override;
    
    NauCameraControllerInterfacePtr cameraController() const;

    void setSelectionCallback(const NauSelectionCallback& callback);

private:
    virtual void onMouseButtonPress(QMouseEvent* event);
    virtual void onMouseButtonRelease(QMouseEvent* event);
    virtual void onMouseMove(QMouseEvent* event);
    virtual void onWheelScroll(QWheelEvent* wheelEvent);
    virtual void onKeyDown(QKeyEvent* event);
    virtual void onKeyRelease(QKeyEvent* keyEvent);

    virtual void onDragEnterEvent(QDragEnterEvent* event);
    virtual void onDragLeaveEvent(QDragLeaveEvent* event);
    virtual void onDragMoveEvent(QDragMoveEvent* event);
    virtual void onDropEvent(QDropEvent* event);

    virtual void processClick(QMouseEvent* clickEvent);

private:
    void updateModifiers(Qt::KeyboardModifiers modifiers);
    void resetInputDeltas();

    bool isCameraActive() const;

private:
    NauViewportInput m_input;
    NauClickMap m_clickHelper;

    NauViewportContentToolsInterfacePtr m_contentTools;
    NauCameraControllerInterfacePtr m_cameraController;
    NauViewportDragDropToolsInterfacePtr m_dragDropTools;
    NauSelectionCallback m_selectionCallback;
};

using NauBaseEditorViewportControllerPtr = std::shared_ptr<NauBaseEditorViewportController>;