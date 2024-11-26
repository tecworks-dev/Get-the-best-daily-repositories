// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/viewport/nau_base_viewport_controller.hpp"
#include "nau/viewport/nau_viewport_utils.hpp"

#include "nau/nau_constants.hpp"
#include "nau/ui.h"
#include "nau/viewport/nau_viewport_widget.hpp"
#include "nau/selection/nau_object_selection.hpp"
#include "nau/render/render_window.h"

#include "nau/editor-engine/nau_editor_engine_services.hpp"

#include "nau/diag/logging.h"
#include "nau/service/service_provider.h"
#include "nau/graphics/core_graphics.h"


// ** NauBaseViewportController

NauBaseViewportController::NauBaseViewportController(NauViewportWidget* viewport)
    : m_viewport(viewport)
    , m_cursorPosition(viewport->cursor().pos())
{

}

void NauBaseViewportController::tick()
{
}

NauEditingMode NauBaseViewportController::editingMode() 
{ 
    return NauEditingMode::Select;
}

bool NauBaseViewportController::isEditingModeActive(NauEditingMode mode)
{ 
    return editingMode() == mode;
}

void NauBaseViewportController::enableDrawGrid(bool value)
{
    auto& graphcis = nau::getServiceProvider().get<nau::ICoreGraphics>();
    graphcis.getDefaultRenderWindow().acquire()->setDrawViewportGrid(value);
}

void NauBaseViewportController::onResize(QResizeEvent* resizeEvent, const float dpi)
{
    const int newWidth = resizeEvent->size().width() * dpi;
    const int newHeight = resizeEvent->size().height() * dpi;

    updateEngineRenderer(newWidth, newHeight);
}

void NauBaseViewportController::updateEngineRenderer(int newWidth, int newHeight)
{
    
    Nau::EditorEngine().viewportManager()->resize(viewport()->viewportName(), newWidth, newHeight);
    //auto& graphcis = nau::getServiceProvider().get<nau::ICoreGraphics>();
    //graphcis.getDefaultRenderWindow().acquire()->requestViewportResize(newWidth, newHeight).detach();
    //TODO: detach is dangerous here. We should wait until resize is done somewhere before new resize.

    nau::getServiceProvider().get<nau::ui::UiManager>().setScreenSize(newWidth, newHeight);
}

void NauBaseViewportController::showCursor(bool visibility)
{
    m_cursorVisible = visibility;
    auto cursorShape = visibility ? Qt::CursorShape::ArrowCursor : Qt::CursorShape::BlankCursor;
    m_viewport->setCursor(cursorShape);
}

void NauBaseViewportController::moveCursor(const QPoint& position, bool forceMove)
{
    m_cursorPosition = position;   
    if (forceMove) QCursor::setPos(position);
}


// All the code inside this namespace is a copy of the classes from the browser.
// TODO: Rework drag drop in viewport

namespace Nau::ViewportDragDrop
{
    enum class NauEditorViewportFileType
    {
        Unrecognized = 0,
        EngineCore,
        Project,
        Config,
        Texture,
        Model,
        Shader,
        Script,
        VirtualRomFS,
        Scene,
        Material,
    };

    struct NauViewportDragContext {
        const std::vector<std::pair<NauEditorViewportFileType, QString>> assetDragList;
    };
    std::optional<NauViewportDragContext> fromMimeData(const QMimeData& mimeData)
    {
        const static QString MIME_TYPE = QStringLiteral("application/88def878-f93b-4bc4-8932-dee216e88359");

        if (!mimeData.hasFormat(MIME_TYPE)) {
            return std::nullopt;
        }

        QTextStream stream(mimeData.data(MIME_TYPE));
        int type{};
        QString fileName;
        std::vector<std::pair<NauEditorViewportFileType, QString>> result;

        while (!stream.atEnd())
        {
            stream >> type >> fileName;
            result.emplace_back(std::make_pair(static_cast<NauEditorViewportFileType>(type), fileName));
        }

        return NauViewportDragContext{ result };
    }
}


// ** NauClickMap

void NauClickMap::mousePressRegister(Qt::MouseButton button, const QPointF& position)
{
    m_clickMap[button] = position;
}

bool NauClickMap::isClick(Qt::MouseButton button, const QPointF& position)
{
    return (m_clickMap[button] - position).isNull();
}


// ** NauBaseEditorViewportController

NauBaseEditorViewportController::NauBaseEditorViewportController(NauViewportWidget* viewport,
                                                                 NauViewportContentToolsInterfacePtr contentTools,
                                                                 NauCameraControllerInterfacePtr cameraController,
                                                                 NauViewportDragDropToolsInterfacePtr dragDropTools)
    : NauBaseViewportController(viewport)
    , m_contentTools(contentTools)
    , m_cameraController(cameraController)
    , m_dragDropTools(dragDropTools)
{
    this->viewport()->setMouseTracking(true);
}


NauBaseEditorViewportController::~NauBaseEditorViewportController()
{
}

void NauBaseEditorViewportController::tick()
{
    const float deltaTime = 0.1;

    m_cameraController->updateCameraMovement(deltaTime, m_input);
    NauBaseViewportController::tick();
    resetInputDeltas();
}

bool NauBaseEditorViewportController::processEvent(QEvent* event)
{
    switch (event->type()) {
    case QEvent::ShortcutOverride:
        if (isCameraActive()) {
            event->accept();
        }
        break;
    case QEvent::KeyPress:
        onKeyDown(static_cast<QKeyEvent*>(event));
        break;
    case QEvent::KeyRelease:
        onKeyRelease(static_cast<QKeyEvent*>(event));
        break;
    case QEvent::MouseButtonPress:
        onMouseButtonPress(static_cast<QMouseEvent*>(event));
        break;
    case QEvent::MouseButtonRelease:
        onMouseButtonRelease(static_cast<QMouseEvent*>(event));
        break;
    case QEvent::MouseMove:
        onMouseMove(static_cast<QMouseEvent*>(event));
        break;
    case QEvent::Wheel:
        onWheelScroll(static_cast<QWheelEvent*>(event));
        break;
    case QEvent::DragEnter:
        onDragEnterEvent(static_cast<QDragEnterEvent*>(event));
        break;
    case QEvent::DragLeave:
        onDragLeaveEvent(static_cast<QDragLeaveEvent*>(event));
        break;
    case QEvent::Drop:
        onDropEvent(static_cast<QDropEvent*>(event));
        break;
    case QEvent::DragMove:
        onDragMoveEvent(static_cast<QDragMoveEvent*>(event));
        break;
    case QEvent::Resize: {
        const float dpi = viewport()->devicePixelRatioF();
        NauBaseViewportController::onResize(static_cast<QResizeEvent*>(event), dpi);
        break;
    }
    default:
        // Event not handled in this controller
        return false;        
    }

    // Event handled
    return true;
}

void NauBaseEditorViewportController::setEditingMode(NauEditingMode mode)
{
    if (m_contentTools) {
        m_contentTools->setEditMode(mode);
    }
}

NauEditingMode NauBaseEditorViewportController::editingMode()
{
    return m_contentTools->editingMode();
}

void NauBaseEditorViewportController::focusOnObject(nau::scene::SceneObject::WeakRef object)
{
    m_cameraController->focusOn(object->getWorldTransform().getMatrix());
}

NauCameraControllerInterfacePtr NauBaseEditorViewportController::cameraController() const
{
    return m_cameraController;
}

// TODO: Set selection callback in controller creation
void NauBaseEditorViewportController::setSelectionCallback(const NauSelectionCallback& callback)
{
    if (m_selectionCallback) {
        NAU_LOG_WARNING("Selection callback for viewport controller already exists!");
    }

    m_selectionCallback = callback;
}

void NauBaseEditorViewportController::updateModifiers(Qt::KeyboardModifiers modifiers)
{
    const bool shiftPressed = modifiers & Qt::ShiftModifier;
    const bool ctrlPressed = modifiers & Qt::ControlModifier;
    const bool altPressed = modifiers & Qt::AltModifier;

    m_input.setKeyDown(Qt::Key_Shift, shiftPressed);
    m_input.setKeyDown(Qt::Key_Control, ctrlPressed);
    m_input.setKeyDown(Qt::Key_Alt, altPressed);
}

void NauBaseEditorViewportController::resetInputDeltas()
{
    m_input.setDeltaWheel(0.0f);
    m_input.setDeltaMouse({ 0.0f, 0.0f });
}

void NauBaseEditorViewportController::onMouseButtonPress(QMouseEvent* event)
{
    const QPoint cursorPosition = QCursor::pos();
    m_clickHelper.mousePressRegister(event->button(), cursorPosition.toPointF());

    moveCursor(cursorPosition, isCameraActive());
    updateModifiers(event->modifiers());
    m_input.setMouseButtonDown(event->button(),true);
    
    showCursor(!isCameraActive());

    if (isCameraActive()) {
        emit viewport()->eventFlyModeToggle(true);
        return;
    }

    if (event->button() == Qt::MouseButton::LeftButton && m_contentTools) {
        m_contentTools->handleMouseInput(event, viewport()->devicePixelRatioF());
    }
}

void NauBaseEditorViewportController::onMouseButtonRelease(QMouseEvent* event)
{   
    const QPoint cursorPosition = QCursor::pos();

    const bool isClick = m_clickHelper.isClick(event->button(), cursorPosition.toPointF());

    moveCursor(cursorPosition, isCameraActive());
    updateModifiers(event->modifiers());
    m_input.setMouseButtonDown(event->button(),false);
    
    showCursor(!isCameraActive());

    if (isCameraActive()) {
        return;
    }

    // TODO: Implement click handling (so that we can separate clicking and pressing the button)
    if (event->button() == Qt::MouseButton::LeftButton) {      
        if (isClick) {
            processClick(event);
        }
        
        if (m_contentTools) {
            m_contentTools->handleMouseInput(event, viewport()->devicePixelRatioF());
        }
    }

    emit viewport()->eventFlyModeToggle(isCameraActive());
}

void NauBaseEditorViewportController::processClick(QMouseEvent* clickEvent)
{      
    if (m_contentTools && m_contentTools->isUsing()) {
        return;
    }

    if (m_selectionCallback) {
        m_selectionCallback(clickEvent, viewport()->devicePixelRatioF());
    }
}

void NauBaseEditorViewportController::onMouseMove(QMouseEvent* event)
{
    const QPoint eventCursorPosition = QCursor::pos();
    const QPoint deltaMouse = cursorPosition() - eventCursorPosition;

    // Prevent m_deltaMouse zeroing and MouseMove events recursion after calling cursor setPos function
    // Do it only in resetDeltas()
    if (deltaMouse.isNull()) {
        return;
    }
    m_input.setDeltaMouse(deltaMouse);

    const QPoint newCursorPosition = isCursorVisible() ? eventCursorPosition : cursorPosition();
    moveCursor(newCursorPosition, isCameraActive());

    if (isCameraActive()) {
        return;
    }

    if (m_contentTools) {
        m_contentTools->handleMouseInput(event, viewport()->devicePixelRatioF());
    }
}

void NauBaseEditorViewportController::onWheelScroll(QWheelEvent* event)
{
    m_input.setDeltaWheel(event->angleDelta().y() / 120);

    // Change the camera settings immediately
    // Camera movment changes once per tick 
    if (isCameraActive()) {
        m_cameraController->changeCameraSpeed(m_input.deltaWheel());
        m_input.setDeltaWheel(0.0f);
    }
}

void NauBaseEditorViewportController::onKeyDown(QKeyEvent* event)
{
    // Handle modifier keys by key()
    if ((event->key() == Qt::Key_Shift) || (event->key() == Qt::Key_Alt)
        || (event->key() == Qt::Key_Control)) {
        m_input.setKeyDown(static_cast<Qt::Key>(event->key()), true);
        return;
    }

   m_input.setKeyDown(static_cast<Qt::Key>(event->nativeVirtualKey()), true);
}

void NauBaseEditorViewportController::onKeyRelease(QKeyEvent* event)
{
    // Handle modifier keys by key()
    if ((event->key() == Qt::Key_Shift) || (event->key() == Qt::Key_Alt)
        || event->key() == Qt::Key_Control) {
        m_input.setKeyDown(static_cast<Qt::Key>(event->key()), false);
        return;
    }

    m_input.setKeyDown(static_cast<Qt::Key>(event->nativeVirtualKey()), false);
}

void NauBaseEditorViewportController::onDragEnterEvent(QDragEnterEvent* event)
{
    if (m_dragDropTools)
        m_dragDropTools->onDragEnterEvent(event);
}

void NauBaseEditorViewportController::onDragLeaveEvent(QDragLeaveEvent* event)
{
    if (m_dragDropTools)
        m_dragDropTools->onDragLeaveEvent(event);
}

void NauBaseEditorViewportController::onDragMoveEvent(QDragMoveEvent* event)
{
    if (m_dragDropTools)
        m_dragDropTools->onDragMoveEvent(event);
}

void NauBaseEditorViewportController::onDropEvent(QDropEvent* event)
{
    if (m_dragDropTools)
        m_dragDropTools->onDropEvent(event);
}

bool NauBaseEditorViewportController::isCameraActive() const
{
    if (!m_cameraController) {
        return false;
    }

    return m_cameraController->isCameraActive(m_input);
}