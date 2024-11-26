// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_scene_editor_viewport_toolbar.hpp"
#include "themes/nau_theme.hpp"

#include <QFileDialog>


// ** NauSceneEditorViewportToolbar

NauSceneEditorViewportToolbar::NauSceneEditorViewportToolbar(NauViewportWidget& viewport, NauShortcutHub* shortcutHub, NauWidget* parent)
    : NauToolbarBase(parent)
    , m_linkedViewport(viewport)
    , m_cameraSettingsView(new NauSceneCameraSettingsWidget(this))
{
    // Transform modes
    auto modesSection = addSection(NauToolbarSection::Left);
    auto buttonHam = modesSection->addButton(Nau::Theme::current().iconHamburger(), "", this, &NauSceneEditorViewportToolbar::handleMenu);
    modesSection->addSeparator();
    
    const auto addTransformModeButton = [this, modesSection, shortcutHub](const QString& name, NauIcon icon, NauEditingMode mode, NauShortcutOperation operation) {
        auto button = modesSection->addButton(icon, tr("%1 objects (%2)").arg(name).arg(shortcutHub->getAssociatedKeySequence(operation).toString()), [this, mode] {
            m_linkedViewport.controller()->setEditingMode(mode);
        });
        button->setObjectName(name + "ActionButton");
        button->setCheckable(true);
        shortcutHub->addApplicationShortcut(operation, std::bind(&NauToolButton::click, button));
        connect(&m_linkedViewport, &NauViewportWidget::eventFlyModeToggle, button, &NauToolButton::setDisabled);
        return button;
    };
    
    m_selectActionButton = addTransformModeButton(tr("Select"), Nau::Theme::current().iconSelectTool(), NauEditingMode::Select, NauShortcutOperation::ViewportSelectTool);
    m_translateActionButton = addTransformModeButton(tr("Select and move"), Nau::Theme::current().iconMoveTool(), NauEditingMode::Translate, NauShortcutOperation::ViewportTranslateTool);
    m_rotateActionButton = addTransformModeButton(tr("Select and rotate"), Nau::Theme::current().iconRotateTool(), NauEditingMode::Rotate, NauShortcutOperation::ViewportRotateTool);
    m_scaleActionButton = addTransformModeButton(tr("Select and scale"), Nau::Theme::current().iconScaleTool(), NauEditingMode::Scale, NauShortcutOperation::ViewportScaleTool);

    // Transform tool settings
    auto rightSection = addSection(NauToolbarSection::Right);
    m_coordinateSpaceMenu = new NauMenu(this);
    const auto registerCoordinateSpaceAction = [this](const QIcon& icon, const QString& title, bool isLocal, bool checked = false)
    {
        auto action = new NauAction(icon, title);
        action->setCheckable(true);
        action->setChecked(checked);
        m_coordinateSpaceMenu->addAction(action);
        connect(action, &QAction::toggled, [this, isLocal, icon]() {
            m_coordinateSpaceButton->setIcon(icon);
            emit eventCoordinateSpaceChanged(isLocal);
        });
    };
    
    registerCoordinateSpaceAction(Nau::Theme::current().iconLocalCoordinateSpace(), "Local", true);
    registerCoordinateSpaceAction(Nau::Theme::current().iconWorldCoordinateSpace(), "World", false);

    m_coordinateSpaceButton = new NauBorderlessButton(Nau::Theme::current().iconLocalCoordinateSpace(), m_coordinateSpaceMenu, rightSection);
    m_coordinateSpaceButton->setObjectName("coordinateSpaceButton");
    m_coordinateSpaceButton->setToolTip(tr("Set transform tools space coordinates"));
    m_coordinateSpaceButton->setFixedHeight(16);
    rightSection->addExternalWidget(m_coordinateSpaceButton);
    //rightSection->addSeparator();

    // Camera settings
    //m_cameraButton = rightSection->addButton(Nau::Theme::current().iconCameraSettings(), tr("Camera settings"), this, &NauSceneEditorViewportToolbar::handleCameraSettings);

    //m_cameraSettingsView->setControlButton(m_cameraButton);
    //connect(m_cameraSettingsView, &NauSceneCameraSettingsWidget::close, [this] {
    //    m_cameraButton->setChecked(false);
    //});

    //// Default settings
    //buttonHam->setEnabled(false);  // TODO: menun
    //m_cameraButton->setCheckable(true);

    //// TODO: Make global update timer for dynamic UI redraw
    auto updateTimer = new QTimer(this);
    connect(updateTimer, &QTimer::timeout, this, &NauSceneEditorViewportToolbar::updateFromViewport);
    updateTimer->start(1000 / 60);
}

void NauSceneEditorViewportToolbar::updateFromViewport()
{
    auto controller = m_linkedViewport.controller();
    if (controller) {
        m_selectActionButton->setChecked(controller->isEditingModeActive(NauEditingMode::Select));
        m_translateActionButton->setChecked(controller->isEditingModeActive(NauEditingMode::Translate));
        m_rotateActionButton->setChecked(controller->isEditingModeActive(NauEditingMode::Rotate));
        m_scaleActionButton->setChecked(controller->isEditingModeActive(NauEditingMode::Scale));
    }
}

void NauSceneEditorViewportToolbar::handleMenu()
{
    Q_UNIMPLEMENTED();
}

void NauSceneEditorViewportToolbar::handleCameraSettings()
{
    const bool showFlag = m_cameraButton->isChecked();
    if (showFlag) {
        const QPoint buttonAbsoluteCoords = m_cameraButton->mapToGlobal(QPoint(0, 0));
        const int newXCoord = buttonAbsoluteCoords.x() - m_cameraSettingsView->size().width() + m_cameraButton->size().width();
        const int newYCoord = buttonAbsoluteCoords.y() + m_cameraButton->size().height() * 1.5;

        m_cameraSettingsView->move(newXCoord, newYCoord);
    }
    m_cameraSettingsView->setHidden(!showFlag);
}

void NauSceneEditorViewportToolbar::handlePlaymodeOn()
{
    //m_cameraButton->setEnabled(false);
    m_selectActionButton->setEnabled(false);
    m_translateActionButton->setEnabled(false);
    m_rotateActionButton->setEnabled(false);
    m_scaleActionButton->setEnabled(false);
}

void NauSceneEditorViewportToolbar::handlePlaymodeOff()
{
    //m_cameraButton->setEnabled(true);
    m_selectActionButton->setEnabled(true);
    m_translateActionButton->setEnabled(true);
    m_rotateActionButton->setEnabled(true);
    m_scaleActionButton->setEnabled(true);
}
