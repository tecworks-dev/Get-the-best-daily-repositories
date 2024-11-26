// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_vfx_editor.hpp"
#include "themes/nau_theme.hpp"
#include "nau_log.hpp"
#include "nau_vfx_file.hpp"
#include "nau_editor.hpp"


// ** NauVFXEditorPageHeader

NauVFXEditorPageHeader::NauVFXEditorPageHeader(const QString& title, const QString& subtitle, NauWidget* parent)
{
    const NauAbstractTheme& theme = Nau::Theme::current();

    setFixedHeight(WidgetHeight);

    auto* layout = new NauLayoutVertical(this);

    auto* layoutMain = new NauLayoutHorizontal();
    layoutMain->setContentsMargins(Margins);
    layoutMain->setSpacing(Spacing);
    layout->addLayout(layoutMain);

    // Image
    auto* label = new QLabel(this);
    label->setPixmap(Nau::Theme::current().iconVFXEditor().pixmap(IconSize));
    layoutMain->addWidget(label);

    // Text
    auto* container = new NauWidget(this);
    container->setMinimumHeight(IconSize);
    container->setMaximumHeight(IconSize);
    container->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    layoutMain->addWidget(container);

    // Title
    m_title = new NauStaticTextLabel(title, container);
    m_title->setFont(theme.fontInputHeaderTitle());
    m_title->setColor(theme.paletteVFXGeneric().color(NauPalette::Role::TextHeader));
    m_title->move(0, TitleMoveYPosition);
    m_title->setFixedWidth(TextWidth);

    // Subtitle
    auto* labelSubtitle = new NauStaticTextLabel(subtitle, container);
    labelSubtitle->setFont(theme.fontInputHeaderSubtitle());
    labelSubtitle->setColor(theme.paletteVFXGeneric().color(NauPalette::Role::Text));
    labelSubtitle->move(0, SubtitleMoveYPosition);
    labelSubtitle->setFixedWidth(TextWidth);

    // Bottom separator
    auto* separator = new QFrame;
    separator->setStyleSheet("background-color: #141414;");
    separator->setFrameShape(QFrame::HLine);
    separator->setFixedHeight(1);
    layout->addWidget(separator);
}

void NauVFXEditorPageHeader::setTitle(const std::string& title)
{
    m_title->setText(title.c_str());
}


// ** NauVFXEditorPageBaseParams

NauVFXEditorPageBaseParams::NauVFXEditorPageBaseParams(NauWidget* parent)
    : NauWidget(parent)
{
    const NauAbstractTheme& theme = Nau::Theme::current();

    setFixedHeight(WidgetHeight);

    auto* layout = new NauLayoutVertical(this);
    layout->setContentsMargins(Margins);

    auto container = new NauWidget();
    layout->addWidget(container);

    auto* layoutMain = new NauLayoutHorizontal(container);

    m_contentLayout = new NauLayoutGrid();
    layoutMain->addLayout(m_contentLayout);

    // Position
    auto labelPosition = new NauStaticTextLabel(tr("Position (Local)"));
    labelPosition->setFixedHeight(LabelHeight);
    labelPosition->setFont(Nau::Theme::current().fontObjectInspector());
    m_contentLayout->addWidget(labelPosition, 0, 0);

    m_revertPositionButton = new NauToolButton();
    m_revertPositionButton->setIcon(Nau::Theme::current().iconUndoAction());
    m_revertPositionButton->setIconSize(ToolButtonSize);
    connect(m_revertPositionButton, &NauToolButton::clicked, [this] {
        resetSpinBoxData(m_position);
    });

    m_contentLayout->addWidget(m_revertPositionButton, 0, 1, Qt::AlignRight);
    m_contentLayout->addItem(new QSpacerItem(0, SpacingBetweenLabelAndTransformWidget), 1, 0);

    m_position = new NauMultiValueDoubleSpinBox(nullptr, NumberOfSpinBoxComponent);
    setupSpinBoxSizes(m_position);
    m_position->setDecimals(TransformDecimalPrecision);
    m_position->setMinimum(std::numeric_limits<float>::lowest());
    m_position->setMaximum(std::numeric_limits<float>::max());
    connect(m_position, &NauMultiValueDoubleSpinBox::eventValueChanged, [this]()
        {
            emit eventPositionChanged(spinBoxData(m_position));
        });

    m_contentLayout->addWidget(m_position, 2, 0);
    m_contentLayout->addItem(new QSpacerItem(0, SpacingBetweenTransformWidgetAndLabel), 3, 0);

    // Rotation
    auto labelRotation = new NauStaticTextLabel(tr("Rotation (Local)"));
    labelRotation->setFixedHeight(LabelHeight);
    labelRotation->setFont(Nau::Theme::current().fontObjectInspector());
    m_contentLayout->addWidget(labelRotation, 4, 0);

    m_revertRotationButton = new NauToolButton();
    m_revertRotationButton->setIcon(Nau::Theme::current().iconUndoAction());
    m_revertRotationButton->setIconSize(ToolButtonSize);
    connect(m_revertRotationButton, &NauToolButton::clicked, [this] {
        resetSpinBoxData(m_rotation);
    });

    m_contentLayout->addWidget(m_revertRotationButton, 4, 1, Qt::AlignRight);
    m_contentLayout->addItem(new QSpacerItem(0, SpacingBetweenLabelAndTransformWidget), 5, 0);

    m_rotation = new NauMultiValueDoubleSpinBox(nullptr, NumberOfSpinBoxComponent);
    setupSpinBoxSizes(m_rotation);
    m_rotation->setDecimals(TransformDecimalPrecision);
    m_rotation->setMinimum(std::numeric_limits<float>::lowest());
    m_rotation->setMaximum(std::numeric_limits<float>::max());
    connect(m_rotation, &NauMultiValueDoubleSpinBox::eventValueChanged, [this]()
        {
            emit eventRotationChanged(spinBoxData(m_rotation));
        });

    m_contentLayout->addWidget(m_rotation, 6, 0);

    layout->addSpacing(SpacingBetweenTransformWidgetAndSeparator);

    // Bottom separator
    auto* separator = new QFrame;
    separator->setStyleSheet("background-color: #141414;");
    separator->setFrameShape(QFrame::HLine);
    separator->setFixedHeight(1);
    layout->addWidget(separator);

    layout->addSpacing(SpacingBetweenTransformSeparatorAndToogleBlock);

    // AutoStart
    auto* autoStartLayoutHorizontal = new NauLayoutHorizontal();
    layout->addLayout(autoStartLayoutHorizontal);

    auto labelAutoStart = new NauStaticTextLabel(tr("Autostart"));
    labelAutoStart->setFixedHeight(ToogleButtonHeight);
    labelAutoStart->setFont(Nau::Theme::current().fontObjectInspector());
    autoStartLayoutHorizontal->addWidget(labelAutoStart, Qt::AlignLeft);

    m_isAutoStartToogleButton = new NauToogleButton();
    m_isAutoStartToogleButton->setFixedSize(NauToogleButtonSize);
    connect(m_isAutoStartToogleButton, &NauToogleButton::stateChanged, [this](int state)
        {
            emit eventAutoStart(state);
        });

    autoStartLayoutHorizontal->addWidget(m_isAutoStartToogleButton, Qt::AlignLeft);

    // Loop
    auto* loopLayoutHorizontal = new NauLayoutHorizontal();
    layout->addLayout(loopLayoutHorizontal);

    auto labelLoop = new NauStaticTextLabel(tr("Loop"));
    labelLoop->setFixedHeight(ToogleButtonHeight);
    labelLoop->setFont(Nau::Theme::current().fontObjectInspector());
    loopLayoutHorizontal->addWidget(labelLoop, Qt::AlignLeft);

    m_isLoopToogleButton = new NauToogleButton();
    m_isLoopToogleButton->setFixedSize(NauToogleButtonSize);
    connect(m_isLoopToogleButton, &NauToogleButton::stateChanged, [this](int state)
        {
            emit eventLoop(state);
        });

    loopLayoutHorizontal->addWidget(m_isLoopToogleButton, Qt::AlignLeft);
}

void NauVFXEditorPageBaseParams::setPosition(const QVector3D& position)
{
    setSpinBoxData(position, m_position);
}

void NauVFXEditorPageBaseParams::setRotation(const QVector3D& rotation)
{
    setSpinBoxData(rotation, m_rotation);
}

void NauVFXEditorPageBaseParams::setAutoStart(bool isAutoStart)
{
    if (!m_isAutoStartToogleButton) {
        return;
    }

    m_isAutoStartToogleButton->setChecked(isAutoStart);
}

void NauVFXEditorPageBaseParams::setLoop(bool isLoop)
{
    if (!m_isLoopToogleButton) {
        return;
    }

    m_isLoopToogleButton->setChecked(isLoop);
}

void NauVFXEditorPageBaseParams::setupSpinBoxSizes(NauMultiValueDoubleSpinBox* spinBox)
{
    if (spinBox == nullptr) {
        return;
    }

    for (int i = 0; i < NumberOfSpinBoxComponent; ++i) {
        (*spinBox)[i]->setFixedHeight(SpinBoxHeight);
    }
}

void NauVFXEditorPageBaseParams::resetSpinBoxData(NauMultiValueDoubleSpinBox* spinBox)
{
    if (spinBox == nullptr) {
        return;
    }

    // TODO Reset value to parent data
    (*spinBox)[0]->setValue(0.0);
    (*spinBox)[1]->setValue(0.0);
    (*spinBox)[2]->setValue(0.0);
}

void NauVFXEditorPageBaseParams::setSpinBoxData(const QVector3D& data, NauMultiValueDoubleSpinBox* spinBox)
{
    if (spinBox == nullptr) {
        return;
    }

    (*spinBox)[0]->setValue(data.x());
    (*spinBox)[1]->setValue(data.y());
    (*spinBox)[2]->setValue(data.z());
}

QVector3D NauVFXEditorPageBaseParams::spinBoxData(const NauMultiValueDoubleSpinBox* spinBox) const
{
    if (spinBox == nullptr) {
        return QVector3D(0.0, 0.0, 0.0);
    }

    const double x = (*spinBox)[0]->value();
    const double y = (*spinBox)[1]->value();
    const double z = (*spinBox)[2]->value();

    return QVector3D(x, y, z);
}


// ** NauVFXEditorPage

NauVFXEditorPage::NauVFXEditorPage(NauWidget* parent)
    : NauWidget(parent)
    , m_layout(new NauLayoutVertical(this))
    , m_editorHeader(new NauVFXEditorPageHeader("VFXFileName", tr("VFX Asset"), this))
    , m_editorBaseParams(new NauVFXEditorPageBaseParams(this))
{
    m_layout->addWidget(m_editorHeader, Qt::AlignTop);
    m_layout->addWidget(m_editorBaseParams, Qt::AlignTop);
    m_layout->addStretch(1);

    connect(m_editorBaseParams, &NauVFXEditorPageBaseParams::eventPositionChanged, [this](const QVector3D& position) {
        emit eventPositionChanged(position);
    });

    connect(m_editorBaseParams, &NauVFXEditorPageBaseParams::eventRotationChanged, [this](const QVector3D& rotation) {
        emit eventRotationChanged(rotation);
    });

    connect(m_editorBaseParams, &NauVFXEditorPageBaseParams::eventAutoStart, [this](bool isAutoStart) {
        emit eventAutoStart(isAutoStart);
    });

    connect(m_editorBaseParams, &NauVFXEditorPageBaseParams::eventLoop, [this](bool isLoop) {
        emit eventLoop(isLoop);
    });
}

void NauVFXEditorPage::setVFX(NauVFXPtr vfx)
{
    m_editorHeader->setTitle(vfx->name());

    m_editorBaseParams->setPosition(vfx->position());
    m_editorBaseParams->setRotation(vfx->rotation());
    m_editorBaseParams->setAutoStart(vfx->isAutoStart());
    m_editorBaseParams->setLoop(vfx->isLoop());

    m_vfxPtr = std::move(vfx);
}


// ** NauVFXEditor

NauVFXEditor::NauVFXEditor(NauDockManager* dockManager) noexcept
    : m_dockManager(dockManager)
    , m_editorWidget(nullptr)
    , m_editorDockWidget(new NauDockWidget(QObject::tr(editorName().data()), nullptr))
{
    m_editorDockWidget->setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromContent);

    //QObject::connect(m_vfxWatcher.get(), &NauAssetWatcher::eventAssetRemove, [this](const std::string& vfxFilePath) {
    //    handleRemovedVFX(vfxFilePath);
    //});

    //QObject::connect(m_vfxWatcher.get(), &NauAssetWatcher::eventAssetChanged, [this](const std::string& vfxFilePath) {
    //    if (m_vfx == nullptr) {
    //        return;
    //    }

    //    const QFileInfo fileInfo{ vfxFilePath.c_str() };
    //    const std::string fileName{ fileInfo.baseName().toUtf8().constData() };
    //    
    //    if (m_vfx->name() == fileName) {
    //        openAsset(vfxFilePath.c_str());
    //    }
    //});
}

void NauVFXEditor::initialize(NauEditorWindowAbstract* editorWindow)
{
    // Init editor systems
    // TODO: Move code from the constructor
}

void NauVFXEditor::terminate()
{
    // TODO: Reset resources
}

void NauVFXEditor::postInitialize()
{
    // TODO: ui initialize
}

void NauVFXEditor::preTerminate()
{
    // TODO: unbind ui from NauEditor window
}

bool NauVFXEditor::openAsset(const std::string& assetPath)
{
    if (const auto* inspector = m_dockManager->findDockWidget("Inspector")) {
        m_vfx = std::make_shared<NauVFX>();

        NauVFXFile vfxFile{ assetPath };
        if (!vfxFile.loadVFX(*m_vfx)) {
            return false;
        }

        reset();
        m_editorWidget->setVFX(m_vfx);

        //QObject::connect(m_vfx.get(), &NauVFX::eventVFXChanged, [assetPath, this]() {
        //    m_vfxWatcher->skipNextEvent(NauAssetWatcher::UpdateReason::Changed);
        //    saveAsset(assetPath);
        //});

        m_dockManager->addDockWidgetTabToArea(m_editorDockWidget, inspector->dockAreaWidget());
        NED_TRACE("Input action asset {} opened in the input editor.", assetPath);
        return true;
    }

    if (m_editorDockWidget->dockAreaWidget() == nullptr) {
        NED_ERROR("Failed to open input editor in tab.");
    }

    return false;
}

bool NauVFXEditor::saveAsset(const std::string& assetPath)
{
    if (m_vfx == nullptr) {
        NED_ERROR("Input action asset is not open.");
        return false;
    }

    NauVFXFile vfxFile{ assetPath };
    vfxFile.saveVFX(*m_vfx);

    NED_TRACE("Input action asset saved to {}.", assetPath);
    return true;
}

std::string NauVFXEditor::editorName() const
{
    return "VFX editor";
}

NauEditorFileType NauVFXEditor::assetType() const
{
    return NauEditorFileType::VFX;
}

void NauVFXEditor::handleRemovedVFX(const std::string& vfxFilePath)
{
    if (m_vfx == nullptr || m_editorWidget == nullptr) {
        return;
    }

    const std::string vfxName = QFileInfo(vfxFilePath.c_str()).baseName().toUtf8().constData();
    if (m_vfx->name() != vfxName) {
        return;
    }

    m_editorDockWidget->takeWidget();
    m_editorWidget->hide();
    m_editorWidget->deleteLater();
    m_editorWidget = nullptr;
}

void NauVFXEditor::reset()
{
    if (m_editorWidget != nullptr) {
        return;
    }

    m_editorWidget = new NauVFXEditorPage(nullptr);
    m_editorWidget->setParent(m_editorDockWidget);
    m_editorDockWidget->setWidget(m_editorWidget);

    // connect signals/slots
    QObject::connect(m_editorWidget, &NauVFXEditorPage::eventPositionChanged, [this](const QVector3D &position) {
        m_vfx->setPosition(position);
    });

    QObject::connect(m_editorWidget, &NauVFXEditorPage::eventRotationChanged, [this](const QVector3D &rotation) {
        m_vfx->setRotation(rotation);
    });

    QObject::connect(m_editorWidget, &NauVFXEditorPage::eventAutoStart, [this](bool isAutoStart) {
        m_vfx->setAutoStart(isAutoStart);
    });

    QObject::connect(m_editorWidget, &NauVFXEditorPage::eventLoop, [this](bool isLoop) {
        m_vfx->setLoop(isLoop);
    });
}
