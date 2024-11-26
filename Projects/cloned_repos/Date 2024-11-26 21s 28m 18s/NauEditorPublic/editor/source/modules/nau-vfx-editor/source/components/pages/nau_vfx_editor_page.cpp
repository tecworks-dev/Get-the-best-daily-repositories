// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "themes/nau_theme.hpp"
#include "nau_log.hpp"
#include "nau_vfx_editor_page.hpp"


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

    // Offset
    setupLabel(tr("Offset"), 0, 0);
    m_revertOffsetButton = setupToolButton();
    connect(m_revertOffsetButton, &NauToolButton::clicked, [this]()
        {
            resetSpinBoxData(m_offset, NumberOfSpinBoxTransformComponent);
        });

    m_contentLayout->addWidget(m_revertOffsetButton, 0, 1, Qt::AlignRight);
    m_contentLayout->addItem(new QSpacerItem(0, SpacingBetweenLabelAndTransformWidget), 1, 0);

    m_offset = new NauMultiValueDoubleSpinBox(nullptr, NumberOfSpinBoxTransformComponent);
    setupSpinBox(m_offset, NumberOfSpinBoxTransformComponent);
    connect(m_offset, &NauMultiValueDoubleSpinBox::eventValueChanged, [this]()
        {
            emit eventOffsetChanged(spinBoxData3D(m_offset));
        });

    m_contentLayout->addWidget(m_offset, 2, 0);
    m_contentLayout->addItem(new QSpacerItem(0, SpacingBetweenTransformWidgetAndLabel), 3, 0);

    // Direction
    setupLabel(tr("Direction"), 3, 0);
    m_revertDirectionButton = setupToolButton();
    connect(m_revertDirectionButton, &NauToolButton::clicked, [this]()
        {
            resetSpinBoxData(m_direction, NumberOfSpinBoxTransformComponent);
        });

    m_contentLayout->addWidget(m_revertDirectionButton, 3, 1, Qt::AlignRight);
    m_contentLayout->addItem(new QSpacerItem(0, SpacingBetweenLabelAndTransformWidget), 1, 0);

    m_direction = new NauMultiValueDoubleSpinBox(nullptr, NumberOfSpinBoxTransformComponent);
    setupSpinBox(m_direction, NumberOfSpinBoxTransformComponent);
    connect(m_direction, &NauMultiValueDoubleSpinBox::eventValueChanged, [this]()
        {
            emit eventDirectionChanged(spinBoxData3D(m_direction));
        });

    m_contentLayout->addWidget(m_direction, 4, 0);
    m_contentLayout->addItem(new QSpacerItem(0, SpacingBetweenTransformWidgetAndLabel), 3, 0);

    // Rotation Speed
    setupLabel(tr("Rotation Speed"), 5, 0);
    m_revertRotationSpeedButton = setupToolButton();
    connect(m_revertRotationSpeedButton, &NauToolButton::clicked, [this]
        {
            resetSpinBoxData(m_rotationSpeed, NumberOfSpinBoxSpeedComponents);
        });

    m_contentLayout->addWidget(m_revertRotationSpeedButton, 5, 1, Qt::AlignRight);
    m_contentLayout->addItem(new QSpacerItem(0, SpacingBetweenLabelAndTransformWidget), 5, 0);

    m_rotationSpeed = new NauMultiValueDoubleSpinBox(nullptr, NumberOfSpinBoxSpeedComponents);
    setupSpinBox(m_rotationSpeed, NumberOfSpinBoxSpeedComponents);
    connect(m_rotationSpeed, &NauMultiValueDoubleSpinBox::eventValueChanged, [this]()
        {
            emit eventRotationSpeedChanged(spinBoxData2D(m_rotationSpeed));
        });

    m_contentLayout->addWidget(m_rotationSpeed, 6, 0);

    // Pull Speed
    setupLabel(tr("Pull Speed"), 7, 0);
    m_revertPullSpeedButton = setupToolButton();
    connect(m_revertPullSpeedButton, &NauToolButton::clicked, [this]()
        {
            resetSpinBoxData(m_pullSpeed, NumberOfSpinBoxSpeedComponents);
        });

    m_contentLayout->addWidget(m_revertPullSpeedButton, 7, 1, Qt::AlignRight);
    m_contentLayout->addItem(new QSpacerItem(0, SpacingBetweenLabelAndTransformWidget), 5, 0);

    m_pullSpeed = new NauMultiValueDoubleSpinBox(nullptr, NumberOfSpinBoxSpeedComponents);
    setupSpinBox(m_pullSpeed, NumberOfSpinBoxSpeedComponents);
    connect(m_pullSpeed, &NauMultiValueDoubleSpinBox::eventValueChanged, [this]()
        {
            emit eventPullSpeedChanged(spinBoxData2D(m_pullSpeed));
        });

    m_contentLayout->addWidget(m_pullSpeed, 8, 0);
}

void NauVFXEditorPageBaseParams::setOffset(const QVector3D& offset)
{
    setSpinBoxData(offset, m_offset);
}

void NauVFXEditorPageBaseParams::setDirection(const QVector3D& direction)
{
    setSpinBoxData(direction, m_direction);
}

void NauVFXEditorPageBaseParams::setRotationSpeed(const QVector2D& rotationSpeed)
{
    setSpinBoxData(rotationSpeed, m_rotationSpeed);
}

void NauVFXEditorPageBaseParams::setPullSpeed(const QVector2D& pullSpeed)
{
    setSpinBoxData(pullSpeed, m_pullSpeed);
}

void NauVFXEditorPageBaseParams::setupLabel(const QString& name, int row, int column)
{
    auto label = new NauStaticTextLabel(name);
    label->setFixedHeight(LabelHeight);
    label->setFont(Nau::Theme::current().fontObjectInspector());
    m_contentLayout->addWidget(label, row, column);
}

NauToolButton* NauVFXEditorPageBaseParams::setupToolButton()
{
    auto toolButton = new NauToolButton();
    toolButton->setIcon(Nau::Theme::current().iconUndoAction());
    toolButton->setIconSize(ToolButtonSize);

    return toolButton;
}

void NauVFXEditorPageBaseParams::setupSpinBox(NauMultiValueDoubleSpinBox* spinBox, int numberOfSpinBoxComponents)
{
    if (spinBox == nullptr) {
        return;
    }

    for (int i = 0; i < numberOfSpinBoxComponents; ++i) {
        (*spinBox)[i]->setFixedHeight(SpinBoxHeight);
    }

    spinBox->setDecimals(TransformDecimalPrecision);
    spinBox->setMinimum(std::numeric_limits<float>::lowest());
    spinBox->setMaximum(std::numeric_limits<float>::max());
}

void NauVFXEditorPageBaseParams::resetSpinBoxData(NauMultiValueDoubleSpinBox* spinBox, int numberOfSpinBoxComponents)
{
    if (spinBox == nullptr) {
        return;
    }

    // TODO Reset value to parent data
    for (int i = 0; i < numberOfSpinBoxComponents; ++i) {
        (*spinBox)[i]->setValue(0.0);
    }
}

void NauVFXEditorPageBaseParams::setSpinBoxData(const QVector2D& data, NauMultiValueDoubleSpinBox* spinBox)
{
    if (spinBox == nullptr) {
        return;
    }

    (*spinBox)[0]->setValue(data.x());
    (*spinBox)[1]->setValue(data.y());
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

QVector2D NauVFXEditorPageBaseParams::spinBoxData2D(const NauMultiValueDoubleSpinBox* spinBox) const
{
    if (spinBox == nullptr) {
        return QVector2D();
    }

    const double x = (*spinBox)[0]->value();
    const double y = (*spinBox)[1]->value();

    return QVector2D(x, y);
}

QVector3D NauVFXEditorPageBaseParams::spinBoxData3D(const NauMultiValueDoubleSpinBox* spinBox) const
{
    if (spinBox == nullptr) {
        return QVector3D();
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

    connect(m_editorBaseParams, &NauVFXEditorPageBaseParams::eventOffsetChanged, [this](const QVector3D& offset) {
        emit eventOffsetChanged(offset);
    });

    connect(m_editorBaseParams, &NauVFXEditorPageBaseParams::eventDirectionChanged, [this](const QVector3D& direction) {
        emit eventDirectionChanged(direction);
    });

    connect(m_editorBaseParams, &NauVFXEditorPageBaseParams::eventRotationSpeedChanged, [this](const QVector2D& rotationSpeed) {
        emit eventRotationSpeedChanged(rotationSpeed);
    });

    connect(m_editorBaseParams, &NauVFXEditorPageBaseParams::eventPullSpeedChanged, [this](const QVector2D& pullSpeed) {
        emit eventPullSpeedChanged(pullSpeed);
    });
}

void NauVFXEditorPage::setName(const std::string name)
{
    m_editorHeader->setTitle(name);
}

void NauVFXEditorPage::setOffset(const QVector3D& offset)
{
    m_editorBaseParams->setOffset(offset);
}

void NauVFXEditorPage::setDirection(const QVector3D& direction)
{
    m_editorBaseParams->setDirection(direction);
}

void NauVFXEditorPage::setRotationSpeed(const QVector2D& rotationSpeed)
{
    m_editorBaseParams->setRotationSpeed(rotationSpeed);
}

void NauVFXEditorPage::setPullSpeed(const QVector2D& pullSpeed)
{
    m_editorBaseParams->setPullSpeed(pullSpeed);
}