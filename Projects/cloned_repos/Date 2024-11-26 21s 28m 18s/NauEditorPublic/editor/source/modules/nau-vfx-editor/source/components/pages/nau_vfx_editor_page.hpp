// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// VFX editor class that stores the basic VFX editors of the VFX system
// (title, vortex properties editing)

#pragma once

#include "nau/rtti/rtti_impl.h"
#include <memory>

#include "baseWidgets/nau_static_text_label.hpp"
#include "baseWidgets/nau_widget.hpp"

#include "nau_dock_manager.hpp"
#include "nau_dock_widget.hpp"


// ** NauVFXEditorPageHeader

class NauVFXEditorPageHeader : public NauWidget
{
    Q_OBJECT

public:
    NauVFXEditorPageHeader(const QString& title, const QString& subtitle, NauWidget* parent = nullptr);

    void setTitle(const std::string& title);

private:
    NauStaticTextLabel* m_title;

private:
    inline static constexpr int TitleMoveYPosition = 8;
    inline static constexpr int SubtitleMoveYPosition = 34;

    inline static constexpr int WidgetHeight = 80;
    inline static constexpr QMargins Margins = QMargins(16, 16, 16, 16);
    inline static constexpr int IconSize = 48;
    inline static constexpr int TextWidth = 600;
    inline static constexpr int Spacing = 16;
};


// ** NauVFXEditorPageBaseParams

class NauVFXEditorPageBaseParams : public NauWidget
{
    Q_OBJECT

public:
    NauVFXEditorPageBaseParams(NauWidget* parent = nullptr);

    void setOffset(const QVector3D& offset);

    void setDirection(const QVector3D& position);

    void setRotationSpeed(const QVector2D& rotationSpeed);
    void setPullSpeed(const QVector2D& pullSpeed);

signals:
    void eventOffsetChanged(const QVector3D& direction);

    void eventDirectionChanged(const QVector3D& direction);

    void eventRotationSpeedChanged(const QVector2D& rotationSpeed);
    void eventPullSpeedChanged(const QVector2D& pullSpeed);

private:
    void setupLabel(const QString& name, int row, int column);
    NauToolButton* setupToolButton();

    void setupSpinBox(NauMultiValueDoubleSpinBox* spinBox, int numberOfSpinBoxComponents);
    void resetSpinBoxData(NauMultiValueDoubleSpinBox* spinBox, int numberOfSpinBoxComponents);

    void setSpinBoxData(const QVector2D& data, NauMultiValueDoubleSpinBox* spinBox);
    void setSpinBoxData(const QVector3D& data, NauMultiValueDoubleSpinBox* spinBox);

    QVector2D spinBoxData2D(const NauMultiValueDoubleSpinBox* spinBox) const;
    QVector3D spinBoxData3D(const NauMultiValueDoubleSpinBox* spinBox) const;

private:
    NauLayoutVertical* m_layout;
    NauLayoutGrid* m_contentLayout;

    NauMultiValueDoubleSpinBox* m_offset;
    NauToolButton* m_revertOffsetButton;

    NauMultiValueDoubleSpinBox* m_direction;
    NauToolButton* m_revertDirectionButton;

    NauMultiValueDoubleSpinBox* m_rotationSpeed;
    NauToolButton* m_revertRotationSpeedButton;

    NauMultiValueDoubleSpinBox* m_pullSpeed;
    NauToolButton* m_revertPullSpeedButton;

private:
    inline static constexpr int NumberOfSpinBoxTransformComponent = 3;
    inline static constexpr int NumberOfSpinBoxSpeedComponents = 2;
    inline static constexpr int SpinBoxHeight = 32;

    inline static constexpr int WidgetHeight = 250;
    inline static constexpr QMargins Margins = QMargins(16, 16, 16, 16);
    inline static constexpr int TransformDecimalPrecision = 2;

    inline static constexpr QSize ToolButtonSize = QSize(16, 16);

    inline static constexpr int LabelHeight = 16;

    inline static constexpr int SpacingBetweenLabelAndTransformWidget = 6;
    inline static constexpr int SpacingBetweenTransformWidgetAndLabel = 14;
};


// ** NauVFXEditorPage

class NauVFXEditorPage : public NauWidget
{
    Q_OBJECT

public:
    explicit NauVFXEditorPage(NauWidget* parent = nullptr);

    void setName(const std::string name);

    void setOffset(const QVector3D& offset);

    void setDirection(const QVector3D& direction);

    void setRotationSpeed(const QVector2D& rotationSpeed);
    void setPullSpeed(const QVector2D& pullSpeed);

signals:
    void eventOffsetChanged(const QVector3D& offset);

    void eventDirectionChanged(const QVector3D& direction);

    void eventRotationSpeedChanged(const QVector2D& rotationSpeed);
    void eventPullSpeedChanged(const QVector2D& pullSpeed);

private:
    NauLayoutVertical* m_layout;

    NauVFXEditorPageHeader* m_editorHeader;
    NauVFXEditorPageBaseParams* m_editorBaseParams;
};