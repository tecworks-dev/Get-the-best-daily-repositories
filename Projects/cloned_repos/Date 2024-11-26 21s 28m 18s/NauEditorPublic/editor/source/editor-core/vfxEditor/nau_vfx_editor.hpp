// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// VFX editor class that stores the basic VFX editors of the VFX system
// (title, properties editing, effects editing)

#pragma once

#include "nau/rtti/rtti_impl.h"
#include <memory>

#include "nau_vfx.hpp"

#include "nau_editor_interfaces.hpp"

#include "nau_dock_manager.hpp"
#include "nau_dock_widget.hpp"

#include "nau_spoiler.hpp"

#include "nau_buttons.hpp"
#include "nau_static_text_label.hpp"


// ** NauVFXEditorPageHeader

class NauVFXEditorPageHeader : public NauWidget
{
    Q_OBJECT

public:
    NauVFXEditorPageHeader(const QString& title, const QString& subtitle, NauWidget* parent);

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
    NauVFXEditorPageBaseParams(NauWidget* parent);

    void setPosition(const QVector3D& position);
    void setRotation(const QVector3D& rotation);
    void setAutoStart(bool isAutoStart);
    void setLoop(bool isLoop);

signals:
    void eventPositionChanged(const QVector3D& position);
    void eventRotationChanged(const QVector3D& rotation);

    void eventAutoStart(bool isAutoStart);
    void eventLoop(bool isLoop);

private:
    void setupSpinBoxSizes(NauMultiValueDoubleSpinBox* spinBox);
    void resetSpinBoxData(NauMultiValueDoubleSpinBox* spinBox);
    void setSpinBoxData(const QVector3D &data, NauMultiValueDoubleSpinBox* spinBox);

    QVector3D spinBoxData(const NauMultiValueDoubleSpinBox* spinBox) const;

private:
    NauLayoutVertical* m_layout;
    NauLayoutGrid* m_contentLayout;

    NauMultiValueDoubleSpinBox* m_position;
    NauToolButton* m_revertPositionButton;

    NauMultiValueDoubleSpinBox* m_rotation;
    NauToolButton* m_revertRotationButton;

    NauToogleButton* m_isAutoStartToogleButton;
    NauToogleButton* m_isLoopToogleButton;

private:
    inline static constexpr int NumberOfSpinBoxComponent = 3;
    inline static constexpr int SpinBoxHeight = 32;

    inline static constexpr int WidgetHeight = 270;
    inline static constexpr int WidgetWidth = 424;
    inline static constexpr QMargins Margins = QMargins(16, 16, 16, 16);
    inline static constexpr int TransformDecimalPrecision = 2;

    inline static constexpr QSize ToolButtonSize = QSize(16, 16);

    inline static constexpr int LabelHeight = 16;

    inline static constexpr int SpacingBetweenLabelAndTransformWidget = 6;
    inline static constexpr int SpacingBetweenTransformWidgetAndLabel = 14;

    inline static constexpr int SpacingBetweenTransformWidgetAndSeparator = 20;
    inline static constexpr int SpacingBetweenTransformSeparatorAndToogleBlock = 22;

    inline static constexpr int ToogleButtonHeight = 32;

    inline static constexpr QSize NauToogleButtonSize = QSize(40, 20);
};


// ** NauVFXEditorPage

class NauVFXEditorPage : public NauWidget
{
    Q_OBJECT

public:
    explicit NauVFXEditorPage(NauWidget* parent);

    void setVFX(NauVFXPtr vfx);

signals:
    void eventPositionChanged(const QVector3D& position);
    void eventRotationChanged(const QVector3D& rotation);

    void eventAutoStart(bool isAutoStart);
    void eventLoop(bool isLoop);

private:
    NauVFXPtr m_vfxPtr;
    NauLayoutVertical* m_layout;

    NauVFXEditorPageHeader* m_editorHeader;
    NauVFXEditorPageBaseParams* m_editorBaseParams;

    // TODO Add NauVFXEditorPageEffects;
};


// ** NauVFXEditor

class NauVFXEditor final : public NauAssetEditorInterface
{
    NAU_CLASS_(NauVFXEditor, NauAssetEditorInterface)

public:
    explicit NauVFXEditor(NauDockManager* dockManager) noexcept;

    // TODO: Implement
    void initialize(NauEditorWindowAbstract* editorWindow) override;
    void terminate() override;
    void postInitialize() override;
    void preTerminate() override;

    void createAsset(const std::string& assetPath) override { /* TODO */ }
    bool openAsset(const std::string& assetPath) override;
    bool saveAsset(const std::string& assetPath) override;

    std::string editorName() const override;
    NauEditorFileType assetType() const override;

private:
    void handleRemovedVFX(const std::string& vfxFilePath);
    void reset();

    NauVFXEditor(const NauVFXEditor&) = default;
    NauVFXEditor(NauVFXEditor&&) = default;
    NauVFXEditor& operator=(const NauVFXEditor&) = default;
    NauVFXEditor& operator=(NauVFXEditor&&) = default;

private:
    NauDockManager* m_dockManager;
    NauVFXEditorPage* m_editorWidget;
    NauDockWidget* m_editorDockWidget;

    NauVFXPtr m_vfx;
};
