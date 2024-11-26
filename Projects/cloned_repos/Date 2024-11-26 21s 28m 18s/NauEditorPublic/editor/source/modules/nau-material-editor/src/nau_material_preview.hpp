// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Material preiview classes

#pragma once

#include "baseWidgets/nau_widget.hpp"

#include <QQuaternion>

#include "inspector/nau_object_inspector.hpp"

// ** NauOrbitTransformController

class NauOrbitTransformController
{
public:
    NauOrbitTransformController() noexcept;
    ~NauOrbitTransformController() noexcept = default;

    void setRotationSpeed(float speed) noexcept;
    void setDistanceMin(float distance) noexcept;
    void setDistanceMax(float distance) noexcept;

    void onMousePressed(bool pressed, const QPointF& mousePosition) noexcept;
    void onMouseMove(const QSize& widgetSize, const QPointF& mousePosition) noexcept;
    void onMouseWheel(const QSize& widgetSize, const QPoint& offset) noexcept;

    [[nodiscard]] const QQuaternion& rotation() const noexcept;
    [[nodiscard]] float distance() const noexcept;

private:
    QQuaternion m_rotationCurrent;
    QQuaternion m_rotationPrevious;
    QPointF m_startMove;

    float m_rotationSpeed = 60.f;
    float m_distanceMin = 1.f;
    float m_distanceMax = 10.f;
    float m_distanceCurrent = 2.f;
    bool m_mousePressed = false;
};

// ** NauMaterialPreviewRender

class QOpenGLBuffer;
class QOpenGLTexture;
class QOpenGLShaderProgram;
class QOpenGLVertexArrayObject;

class NauMaterialPreviewRender final : public Nau3DWidget
{
    Q_OBJECT

public:
    explicit NauMaterialPreviewRender(QWidget* parent);
    ~NauMaterialPreviewRender() noexcept override;

    [[nodiscard]] bool hasHeightForWidth() const override;
    [[nodiscard]] int heightForWidth(int width) const override;

    void setAlbedo(const QString& textureName);

protected:
    void onRender() override;
    void resizeEvent(QResizeEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

    void resetRender();
    void createShader();
    void createTextures();
    void createCube();

private:
    NauOrbitTransformController m_controller;

    QOpenGLShaderProgram* m_shader = nullptr;
    QOpenGLTexture* m_texture = nullptr;

    QOpenGLVertexArrayObject* m_vao = nullptr;
    QOpenGLBuffer* m_vertexBuffer = nullptr;
    QOpenGLBuffer* m_indexBuffer = nullptr;

    QVector3D m_lightPosition;

    int32_t m_positionLocation = 0;
    int32_t m_normalLocation = 0;
    int32_t m_uvLocation = 0;
    int32_t m_colorLocation = 0;
    int32_t m_albedoLocation = 0;
    int32_t m_mvpLocation = 0;
    int32_t m_lightPositionLocation = 0;

    int32_t m_indexCount = 0;
};

// ** NauMaterialPreview

class NauMaterialPreview final : public NauWidget
{
    Q_OBJECT
public:
    explicit NauMaterialPreview(NauWidget* parent = nullptr);

    void setTitle(const QString& title);
    void setMaterial(const NauPropertiesContainer& materialParams);

private:
    void createPreviewSpoiler();
    void createPreviewContent();

private:
    NauInspectorSubWindow* m_spoiler = nullptr;

    NauMaterialPreviewRender* m_previewContent = nullptr;
};
