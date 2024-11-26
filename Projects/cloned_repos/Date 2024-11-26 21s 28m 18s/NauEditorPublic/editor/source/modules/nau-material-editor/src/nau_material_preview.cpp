// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_material_preview.hpp"

#include "inspector/nau_object_inspector.hpp"
#include "nau_log.hpp"

#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>
#include <magic_enum/magic_enum.hpp>

#include "nau_assert.hpp"


constexpr float NAU_ORBIT_INITIAL_SLOPE = 15.f;

// ** NauOrbitTransformController

NauOrbitTransformController::NauOrbitTransformController() noexcept
    : m_rotationCurrent(QQuaternion::fromEulerAngles({NAU_ORBIT_INITIAL_SLOPE, 0.f, 0.f}))
{
}

void NauOrbitTransformController::setRotationSpeed(float speed) noexcept
{
    m_rotationSpeed = speed > 0.f ? speed : m_rotationSpeed;
}

void NauOrbitTransformController::setDistanceMin(float distance) noexcept
{
    m_distanceMin = std::min(distance, m_distanceMax);
}

void NauOrbitTransformController::setDistanceMax(float distance) noexcept
{
    m_distanceMax = std::max(distance, m_distanceMin);
}

void NauOrbitTransformController::onMousePressed(bool pressed, const QPointF& mousePosition) noexcept
{
    if (pressed != m_mousePressed) {
        m_mousePressed = pressed;
        m_startMove = mousePosition;
        m_rotationPrevious = m_rotationCurrent;
    }
}

void NauOrbitTransformController::onMouseMove(const QSize& widgetSize, const QPointF& mousePosition) noexcept
{
    if (!m_mousePressed) {
        return;
    }
    const QPointF positionDiff = mousePosition - m_startMove;
    const float size = static_cast<float>(std::min(widgetSize.width(), widgetSize.height()));
    const auto diffX = static_cast<float>(positionDiff.x());
    const auto diffY = static_cast<float>(positionDiff.y());
    const QVector3D force = QVector3D{ diffY, diffX, 0.f } * (m_rotationSpeed / size);
    m_rotationCurrent = QQuaternion::fromEulerAngles(force) * m_rotationPrevious;
}

void NauOrbitTransformController::onMouseWheel(const QSize& widgetSize, const QPoint& offset) noexcept
{
    const float size = static_cast<float>(std::min(widgetSize.width(), widgetSize.height()));
    const float diff = static_cast<float>(offset.x() + offset.y()) / size;
    m_distanceCurrent = std::clamp(m_distanceCurrent + diff, m_distanceMin, m_distanceMax);
}

const QQuaternion& NauOrbitTransformController::rotation() const noexcept
{
    return m_rotationCurrent;
}

float NauOrbitTransformController::distance() const noexcept
{
    return m_distanceCurrent;
}

// ** NauMaterialPreviewRender

NauMaterialPreviewRender::NauMaterialPreviewRender(QWidget* parent)
    : Nau3DWidget(parent)
    , m_lightPosition(0.85f, 0.9f, 1.f)
{
    if (!isValid()) {
        return;
    }
    makeCurrent();
    createShader();
    createTextures();
    createCube();
    doneCurrent();
}

NauMaterialPreviewRender::~NauMaterialPreviewRender() noexcept
{
    resetRender();
}

bool NauMaterialPreviewRender::hasHeightForWidth() const
{
    return true;
}

int NauMaterialPreviewRender::heightForWidth(int width) const
{
    // TODO: Support for different aspect ratios
    return width;
}

void NauMaterialPreviewRender::setAlbedo(const QString& textureName)
{
    if (!isValid()) {
        return;
    }
    makeCurrent();
    const QString textureFilePath = "";//NauEngineResourceAPI::getTextureFileByName(textureName);
    const QImage image(textureFilePath);
    delete m_texture;
    m_texture = new QOpenGLTexture(image);
    doneCurrent();

    render();
}

void NauMaterialPreviewRender::onRender()
{
    constexpr float COLOR_BYTE = 34.f / 255.f;
    const QSize widgetSize = size();
    auto* glFuncs = context()->functions();

    glFuncs->glViewport(0, 0, widgetSize.width(), widgetSize.height());
    glFuncs->glEnable(GL_DEPTH_TEST);
    glFuncs->glDepthFunc(GL_LEQUAL);
    glFuncs->glClearColor(COLOR_BYTE, COLOR_BYTE, COLOR_BYTE, 1.f);
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    constexpr float HALF_FOV = 60.f;
    constexpr float NEAR_PLANE = 0.1f;
    constexpr float FAR_PLANE = 1000.f;
    const float aspect = static_cast<float>(widgetSize.width()) / static_cast<float>(widgetSize.height());

    QMatrix4x4 viewProjection;
    viewProjection.perspective(HALF_FOV, aspect, NEAR_PLANE, FAR_PLANE);
    viewProjection.translate(0.f, 0.f, -m_controller.distance());
    viewProjection.rotate(m_controller.rotation());

    m_shader->bind();
    m_shader->setUniformValue(m_mvpLocation, viewProjection);
    m_shader->setUniformValue(m_lightPositionLocation, m_lightPosition);
    m_texture->bind();
    m_vao->bind();

    glFuncs->glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_SHORT, nullptr);
}

void NauMaterialPreviewRender::resizeEvent(QResizeEvent* event)
{
    Nau3DWidget::resizeEvent(event);
    render();
}

void NauMaterialPreviewRender::mouseMoveEvent(QMouseEvent* event)
{
    Nau3DWidget::mouseMoveEvent(event);
    m_controller.onMouseMove(size(), event->globalPosition());
    render();
}

void NauMaterialPreviewRender::mousePressEvent(QMouseEvent* event)
{
    Nau3DWidget::mousePressEvent(event);
    m_controller.onMousePressed(true, event->globalPosition());
    render();
}

void NauMaterialPreviewRender::mouseReleaseEvent(QMouseEvent* event)
{
    Nau3DWidget::mouseReleaseEvent(event);
    m_controller.onMousePressed(false, event->globalPosition());
}

void NauMaterialPreviewRender::wheelEvent(QWheelEvent* event)
{
    m_controller.onMouseWheel(size(), event->angleDelta());
    render();
}

void NauMaterialPreviewRender::resetRender()
{
    if (!isValid()) {
        return;
    }
    makeCurrent();
    delete m_texture;
    m_texture = nullptr;

    delete m_shader;
    m_shader = nullptr;

    delete m_vao;
    m_vao = nullptr;

    delete m_vertexBuffer;
    m_vertexBuffer = nullptr;

    delete m_indexBuffer;
    m_indexBuffer = nullptr;
    doneCurrent();
}

void NauMaterialPreviewRender::createShader()
{
    NED_ASSERT(m_shader && "Shader must equal nullptr!");
    // TODO: temporary shader implementation
    constexpr std::string_view VERTEX_SHADER_SRC =
        "#version 450 core\n"
        "layout (location=0) in vec3 a_position;\n"
        "layout (location=1) in vec3 a_normal;\n"
        "layout (location=2) in vec2 a_uv;\n"
        "layout (location=3) in vec4 a_color;\n"
        "uniform mat4 u_mvp;\n"
        "out vec3 fragPosition;\n"
        "out vec3 normal;\n"
        "out vec4 color;\n"
        "out vec2 uv;\n"
        "void main() {\n"
        "   // object location\n"
        "   fragPosition = a_position;\n"
        "   normal = a_normal;\n"
        "   color = a_color;\n"
        "   uv = a_uv;\n"
        "   gl_Position = u_mvp * vec4(a_position, 1.0);\n"
        "}\n";

    constexpr std::string_view FRAGMENT_SHADER_SRC =
        "#version 450 core\n"
        "uniform sampler2D u_albedo;\n"
        "uniform vec3 u_lightPos;\n"
        "in vec3 fragPosition;\n"
        "in vec3 normal;\n"
        "in vec4 color;\n"
        "in vec2 uv;\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "   // diffuse color\n"
        "   vec4 tex_color = texture( u_albedo, uv );\n"
        "   vec3 lightPath = u_lightPos - fragPosition;\n"
        "   const float distance = length(lightPath);\n"
        "   const float attenuation = 1.0 / (0.6 + 0.09f * distance + 0.032f * (distance * distance));\n"
        "   vec3 lightDir = lightPath / mix(1.0, distance, distance < 0.001);\n"
        "   float diff = max( dot( normal, lightDir ), 0.2 ) * attenuation;\n"
        "   // result color\n"
        "   FragColor = vec4( ( color * tex_color ).xyz * diff, 1.0 );\n"
        "}\n";

    m_shader = new QOpenGLShaderProgram;
    m_shader->addShaderFromSourceCode(QOpenGLShader::Vertex, VERTEX_SHADER_SRC.data());
    m_shader->addShaderFromSourceCode(QOpenGLShader::Fragment, FRAGMENT_SHADER_SRC.data());
    m_shader->link();
    m_shader->bind();

    m_positionLocation = m_shader->attributeLocation("a_position");
    m_normalLocation = m_shader->attributeLocation("a_normal");
    m_uvLocation = m_shader->attributeLocation("a_uv");
    m_colorLocation = m_shader->attributeLocation("a_color");
    m_albedoLocation = m_shader->uniformLocation("u_albedo");
    m_mvpLocation = m_shader->uniformLocation("u_mvp");
    m_lightPositionLocation = m_shader->uniformLocation("u_lightPos");
    m_shader->release();
}

void NauMaterialPreviewRender::createTextures()
{
    // TODO: init texture
    NED_ASSERT(m_texture && "Texture must equal nullptr!");
    m_texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
}

void NauMaterialPreviewRender::createCube()
{
    if (!isValid()) {
        return;
    }
    NED_ASSERT(m_vao && "Mesh must equal nullptr!");
    // TODO: temporary cube implementation, need get from resources
    struct Vertex
    {
        QVector3D position;
        QVector3D normal;
        QVector2D uv;
        QRgba64 color = QRgba64::fromRgba64(0xffff, 0xffff, 0xffff, 0xffff);
    };
    struct Plane
    {
        constexpr Plane( const Vertex& v0, const Vertex& v1, const Vertex& v2, const Vertex& v3 )
            : vertices{ v0, v1, v2, v3 }
        {
            //const QVector3D normal = QVector3D::crossProduct(v2.position - v1.position, v0.position - v1.position);
            const QVector3D normal = QVector3D::crossProduct(v0.position - v1.position, v2.position - v1.position);
            vertices[0].normal = normal;
            vertices[0].uv = QVector2D{ 0.f, 0.f };

            vertices[1].normal = normal;
            vertices[1].uv = QVector2D{ 0.f, 1.f };

            vertices[2].normal = normal;
            vertices[2].uv = QVector2D{ 1.f, 1.f };

            vertices[3].normal = normal;
            vertices[3].uv = QVector2D{ 1.f, 0.f };
        }

        Vertex vertices[4];
    };
    constexpr std::array CUBE_DATA
    {
        // front
        Plane { Vertex{ QVector3D{ -0.5f, -0.5f,  0.5f } },
                Vertex{ QVector3D{ -0.5f,  0.5f,  0.5f } },
                Vertex{ QVector3D{  0.5f,  0.5f,  0.5f } },
                Vertex{ QVector3D{  0.5f, -0.5f,  0.5f } }, },
        // left
        Plane { Vertex{ QVector3D{ -0.5f, -0.5f, -0.5f } },
                Vertex{ QVector3D{ -0.5f,  0.5f, -0.5f } },
                Vertex{ QVector3D{ -0.5f,  0.5f,  0.5f } },
                Vertex{ QVector3D{ -0.5f, -0.5f,  0.5f } } },
        // right
        Plane { Vertex{ QVector3D{  0.5f, -0.5f,  0.5f } },
                Vertex{ QVector3D{  0.5f,  0.5f,  0.5f } },
                Vertex{ QVector3D{  0.5f,  0.5f, -0.5f } },
                Vertex{ QVector3D{  0.5f, -0.5f, -0.5f } } },
        // back
        Plane { Vertex{ QVector3D{  0.5f, -0.5f, -0.5f } },
                Vertex{ QVector3D{  0.5f,  0.5f, -0.5f } },
                Vertex{ QVector3D{ -0.5f,  0.5f, -0.5f } },
                Vertex{ QVector3D{ -0.5f, -0.5f, -0.5f } } },
        // top
        Plane { Vertex{ QVector3D{ -0.5f,  0.5f,  0.5f } },
                Vertex{ QVector3D{ -0.5f,  0.5f, -0.5f } },
                Vertex{ QVector3D{  0.5f,  0.5f, -0.5f } },
                Vertex{ QVector3D{  0.5f,  0.5f,  0.5f } } },
        // bottom
        Plane { Vertex{ QVector3D{ -0.5f, -0.5f, -0.5f } },
                Vertex{ QVector3D{ -0.5f, -0.5f,  0.5f } },
                Vertex{ QVector3D{  0.5f, -0.5f,  0.5f } },
                Vertex{ QVector3D{  0.5f, -0.5f, -0.5f } } }
    };
    constexpr std::array<uint16_t, 36> INDICIES_DATA
    {
         0,  1,  2,  0,  2,  3, // front
         4,  5,  6,  4,  6,  7, // left
         8,  9, 10,  8, 10, 11, // right
        12, 13, 14, 12, 14, 15, // back
        16, 17, 18, 16, 18, 19, // top
        20, 21, 22, 20, 22, 23  // bottom
    };
    constexpr int32_t VERTEX_STRIDE = sizeof(Vertex);
    constexpr int32_t POSITION_OFFSET = offsetof(Vertex, position);
    constexpr int32_t NORMAL_OFFSET = offsetof(Vertex, normal);
    constexpr int32_t UV_OFFSET = offsetof(Vertex, uv);
    constexpr int32_t COLOR_OFFSET = offsetof(Vertex, color);

    m_vao = new QOpenGLVertexArrayObject;
    m_vao->create();
    m_vao->bind();
    m_vertexBuffer = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    m_vertexBuffer->create();
    m_vertexBuffer->bind();
    m_vertexBuffer->allocate(CUBE_DATA.data(), sizeof(CUBE_DATA));
    m_shader->bind();
    m_shader->setAttributeBuffer(m_positionLocation, GL_FLOAT, POSITION_OFFSET, 3, VERTEX_STRIDE);
    m_shader->enableAttributeArray(m_positionLocation);
    m_shader->setAttributeBuffer(m_normalLocation, GL_FLOAT, NORMAL_OFFSET, 3, VERTEX_STRIDE);
    m_shader->enableAttributeArray(m_normalLocation);
    m_shader->setAttributeBuffer(m_uvLocation, GL_FLOAT, UV_OFFSET, 2, VERTEX_STRIDE);
    m_shader->enableAttributeArray(m_uvLocation);
    m_shader->setAttributeBuffer(m_colorLocation, GL_UNSIGNED_SHORT, COLOR_OFFSET, 4, VERTEX_STRIDE);
    m_shader->enableAttributeArray(m_colorLocation);
    m_indexBuffer = new QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
    m_indexBuffer->create();
    m_indexBuffer->bind();
    m_indexBuffer->allocate(INDICIES_DATA.data(), sizeof(INDICIES_DATA));

    m_indexCount = static_cast<int32_t>(INDICIES_DATA.size());
}

// ** NauMaterialPreview

NauMaterialPreview::NauMaterialPreview(NauWidget* parent)
    : NauWidget(parent)
{
    setLayout(new NauLayoutHorizontal(this));
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    layout()->setContentsMargins(0, 0, 0, 0);

    createPreviewSpoiler();
    createPreviewContent();
}

void NauMaterialPreview::setTitle(const QString& title)
{
    m_spoiler->setText(title);
}


void NauMaterialPreview::setMaterial(const NauPropertiesContainer& materialParams)
{
    // TODO: temporary texture setup
    const auto albedoIt = materialParams.find("texName");
    if (albedoIt != materialParams.end()) {
        m_previewContent->setAlbedo(albedoIt.value().value().convert<QString>());
    }
}

void NauMaterialPreview::createPreviewSpoiler()
{
    m_spoiler = new NauInspectorSubWindow(this);
    layout()->addWidget(m_spoiler);
}

void NauMaterialPreview::createPreviewContent()
{
    m_previewContent = new NauMaterialPreviewRender(this);
    auto* contentLayout = new NauLayoutVertical();
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->addWidget(m_previewContent);
    m_spoiler->setContentLayout(*contentLayout);
}
