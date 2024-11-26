// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/math/nau_matrix_math.hpp"

#include <nau/math/math.h>

#include <QVector4D>

namespace Nau
{
    static nau::math::AffineTransform decomposeToAffineTransform(const nau::math::Transform3& matrix)
    {
        using vec3_t = nau::math::vec3;

        nau::math::AffineTransform affineTransform;
        const vec3_t col0 = matrix.getCol0();
        const vec3_t col1 = matrix.getCol1();
        const vec3_t col2 = matrix.getCol2();

        affineTransform.translation = matrix.getTranslation();
        affineTransform.scale = {
            length(col0),
            length(col1),
            length(col2)
        };
        if (dot(col2, cross(col0, col1)) < 0.f) {
            affineTransform.scale.setZ(affineTransform.scale.getZ() * -1.f);
        }
        affineTransform.rotation = normalize(nau::math::quat{ {
            col0 / affineTransform.scale.getX(),
            col1 / affineTransform.scale.getY(),
            col2 / affineTransform.scale.getZ()
        } });

        return affineTransform;
    }
}


// ** NauMathMatrixUtils

bool NauMathMatrixUtils::isEqual(float left, float right, float tolerance) noexcept
{
    return std::abs(left - right) <= tolerance;
}


NauMathMatrixUtils::RotationRadians NauMathMatrixUtils::convertQuaternionToRadiansYX(const Vectormath::SSE::Quat& quaternion) noexcept
{
    using quat_t = nau::math::quat;
    using vec3_t = nau::math::vec3;

    RotationRadians radians;
    quat_t newCameraRotation = quaternion;
    {
        const vec3_t rightAxis = normalize(vec3_t{ rotate(newCameraRotation, vec3_t::xAxis()) }.setY(0.f));
        radians.yaw = std::acosf(dot(rightAxis, vec3_t::xAxis()));
        if (rightAxis.getZ() > 0.f) {
            radians.yaw = -radians.yaw;
        }
    }
    // Remove rotation around y-axis
    newCameraRotation = quat_t::rotationY(-radians.yaw) * newCameraRotation;
    {
        const vec3_t forwardAxis = normalize(vec3_t{ rotate(newCameraRotation, vec3_t::zAxis()) }.setX(0.f));
        radians.pitch = std::acosf(dot(forwardAxis, vec3_t::zAxis()));
        if (forwardAxis.getY() > 0.f) {
            radians.pitch = -radians.pitch;
        }
    }

    return radians;
}


void NauMathMatrixUtils::convertQMatrixToNauTransform(const QMatrix4x3& matrix, nau::math::Transform3& result) noexcept
{
    constexpr int MAX_ROWS = 3;
    constexpr int MAX_COLUMNS = 4;

    const float* matrixData = matrix.data();

    for (int column = 0; column < MAX_COLUMNS; ++column) {
        for (int row = 0; row < MAX_ROWS; ++row) {
            result.setElem( column, row, matrixData[column * MAX_ROWS + row] );
        }
    }
    result.setElem(3, 3, 1.f);
}


QMatrix4x3 NauMathMatrixUtils::convertNauTransformToQMatrix(const nau::math::Transform3& transform) noexcept
{
    constexpr int MAX_ROWS = 3;
    constexpr int MAX_COLUMNS = 4;

    QMatrix4x3 result;
    for (int column = 0; column < MAX_COLUMNS; ++column) {
        for (int row = 0; row < MAX_ROWS; ++row) {
            result.data()[column * MAX_ROWS + row] = transform.getElem(column, row);
        }
    }
    return result;
}


QMatrix3x3 NauMathMatrixUtils::MatrixToTRSComposition(const QMatrix4x3& matrix) noexcept
{
    nau::math::Transform3 matrixTransform;
    convertQMatrixToNauTransform(matrix, matrixTransform);

    const nau::math::AffineTransform transform = Nau::decomposeToAffineTransform(matrixTransform);
    const nau::math::quat& rotation = transform.rotation;
    const auto [x, y, z] = convertQuaternionToRadiansYX(rotation);

    QMatrix3x3 trsComposition;

    //// Set transform
    trsComposition.data()[0] = transform.translation[0];
    trsComposition.data()[1] = transform.translation[1];
    trsComposition.data()[2] = transform.translation[2];

    //// Set rotation
    trsComposition.data()[3] = nau::math::radToDeg(x);
    trsComposition.data()[4] = nau::math::radToDeg(y);
    trsComposition.data()[5] = nau::math::radToDeg(z);

    //// Set scale
    trsComposition.data()[6] = transform.scale[0];
    trsComposition.data()[7] = transform.scale[1];
    trsComposition.data()[8] = transform.scale[2];

    return trsComposition;
}


QMatrix4x3 NauMathMatrixUtils::TRSCompositionToMatrix(const QMatrix3x3& trsComposition) noexcept
{
    using vec3_t = nau::math::vec3;
    using mat3_t = nau::math::mat3;
    using quat_t = nau::math::quat;
    using tf3_t = nau::math::Transform3;

    constexpr int TR_FIRST_INDEX = 0;
    constexpr int ROT_FIRST_INDEX = 3;
    constexpr int SCL_FIRST_INDEX = 6;

    const float* data = trsComposition.data();
    const auto toRad = [] (float deg) { return nau::math::degToRad(deg); };
    const vec3_t translate{ data[TR_FIRST_INDEX],  data[TR_FIRST_INDEX + 1],  data[TR_FIRST_INDEX + 2] };
    const vec3_t rotate{ toRad(data[ROT_FIRST_INDEX]), toRad(data[ROT_FIRST_INDEX + 1]), toRad(data[ROT_FIRST_INDEX + 2]) };
    const vec3_t scale{ data[SCL_FIRST_INDEX], data[SCL_FIRST_INDEX + 1], data[SCL_FIRST_INDEX + 2] };

    const mat3_t rotation = mat3_t::rotationZYX(rotate);
    const tf3_t transform{ nau::math::appendScale(rotation, scale), translate };

    return convertNauTransformToQMatrix(transform);
}

void NauMathMatrixUtils::orthonormalize(nau::math::mat4& m)
{
    nau::math::mat3 m3 = m.getUpper3x3();
    m3.setCol2(Vectormath::SSE::normalize(
        Vectormath::cross(m3.getCol0(), m3.getCol1()))
    );
    m3.setCol1(Vectormath::SSE::normalize(
        Vectormath::cross(m3.getCol2(), m3.getCol0()))
    );
    m3.setCol0(Vectormath::SSE::normalize(
        Vectormath::cross(m3.getCol1(), m3.getCol2()))
    );
    m.setUpper3x3(m3);
}