// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Storage location for the editor engine public math utilities

#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include "nau/math/math.h"

#include <QGenericMatrix>


// ** NauMathMatrixUtils
//
// Public math api for matrix

namespace Vectormath::SSE{
class Transform3;
class Quat;
}

class NAU_EDITOR_ENGINE_API NauMathMatrixUtils
{
public:
    NauMathMatrixUtils() = delete;

    struct RotationRadians
    {
        float pitch = 0.f;
        float yaw = 0.f;
        float roll = 0.f;
    };

    static bool isEqual(float left, float right, float tolerance = 0.00001f) noexcept;
    static RotationRadians convertQuaternionToRadiansYX(const Vectormath::SSE::Quat& quaternion) noexcept;
    static void convertQMatrixToNauTransform(const QMatrix4x3& matrix, Vectormath::SSE::Transform3& result) noexcept;
    static QMatrix4x3 convertNauTransformToQMatrix(const Vectormath::SSE::Transform3& transform) noexcept;
    static QMatrix3x3 MatrixToTRSComposition(const QMatrix4x3& matrix) noexcept;
    static QMatrix4x3 TRSCompositionToMatrix(const QMatrix3x3& trsComposition) noexcept;
    static void orthonormalize(nau::math::mat4& m);
};
