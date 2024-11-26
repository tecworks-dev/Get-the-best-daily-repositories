// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// The file in which the concepts used in the engine will be stored

#pragma once

#include <concepts>
#include <type_traits>

#include <QString>
#include <QColor>
#include <QVector2D>
#include <QVector3D>
#include <QVector4D>
#include <QGenericMatrix>

#include "nau_types.hpp"


// A basic concept that checks that the type (T) passed to the template corresponds to one of a set (U)
template<typename T, typename ... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

// Specialization of the IsAnyOf concept to work with types allowed to work in the engine
// (For example, those allowed to work in the entity manager)
template<typename T>
concept IsNauType = IsAnyOf<T, bool, int, float, std::string, QString, QMatrix4x3, QVector2D, QVector3D, QVector4D, QColor,
        NauRangedValue<int>, NauRangedValue<float>, NauRangedPair<int>, NauRangedPair<float>>;
