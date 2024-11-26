// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// VFX class that stores basic VFX properties and auxiliary classes

#pragma once

#include "scene/nau_world.hpp"
#include "inspector/nau_object_inspector.hpp"

#include <string>
#include <vector>

#include <QVector3D>


// ** NauVFXPtr

class NauVFX;
using NauVFXPtr = std::shared_ptr<NauVFX>;


// ** NauEmitterType

enum class NauEmitterType : uint8_t
{
    Unknown,
    Box,
    Cone,
    Sphere,
    SphereSector,
};


// ** NauVFX

class NauVFX : public QObject
{
    Q_OBJECT

public:
    NauVFX();

    void setName(const std::string& vfxName);
    std::string name() const noexcept;

    void setPosition(const QVector3D& vfxName);
    QVector3D position() const noexcept;

    void setRotation(const QVector3D& vfxName);
    QVector3D rotation() const noexcept;

    void setAutoStart(bool isAutoStart);
    bool isAutoStart() const noexcept;

    void setLoop(bool isLoop);
    bool isLoop() const noexcept;

signals:
    void eventVFXChanged();

private:
    NauObjectProperty m_vfxName;
    NauObjectProperty m_canOverride;

    NauObjectProperty m_position;
    NauObjectProperty m_rotation;

    NauObjectProperty m_isAutoStart;
    NauObjectProperty m_isLoop;
};
