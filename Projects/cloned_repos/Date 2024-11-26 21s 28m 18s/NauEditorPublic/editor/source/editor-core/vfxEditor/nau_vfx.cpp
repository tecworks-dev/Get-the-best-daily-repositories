// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_vfx.hpp"
#include "nau_log.hpp"
#include "magic_enum/magic_enum.hpp"
#include "nau_assert.hpp"

#include <utility>


// ** NauVFX

NauVFX::NauVFX()
    : m_vfxName(tr("VFX name"), NauAnyType(QString()))
    , m_canOverride(tr("Override"), NauAnyType(false))
{
}

void NauVFX::setName(const std::string& vfxName)
{
    m_vfxName.setValue(static_cast<QString>(vfxName.c_str()));
    emit eventVFXChanged();
}

std::string NauVFX::name() const noexcept
{
    return m_vfxName.value().convert<QString>().toUtf8().constData();
}

void NauVFX::setPosition(const QVector3D& position)
{
    m_position.setValue(position);
    emit eventVFXChanged();
}

QVector3D NauVFX::position() const noexcept
{
    return m_position.value().convert<QVector3D>();
}

void NauVFX::setRotation(const QVector3D& rotation)
{
    m_rotation.setValue(rotation);
    emit eventVFXChanged();
}

QVector3D NauVFX::rotation() const noexcept
{
    return m_rotation.value().convert<QVector3D>();
}

void NauVFX::setAutoStart(bool isAutoStart)
{
    m_isAutoStart.setValue(isAutoStart);
    emit eventVFXChanged();
}

bool NauVFX::isAutoStart() const noexcept
{
    return m_isAutoStart.value().convert<bool>();
}

void NauVFX::setLoop(bool isLoop)
{
    m_isLoop.setValue(isLoop);
    emit eventVFXChanged();
}

bool NauVFX::isLoop() const noexcept
{
    return m_isLoop.value().convert<bool>();
}
