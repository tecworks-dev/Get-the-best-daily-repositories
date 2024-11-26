// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_vfx_file.hpp"

#include "nau_vfx.hpp"
#include "nau_log.hpp"
#include "nau_utils.hpp"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

// ** NauVFXFile

NauVFXFile::NauVFXFile(std::string filePath) noexcept
    : m_filePath(std::move(filePath))
{
}

void NauVFXFile::createVFX() const
{
    QFile actionFile(m_filePath.c_str());
    if (!actionFile.open(QIODevice::WriteOnly)) {
        NED_ERROR("Failed to create file for VFX {}.", m_filePath);
        return;
    }

    // Structure of the VFX system json file for the first time
    /* {
        "position": [] ,
        "rotation" : [] ,
        "auto_start" : true,
        "loop" : true,
        "effects" : [
            "name1" :[spawn, life, position, color, size],
            "name2" : [spawn, life, position, color, size],
            ...
            "nameN" : [spawn, life, position, color, size],
        ]
    }
    */

    QJsonObject root;
    root["position"] = convertUtils::QVector3DToJsonValue(QVector3D(0, 0, 0));
    root["rotation"] = convertUtils::QVector3DToJsonValue(QVector3D(0, 0, 0));
    root["auto_start"] = false;
    root["loop"] = false;
    root["effects"] = QJsonArray();

    const QJsonDocument vfxDocument{ root };
    const QByteArray vfxData = vfxDocument.toJson();

    if (actionFile.write(vfxData.data()) == -1) {
        NED_ERROR("Failed write to file for VFX {}.", m_filePath);
        return;
    }
    NED_TRACE("Create VFX: %s.", m_filePath);
}

bool NauVFXFile::loadVFX(NauVFX& vfx) const
{
    QFile vfxFile(m_filePath.c_str());
    if (!vfxFile.open(QIODevice::ReadOnly)) {
        NED_ERROR("Failed to create vfxPath for VFX {}.", m_filePath);
        return false;
    }

    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(vfxFile.readAll(), &error);
    if (error.error != QJsonParseError::NoError) {
        NED_ERROR("Failed to load VFX with error: {}.", error.errorString());
        return false;
    }

    const std::string vfxName = QFileInfo(m_filePath.c_str()).baseName().toUtf8().constData();
    vfx.setName(vfxName);

    const QJsonObject root = document.object();

    QString errorMessage;
    const QVector3D position = convertUtils::JsonValueToQVector3D(root["position"], errorMessage);
    if (!errorMessage.isEmpty()) {
        NED_ERROR("Failed to load position in VFX with error: {}.", errorMessage);
        return false;
    }
    vfx.setPosition(position);

    const QVector3D rotation = convertUtils::JsonValueToQVector3D(root["rotation"], errorMessage);
    if (!errorMessage.isEmpty()) {
        NED_ERROR("Failed to load rotation in VFX with error: {}.", errorMessage);
        return false;
    }
    vfx.setRotation(rotation);

    if (!root.contains("auto_start") || !root["auto_start"].isBool()) {
        NED_ERROR("Invalid or missing auto_start value.");
        return false;
    }
    const bool isAutoStart = root["auto_start"].toBool();
    vfx.setAutoStart(isAutoStart);

    if (!root.contains("loop") || !root["loop"].isBool()) {
        NED_ERROR("Invalid or missing loop value.");
        return false;
    }
    const bool isLoop = root["loop"].toBool();
    vfx.setLoop(isLoop);

    // TODO Parcing effects

    return true;
}

void NauVFXFile::saveVFX(const NauVFX& vfx) const
{
    QFile actionsFile(m_filePath.c_str());
    if (!actionsFile.open(QIODevice::WriteOnly)) {
        NED_ERROR("Failed to open file for VFX {}.", m_filePath);
        return;
    }

    QJsonObject root;
    root["position"] = convertUtils::QVector3DToJsonValue(vfx.position());
    root["rotation"] = convertUtils::QVector3DToJsonValue(vfx.rotation());
    root["auto_start"] = vfx.isAutoStart();
    root["loop"] = vfx.isLoop();
    root["effects"] = QJsonArray();

    // TODO Write effects

    const QJsonDocument vfxDocument{ root };
    const QByteArray vfxData = vfxDocument.toJson();

    if (actionsFile.write(vfxData.data()) == -1) {
        NED_ERROR("Failed write to file for VFX {}.", m_filePath);
        return;
    }
}

