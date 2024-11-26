// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/physics/nau_phyisics_collision_channel_model.hpp"
#include "nau/service/service_provider.h"
#include "nau/physics/physics_world.h"
#include "nau/physics/core_physics.h"
#include "nau/diag/logging.h"

#include <QFile>
#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <iterator>


// ** NauPhysicsCollisionChannelModel

void NauPhysicsCollisionChannelModel::initialize(const std::filesystem::path& projectPath)
{
    m_projectPath = projectPath;

    loadChannels();
}

const NauPhysicsCollisionChannelModel::CollisionChannel& NauPhysicsCollisionChannelModel::addChannel(const std::string& channelName)
{
    CollisionChannel collisionChannel;
    std::optional<Channel> maxChannel;

    for (const auto& channel : m_channels) {
        if (!maxChannel || channel.channel > maxChannel) {
            maxChannel = channel.channel;
        }
    }

    collisionChannel.channel = maxChannel ? *maxChannel + 1 : 0;
    collisionChannel.name = channelName.empty() 
        ? tr("Channel %1").arg(collisionChannel.channel).toStdString()
        : channelName;

    m_channels.push_back(collisionChannel);

    return m_channels.back();
}

bool NauPhysicsCollisionChannelModel::deleteChannel(Channel channel)
{
    bool deleted = false;
    for (auto it = m_channels.begin(); it != m_channels.end();) {
        if (it->channel == channel) {
            it = m_channels.erase(it);
            deleted = true;
        } else {
            it->collidesWithChannel.erase(channel);
            ++it;
        }
    }

    return deleted;
}

bool NauPhysicsCollisionChannelModel::renameChannel(Channel channel, const std::string& newName)
{
    if (CollisionChannel* collisionChannel = getChannelInternal(channel)) {
        collisionChannel->name = newName;
        return true;
    }

    return false;
}

bool NauPhysicsCollisionChannelModel::loadChannels()
{
    const auto collisionChannelFileName = getCollisionChannelSettingsFile();

    QFile file(collisionChannelFileName);
    if (!file.open(QIODevice::ReadOnly)) {
        NAU_LOG_WARNING("Collision settings file {} missing or contains none channels. "
             "Creating default channel, for the physics world has at least one channel", collisionChannelFileName.string());

        return resetToDefault();
    }

    m_channels.clear();

    const auto jsonDoc = QJsonDocument::fromJson(file.readAll());
    const QJsonObject object = jsonDoc.object();
    for (const QJsonValue& value : object["collisionChannels"].toArray()) {
        QJsonObject obj = value.toObject();
        
        m_channels.push_back(CollisionChannel{
            .name = obj["name"].toString().toStdString(),
            .channel = static_cast<Channel>(obj["channel"].toInteger()),
            .collidesWithChannel = {}
        });

        for (const QJsonValue& colValue : obj["collisions"].toArray()) {
            m_channels.back().collidesWithChannel.insert(colValue.toInteger());
        }
    }

    if (m_channels.empty()) {
        NAU_LOG_WARNING("Collision channel settings has none channels."
                "Creating default channel, for the physics world has at least one channel");

       return resetToDefault();
    }

    NAU_LOG_DEBUG("Physics collision settings loaded from {}", collisionChannelFileName.string());
    return true;
}

bool NauPhysicsCollisionChannelModel::saveChannels()
{
    const auto fileName = getCollisionChannelSettingsFile();
    const QDir filePath = QFileInfo(fileName).absolutePath();

    const bool mkResult = filePath.mkpath(".");
    if (!mkResult) {
        NAU_LOG_ERROR("Failed to save physics channel settings {}, cannot create path:{}",
            fileName.string(), filePath.absolutePath().toStdString());
        return false;
    }

    QFile file{fileName};
    if (!file.open(QIODevice::WriteOnly)) {
        NAU_LOG_ERROR("Failed to save physics channel settings {}:{}", fileName.string(), file.errorString().toStdString());
        return false;
    }

    QJsonObject result;
    QJsonArray records;

    for (const auto& channel : m_channels) {
        QJsonObject jsonRow;
        QJsonArray jsonCollisions;
        for (const auto& collidableChannel : channel.collidesWithChannel) {
            jsonCollisions.push_back(static_cast<int>(collidableChannel));
        }

        jsonRow["collisions"] = jsonCollisions;
        jsonRow["channel"] = static_cast<int>(channel.channel);
        jsonRow["name"] = QString::fromStdString(channel.name);
        records.append(jsonRow);
    }
    result.insert("collisionChannels", records);

    QJsonDocument jsonDoc;
    jsonDoc.setObject(result);
    file.write(jsonDoc.toJson());
    file.close();

    NAU_LOG_DEBUG("Saved collision channel configuration to {}", fileName.string());
    return true;
}

const std::vector<NauPhysicsCollisionChannelModel::CollisionChannel>&
NauPhysicsCollisionChannelModel::channels() const
{
    return m_channels;
}

void NauPhysicsCollisionChannelModel::applySettingsToPhysicsWorld() const
{
    using namespace nau;
    using namespace nau::physics;

    std::set<std::pair<Channel, Channel>> collisions;
    for (const auto& channel : m_channels) {
        for (Channel otherChannel : channel.collidesWithChannel) {
            collisions.insert(std::make_pair(
                std::min(channel.channel, otherChannel),
                std::max(channel.channel, otherChannel)
            ));
        }
    }

    auto physWorld = getServiceProvider().get<ICorePhysics>().getDefaultPhysicsWorld();

    physWorld->resetChannelsCollisionSettings();

    for (const auto&[channelA, channelB] : collisions) {
        physWorld->setChannelsCollidable(channelA, channelB);
    }
}

NauPhysicsCollisionChannelModel::CollisionChannel const*
NauPhysicsCollisionChannelModel::getCollisionChannel(Channel collisionChannel) const
{
    for (const auto& channel : m_channels) {
        if (channel.channel == collisionChannel) {
            return &channel;
        }
    }

    return nullptr;
}

bool NauPhysicsCollisionChannelModel::setChannelsCollideable(Channel channelA, Channel channelB, bool collideable)
{
    CollisionChannel* collisionChannelA = getChannelInternal(channelA);
    CollisionChannel* collisionChannelB = getChannelInternal(channelB);

    if (collisionChannelA && collisionChannelB) {
        if (collideable) {
            collisionChannelA->collidesWithChannel.insert(channelB);
            collisionChannelB->collidesWithChannel.insert(channelA);
        } else {
            collisionChannelA->collidesWithChannel.erase(channelB);
            collisionChannelB->collidesWithChannel.erase(channelA);
        }

        return true;
    }

    return false;
}

bool NauPhysicsCollisionChannelModel::channelsCollideable(Channel channelA, Channel channelB) const
{
    const CollisionChannel* collisionChannelA = getChannelInternal(channelA);
    return collisionChannelA && collisionChannelA->collidesWithChannel.contains(channelB);
}

NauPhysicsCollisionChannelModel::CollisionChannel* NauPhysicsCollisionChannelModel::getChannelInternal(Channel collisionChannel)
{
    for (auto& channel : m_channels) {
        if (channel.channel == collisionChannel) {
            return &channel;
        }
    }

    return nullptr;
}

const NauPhysicsCollisionChannelModel::CollisionChannel* NauPhysicsCollisionChannelModel::getChannelInternal(Channel collisionChannel) const
{
    for (const auto& channel : m_channels) {
        if (channel.channel == collisionChannel) {
            return &channel;
        }
    }

    return nullptr;
}

NauPhysicsCollisionChannelModel::Channel NauPhysicsCollisionChannelModel::defaultChannel()
{
    return Channel{};
}

std::filesystem::path NauPhysicsCollisionChannelModel::getCollisionChannelSettingsFile() const
{
    return m_projectPath / std::filesystem::path("content/physics/channels.data");
}

bool NauPhysicsCollisionChannelModel::resetToDefault()
{
    m_channels = {
        CollisionChannel{
            .name = "Default",
            .channel = defaultChannel(),
            .collidesWithChannel = {defaultChannel()}
        }
    };

    return saveChannels();
}

namespace Nau
{
    NauPhysicsCollisionChannelModel& getPhysicsCollisionChannelModel()
    {
        static NauPhysicsCollisionChannelModel model;
        return model;
    }
}
