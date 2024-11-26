// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include <QApplication>
#include <QString>

#include <set>
#include <vector>
#include <filesystem>


// ** NauPhysicsCollisionChannelModel
// Operates collision channels of the physics.

class NAU_EDITOR_ENGINE_API NauPhysicsCollisionChannelModel
{
    Q_DECLARE_TR_FUNCTIONS(NauPhysicsCollisionChannelModel)

public:
    using Channel = int;

    struct CollisionChannel
    {
        std::string name;
        Channel channel;
        std::set<Channel> collidesWithChannel;
    };

    void initialize(const std::filesystem::path& projectPath);

    const CollisionChannel& addChannel(const std::string& channelName = std::string());
    bool deleteChannel(Channel channel);
    bool renameChannel(Channel channel, const std::string& newName);

    bool loadChannels();
    bool saveChannels();

    const std::vector<CollisionChannel>& channels() const;
    void applySettingsToPhysicsWorld() const;

    CollisionChannel const* getCollisionChannel(Channel channel) const;

    bool setChannelsCollideable(Channel channelA, Channel channelB, bool collideable);
    bool channelsCollideable(Channel channelA, Channel channelB) const;

    static Channel defaultChannel();
private:
    CollisionChannel* getChannelInternal(Channel channel);
    const CollisionChannel* getChannelInternal(Channel channel) const;
    std::filesystem::path getCollisionChannelSettingsFile() const;
    bool resetToDefault();

private:
    std::filesystem::path m_projectPath;
    std::vector<CollisionChannel> m_channels;
};

namespace Nau
{
    NAU_EDITOR_ENGINE_API NauPhysicsCollisionChannelModel& getPhysicsCollisionChannelModel();
}