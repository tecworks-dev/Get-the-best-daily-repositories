// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Class for the Countly analytics provider

#pragma once

#include "nau_analytics_provider_interface.hpp"


// ** NauAnalyticsProviderCountly

class NauAnalyticsProviderCountly : public NauAnalyticsProviderInterface
{
public:
    NauAnalyticsProviderCountly(const std::string& appKey, const std::string& host, int port,
        const std::string& deviceId, bool isSameUser,
        // updateInterval in sec
        // batchSize - size of request queue
        const std::string& pathToDatabase, unsigned short updateInterval = 15, unsigned int batchSize = 5);
    ~NauAnalyticsProviderCountly();

public:
    void sendUserInfo(const std::map<std::string, std::string>& userDetails) override;
    void sendSystemMetrics(const std::string& os, const std::string& osVersion, const std::string& device, const std::string& resolution, const std::string& appVersion) override;

public:
    void sendEvent(const std::string& key) override;
    void sendEvent(const std::string& key, const std::map<std::string, std::string>& value) override;
};
