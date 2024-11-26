// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Interface for any analytics provider

#pragma once

#include <string>
#include <map>


// ** NauAnalyticsProviderInterface

class NauAnalyticsProviderInterface
{
public:
    virtual ~NauAnalyticsProviderInterface() = default;

public:
    virtual void sendUserInfo(const std::map<std::string, std::string>& userDetails) = 0;
    virtual void sendSystemMetrics(const std::string& os, const std::string& osVersion, const std::string& device, const std::string& resolution, const std::string& appVersion) = 0;

public:
    virtual void sendEvent(const std::string& key) = 0;
    virtual void sendEvent(const std::string& key, const std::map<std::string, std::string>& value) = 0;
};
