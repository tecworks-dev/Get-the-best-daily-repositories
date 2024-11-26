// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_analytics.hpp"

#include "providers/nau_analytics_provider_interface.hpp"


// ** NauAnalytics

void NauAnalytics::addAnalyticsProvider(std::unique_ptr<NauAnalyticsProviderInterface> analyticsProvider)
{
    m_analyticsProvider.push_back(std::move(analyticsProvider));
}

void NauAnalytics::sendUserInfo(const std::map<std::string, std::string>& userDetails)
{
    for (const auto& analyticProvider : m_analyticsProvider) {
        if (analyticProvider) {
            analyticProvider->sendUserInfo(userDetails);
        }
    }
}

void NauAnalytics::sendSystemMetrics(const std::string& os, const std::string& osVersion, const std::string& device, const std::string& resolution, const std::string& appVersion)
{
    for (const auto& analyticProvider : m_analyticsProvider) {
        if (analyticProvider) {
            analyticProvider->sendSystemMetrics(os, osVersion, device, resolution, appVersion);
        }
    }
}

void NauAnalytics::sendEvent(const std::string& key)
{
    for (const auto& analyticProvider : m_analyticsProvider) {
        if (analyticProvider) {
            analyticProvider->sendEvent(key);
        }
    }
}

void NauAnalytics::sendEvent(const std::string& key, const std::map<std::string, std::string>& value)
{
    for (const auto& analyticProvider : m_analyticsProvider) {
        if (analyticProvider) {
            analyticProvider->sendEvent(key, value);
        }
    }
}
