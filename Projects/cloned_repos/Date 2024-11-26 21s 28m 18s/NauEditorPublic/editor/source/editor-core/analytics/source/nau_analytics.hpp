// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// An analytics class that contains analytics providers and sends analytics to the services
// (Countly, Yandex, Google e.t.c)

#pragma once

#include <string>
#include <memory>
#include <list>
#include <map>


class NauAnalyticsProviderInterface;

// ** NauAnalytics

class NauAnalytics
{
public:
    NauAnalytics() = default;
    ~NauAnalytics() = default;

public:
    // Non copyable
    NauAnalytics(const NauAnalytics&) = delete;
    NauAnalytics& operator=(const NauAnalytics&) = delete;

    // Non moveable
    NauAnalytics(const NauAnalytics&&) noexcept = delete;
    NauAnalytics& operator=(NauAnalytics&&) noexcept = delete;

public:
    void addAnalyticsProvider(std::unique_ptr<NauAnalyticsProviderInterface> analyticsProvider);

public:
    void sendUserInfo(const std::map<std::string, std::string>& userDetails);
    void sendSystemMetrics(const std::string& os, const std::string& osVersion, const std::string& device, const std::string& resolution, const std::string& appVersion);

public:
    void sendEvent(const std::string& key);
    void sendEvent(const std::string& key, const std::map<std::string, std::string>& value);

private:
    std::list<std::unique_ptr<NauAnalyticsProviderInterface>> m_analyticsProvider;
};
