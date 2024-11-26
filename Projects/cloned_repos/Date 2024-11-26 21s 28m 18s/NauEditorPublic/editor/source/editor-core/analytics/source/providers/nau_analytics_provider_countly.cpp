// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_analytics_provider_countly.hpp"

#include "countly.hpp"
#include <curl/curl.h>


// ** NauAnalyticsProviderCountly

NauAnalyticsProviderCountly::NauAnalyticsProviderCountly(const std::string& appKey, const std::string& host, int port,
    const std::string& deviceId, bool isSameUser,
    const std::string& pathToDatabase, unsigned short updateInterval, unsigned int batchSize)
{
    cly::Countly& countlyInstance = cly::Countly::getInstance();

    countlyInstance.alwaysUsePost(true);
    countlyInstance.setHTTPClient([=](bool isPost, const std::string & url, const std::string & data)->cly::HTTPResponse {
        CURL* curl;
        CURLcode res;
        cly::HTTPResponse response;
        response.success = false;

        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl = curl_easy_init();
        if (curl) {
            std::string request = host + ":" + std::to_string(port) + url + "?" + data;
            curl_easy_setopt(curl, CURLOPT_URL, request.c_str());
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

            std::string response_string;
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](void* ptr, size_t size, size_t nmemb, std::string* data) {
                data->append((char*)ptr, size * nmemb);
                return size * nmemb;
                });
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

            res = curl_easy_perform(curl);

            if (res != CURLE_OK) {
                auto tmp = curl_easy_strerror(res);
            }
            else {
                response.success = true;
                response.data = nlohmann::json::parse(response_string);
            }

            curl_easy_cleanup(curl);
        }

        curl_global_cleanup();

        return response;
    });

    // Path to database for storing and processing unprocessed events
    if (pathToDatabase.empty()) {
        countlyInstance.SetPath("nau_countly_analytics_provider.db");
    } else {
        countlyInstance.SetPath(pathToDatabase);
    }

    countlyInstance.setAutomaticSessionUpdateInterval(updateInterval);
    countlyInstance.setMaxRQProcessingBatchSize(batchSize);

    // Countly needs to get the deviceID before start
    countlyInstance.setDeviceID(deviceId, isSameUser);

    // The fourth parameter must be true (separate thread).
    countlyInstance.start(appKey, host, port, true);
}

NauAnalyticsProviderCountly::~NauAnalyticsProviderCountly()
{
    cly::Countly::getInstance().stop();
}

void NauAnalyticsProviderCountly::sendUserInfo(const std::map<std::string, std::string>& userDetails)
{
    cly::Countly::getInstance().setUserDetails(userDetails);
}

void NauAnalyticsProviderCountly::sendSystemMetrics(const std::string& os, const std::string& osVersion, const std::string& device, const std::string& resolution, const std::string& appVersion)
{
    // The empty string means "carrier".
    cly::Countly::getInstance().setMetrics(os, osVersion, device, resolution, "", appVersion);
}

void NauAnalyticsProviderCountly::sendEvent(const std::string& key)
{
    // The second parameter means the number of events (the internal counter incemented by 1)
    cly::Countly::getInstance().RecordEvent(key, 1);
}

void NauAnalyticsProviderCountly::sendEvent(const std::string& key, const std::map<std::string, std::string>& value)
{
    // The third parameter is similar to the previous one 
    cly::Countly::getInstance().RecordEvent(key, value, 1);
}
