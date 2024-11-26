// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Log module related utils

#pragma once

#include "nau/nau_editor_engine_log_config.hpp"
#include "nau/diag/source_info.h"
#include "nau/diag/logging.h"

#include <QObject>


// ** NauEngineLogLevel

enum class NauEngineLogLevel : uint16_t
{
    /** Prints a info in Debug configuration to console (and log file) */
    Debug,

    /** Prints a info to console (and log file) */
    Info,

    /** Prints a warning to console (and log file) */
    Warning,

    /** Sends an error and crashes in release builds */
    Error,

    /** Sends a fatal error and crashes */
    Critical,

    /** Sends a verbose message (if Verbose logging is enabled), usually used for detailed logging. */
    Verbose,
};


// ** NauEngineLogger

class NAU_EDITOR_ENGINE_LOG_API NauEngineLogger : public QObject
{
    Q_OBJECT
public:
    NauEngineLogger(const std::string& loggerName);

    template <typename S, typename... Args>
    void logMessage(NauEngineLogLevel level, const char* funcName,
        const char* fileName, unsigned line, S&& formatStr, Args&&... args);

    void clear();

    static void init(std::string sessionId);
    static std::string writableLocation(const std::string& loggerName);

signals:
    // Emitted when a message through this logger has been reported.
    // Note time here is a seconds from the epoch.
    void eventMessageReceived(int64_t time, NauEngineLogLevel level, const std::string& loggerName, const std::string& message);

private:
    void handleMessage(const nau::diag::LoggerMessage& msg);

private:
    inline static std::string SessionID = "unknown";

    nau::diag::Logger::Ptr                 m_logger;
    const std::string                      m_name;
    nau::diag::Logger::SubscriptionHandle  m_handler;
    nau::diag::Logger::SubscriptionHandle  m_handleFile;
};


// ** NauEngineEngineLogger
// 
// Underlying engine-wide logger.

class NAU_EDITOR_ENGINE_LOG_API NauEngineEngineLogger : public QObject
{
    Q_OBJECT
public:
    // Subscribes itself to global engine logger.
    void init();
    void terminate();

signals:
    // Emitted when a message through engine logger has been reported.
    // Note time here is a seconds from the epoch.
    void eventMessageReceived(int64_t time, NauEngineLogLevel level, const std::string& loggerName, const std::string& message);

private:
    void handleMessage(const nau::diag::LoggerMessage& msg);

    nau::diag::Logger::SubscriptionHandle m_handler;
    nau::diag::Logger::SubscriptionHandle m_handleFile;
};


template<typename S, typename ...Args>
inline void NauEngineLogger::logMessage(NauEngineLogLevel level, const char* funcName,
    const char* fileName, unsigned line, S&& formatStr, Args && ...args)
{
    const nau::diag::SourceInfo sourceInfo {m_name.data(), funcName, fileName, line };
    const auto engineLevel = static_cast<nau::diag::LogLevel>(level);

    if constexpr (sizeof...(Args) > 0)
    {
        auto message = nau::utils::format(formatStr, std::forward<Args>(args)...);
        m_logger->logMessage(engineLevel, {}, sourceInfo, std::move(message));
    }
    else
    {
        auto message = nau::utils::format("{}", formatStr);
        m_logger->logMessage(engineLevel, {}, sourceInfo, std::move(message));
    }
}
