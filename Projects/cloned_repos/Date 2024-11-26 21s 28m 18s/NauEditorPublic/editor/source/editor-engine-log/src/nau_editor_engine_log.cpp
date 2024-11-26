// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/nau_editor_engine_log.hpp"

#include <QStandardPaths>

#include "nau/diag/logging.h"
#include "nau/diag/log_subscribers.h"

// ** NauEngineLogger

NauEngineLogger::NauEngineLogger(const std::string& loggerName)
    : m_logger(std::move(nau::diag::createLogger()))
    , m_name(loggerName)
{
    m_handler = m_logger->subscribe(std::bind(&NauEngineLogger::handleMessage, this, std::placeholders::_1));
    m_handleFile = m_logger->subscribe(nau::diag::createFileOutputLogSubscriber({ 
        writableLocation(loggerName).c_str() }));
}

void NauEngineLogger::clear()
{
    if (m_handler) {
        m_handler.release();
    }
}

void NauEngineLogger::init(std::string sessionId)
{
    SessionID = sessionId;
}

std::string NauEngineLogger::writableLocation(const std::string& loggerName)
{
    static const auto appDataPath = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation).toStdString();
    return std::format("{}/{}/{}/{}", appDataPath, "Logs", SessionID, loggerName);
}

void NauEngineLogger::handleMessage(const nau::diag::LoggerMessage& msg)
{
    const auto editorLevel = static_cast<NauEngineLogLevel>(msg.level);

    emit eventMessageReceived(msg.time, editorLevel,
        std::string(msg.source.moduleName.data()), std::string(msg.data.c_str()));
}


// ** NauEngineEngineLogger

void NauEngineEngineLogger::init()
{
    auto& logger = nau::diag::getLogger();
    m_handler = logger.subscribe(
        std::bind(&NauEngineEngineLogger::handleMessage, this, std::placeholders::_1));
    m_handleFile = logger.subscribe(nau::diag::createFileOutputLogSubscriber({
        NauEngineLogger::writableLocation("Engine").c_str()}));
}

void NauEngineEngineLogger::terminate()
{
    if (m_handler) {
        m_handler.release();
    }
}

void NauEngineEngineLogger::handleMessage(const nau::diag::LoggerMessage& msg)
{
    static const std::string loggerName{"Engine"};
    const auto editorLevel = static_cast<NauEngineLogLevel>(msg.level);

    emit eventMessageReceived(msg.time, editorLevel, loggerName, std::string(msg.data.c_str()));
}
