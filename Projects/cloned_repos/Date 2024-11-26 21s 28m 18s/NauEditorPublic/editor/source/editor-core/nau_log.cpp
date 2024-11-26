// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_log.hpp"

#include "nau_widget.hpp"
#include "log/nau_log_constants.hpp"

#include <memory>

#include <QStandardPaths>

#include "nau/diag/logging.h"
#include "nau/diag/log_subscribers.h"
#include "nau/app/nau_qt_app.hpp"

// ** NauLog

NauEngineEngineLogger NauLog::m_loggerEngine;
NauLoggerPtr NauLog::m_loggerEditor;
NauLoggerPtr NauLog::m_loggerBuild;
NauLogModel NauLog::m_editorLogModel;

void NauLog::init()
{
    NauEngineLogger::init(NauApp::sessionID());

    m_loggerEditor = std::make_shared<NauEngineLogger>(NauLogConstants::editorSourceName().toUtf8().constData());
    m_loggerBuild = std::make_shared<NauEngineLogger>(NauLogConstants::buildSourceName().toUtf8().constData());

    QObject::connect(m_loggerEditor.get(), &NauEngineLogger::eventMessageReceived, 
        &m_editorLogModel, &NauLogModel::addEntry);

    QObject::connect(m_loggerBuild.get(), &NauEngineLogger::eventMessageReceived,
        &m_editorLogModel, &NauLogModel::addEntry);
}

void NauLog::initEngineLog()
{
    m_loggerEngine.init();

    // Process engine logger messages only in our thread.
    QObject::connect(&m_loggerEngine, &NauEngineEngineLogger::eventMessageReceived, 
        &m_editorLogModel, &NauLogModel::addEntry, Qt::QueuedConnection);
}

void NauLog::close()
{
    QObject::disconnect(m_loggerEditor.get(), &NauEngineLogger::eventMessageReceived, 
        &m_editorLogModel, &NauLogModel::addEntry);

    QObject::disconnect(m_loggerBuild.get(), &NauEngineLogger::eventMessageReceived, 
        &m_editorLogModel, &NauLogModel::addEntry);

    QObject::disconnect(&m_loggerEngine, &NauEngineEngineLogger::eventMessageReceived, 
        &m_editorLogModel, &NauLogModel::addEntry);

    m_loggerEditor.reset();
    m_loggerBuild.reset();
    m_loggerEngine.terminate();
}

NauLoggerRef NauLog::editorLogger()
{
    return *m_loggerEditor;
}

NauLoggerRef NauLog::buildLogger()
{
    return *m_loggerBuild;
}

NauLogModel& NauLog::editorLogModel()
{
    return m_editorLogModel;
}
