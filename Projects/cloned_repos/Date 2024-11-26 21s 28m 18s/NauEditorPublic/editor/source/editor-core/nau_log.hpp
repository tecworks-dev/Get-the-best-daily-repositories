// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Base logging

#pragma once

#include "baseWidgets/nau_widget_utility.hpp"

#ifndef NAU_UNIT_TESTS

#include "log/nau_log_model.hpp"
#include "nau/nau_editor_engine_log.hpp"

#include <memory>

// ** NauLog

using NauLogger = NauEngineLogger;
using NauLoggerRef = NauEngineLogger&;
using NauLoggerPtr = std::shared_ptr<NauEngineLogger>;
using NauLogLevel = NauEngineLogLevel;

class NAU_EDITOR_API NauLog
{
public:
    NauLog() = delete;

    static void init();
    static void initEngineLog();
    static void close();

    static NauLoggerRef editorLogger();
    static NauLoggerRef buildLogger();

    static NauLogModel& editorLogModel();

private:
    static NauLogModel  m_editorLogModel;
    static NauEngineEngineLogger m_loggerEngine;
    static NauLoggerPtr m_loggerEditor;
    static NauLoggerPtr m_loggerBuild;
};


// Shortcuts for logging

#define NED_CRITICAL(...) NauLog::editorLogger().logMessage(NauEngineLogLevel::Critical, __FUNCTION__, __FILE__, static_cast<unsigned>(__LINE__), __VA_ARGS__)
#define NED_ERROR(...)    NauLog::editorLogger().logMessage(NauEngineLogLevel::Error, __FUNCTION__, __FILE__, static_cast<unsigned>(__LINE__), __VA_ARGS__)
#define NED_WARNING(...)  NauLog::editorLogger().logMessage(NauEngineLogLevel::Warning, __FUNCTION__, __FILE__, static_cast<unsigned>(__LINE__), __VA_ARGS__)
#ifndef NDEBUG
#define NED_DEBUG(...)    NauLog::editorLogger().logMessage(NauEngineLogLevel::Debug, __FUNCTION__, __FILE__, static_cast<unsigned>(__LINE__), __VA_ARGS__)
#else
#define NED_DEBUG(...)    {(void)(__VA_ARGS__);}  // Debug messages only show up in debug builds
#endif
#define NED_INFO(...)     NauLog::editorLogger().logMessage(NauEngineLogLevel::Info, __FUNCTION__, __FILE__, static_cast<unsigned>(__LINE__), __VA_ARGS__)
#define NED_TRACE(...)    NauLog::editorLogger().logMessage(NauEngineLogLevel::Verbose, __FUNCTION__, __FILE__, static_cast<unsigned>(__LINE__), __VA_ARGS__)

template<>
struct fmt::formatter<QString, char> : fmt::formatter<const char*, char>
{
    template <typename FormatContext>
    auto format(const QString& input, FormatContext& ctx) const -> decltype(ctx.out()) 
    {
        if (!input.startsWith("'") && !input.startsWith("\"")) {
            return fmt::format_to(ctx.out(), "'{}'", input.toUtf8().constData());
        }

        return fmt::format_to(ctx.out(), "{}", input.toUtf8().constData());
    }
};

#else
#define NED_CRITICAL(...)  {}
#define NED_ERROR(...)     {}
#define NED_WARNING(...)   {}
#ifndef NDEBUG
#define NED_DEBUG(...)     {}
#else
#define NED_DEBUG
#endif
#define NED_TRACE(...)     {}
#endif // !
