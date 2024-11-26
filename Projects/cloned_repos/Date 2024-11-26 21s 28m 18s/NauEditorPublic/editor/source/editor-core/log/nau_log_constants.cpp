// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_log_constants.hpp"

QString NauLogConstants::dateTimeDisplayFormat()
{
    return QStringLiteral("yyyy-MM-dd hh:mm:ss");
}

const std::unordered_map<NauLogLevel, QString>& NauLogConstants::levelByNameMap()
{
    static const std::unordered_map<NauLogLevel, QString> map = {
        { NauLogLevel::Debug, tr("Debug") },
        { NauLogLevel::Info, tr("Info") },
        { NauLogLevel::Warning, tr("Warning") },
        { NauLogLevel::Error, tr("Error") },
        { NauLogLevel::Critical, tr("Critical") },
        { NauLogLevel::Verbose, tr("Trace") },
    };

    return map;
}

QString NauLogConstants::consoleTabName()
{
    return QStringLiteral("Console");
}

QString NauLogConstants::editorSourceName()
{
    return QStringLiteral("Editor");
}

QString NauLogConstants::engineSourceName()
{
    return QStringLiteral("Engine");
}

QString NauLogConstants::playModeSourceName()
{
    return QStringLiteral("PlayMode");
}

QString NauLogConstants::buildSourceName()
{
    return QStringLiteral("Build");
}

QString NauLogConstants::externalSourceName()
{
    return QStringLiteral("External");
}

QString NauLogConstants::importSourceName()
{
    return QStringLiteral("Import");
}

int NauLogConstants::bottomInfoPanelHeight()
{
    return 32;
}

int NauLogConstants::detailsHeaderPanelHeight()
{
    return 48;
}

int NauLogConstants::settingsButtonHeight()
{
    return 32;
}

int NauLogConstants::messageItemHeight()
{
    return 32;
}

int NauLogConstants::sourceItemHeight()
{
    return 40;
}
