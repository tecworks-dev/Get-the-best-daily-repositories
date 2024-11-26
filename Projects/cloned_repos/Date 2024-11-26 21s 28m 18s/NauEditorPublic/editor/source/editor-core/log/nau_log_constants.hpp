// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Constants used in logger/console widgets/blocks.

#pragma once

#include "nau_log.hpp"

#include <QCoreApplication>
#include <QString>
#include <unordered_map>


// ** NauLogConstants

class NauLogConstants
{
    Q_DECLARE_TR_FUNCTIONS(NauLogConstants)
public:
    static QString dateTimeDisplayFormat();

    static const std::unordered_map<NauLogLevel, QString>& levelByNameMap();

    static QString consoleTabName();

    static QString editorSourceName();
    static QString engineSourceName();
    static QString playModeSourceName();
    static QString buildSourceName();
    static QString externalSourceName();
    static QString importSourceName();

    static int bottomInfoPanelHeight();
    static int detailsHeaderPanelHeight();
    static int settingsButtonHeight();
    static int messageItemHeight();
    static int sourceItemHeight();
};
