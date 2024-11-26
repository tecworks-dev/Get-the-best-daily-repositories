// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Script managing classes

#pragma once

#include "nau/nau_delegate.hpp"

#include <QString>
#include <QFileSystemWatcher>


// ** NauScriptManager
// 
// Provides script hotreload functionality. Activates scripts hotreload automaticly by editorModeChanged event

class NauScriptManager : public QObject
{
    Q_OBJECT
public:
    NauScriptManager();
    ~NauScriptManager();

private:
    void startWatch();
    void stopWatch();

    void reloadScript(const QString& path);

private:
    QFileSystemWatcher m_scriptsWatcher;
    NauCallbackId m_scriptChangesWatchCbId;
};