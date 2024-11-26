// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Used to store user settings

#pragma once

#include "project/nau_project.hpp"
#include "project/nau_project_path.hpp"
#include "baseWidgets/nau_widget_utility.hpp"


// ** NauSettings

class NauSettings
{
public:
    static void addRecentProjectPath(const NauProjectPath& path);
    static bool tryAndRemoveRecentProjectPath(const NauProjectPath& path);
    static NauProjectPathList recentProjectPaths();

    static void setRecentProjectDirectory(const NauDir& directory);
    static NauDir recentProjectDirectory();

    static void setRecentLauncherOutputDirectory(const NauDir& directory);
    static NauDir recentLauncherOutputDirectory();

    static void setUndoRedoStackSize(size_t size);
    static size_t undoRedoStackSize();

    static void setDeviceId(const QString& deviceId);
    static QString deviceId();

    static void setEditorLanguage(const QString& lang);
    static QString editorLanguage();

#ifdef NAU_UNIT_TESTS
    static NauDir defaultLauncherOutputDirectory();
    static NauDir defaultProjectDirectory();
    static void clearRecentProjectPaths();
    static void clearRecentProjectDirectory();
    static void clearRecentLauncherOutputDirectory();
#endif

private:
    enum class Setting
    {
        RecentLauncherOutputDirectory,
        RecentProjects,
        DefaultProjectLocation,
        UndoRedoStackSize,
        DeviceId,
        Language,
    };

private:
    static QStringList recentProjectPathsRaw();
    static void saveRecentProjectPathsRaw(QStringList& paths);
    static QString settingName(Setting key);
    static QVariant defaultValue(Setting key);
    static void removeSetting(Setting key);
};
