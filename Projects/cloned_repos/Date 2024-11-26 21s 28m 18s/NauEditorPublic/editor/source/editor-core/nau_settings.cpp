// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_settings.hpp"
#include "nau_assert.hpp"
#include "nau_log.hpp"
#include "commands/nau_commands.hpp"

#include "magic_enum/magic_enum.hpp"

#include <QSettings>
#include <QStandardPaths>


// ** NauSettings

void NauSettings::addRecentProjectPath(const NauProjectPath& path)
{
    QSettings settings;
    tryAndRemoveRecentProjectPath(path);  // If the project is already in the list it needs to be removed first
    QStringList pathsRaw = recentProjectPathsRaw();
    pathsRaw.append(path);
    saveRecentProjectPathsRaw(pathsRaw);
}

bool NauSettings::tryAndRemoveRecentProjectPath(const NauProjectPath& path)
{
    QStringList pathsRaw = recentProjectPathsRaw();
    const bool result = pathsRaw.removeIf([&path](QString item) {
        return item == path;
    }) > 0;
    saveRecentProjectPathsRaw(pathsRaw);
    return result;
}

NauProjectPathList NauSettings::recentProjectPaths()
{
    QSettings settings;
    NauProjectPathList result;
    const auto recentProjects = recentProjectPathsRaw();
    for (const auto& rawPath : recentProjects) {
        const auto projectPath = NauProjectPath(std::string(rawPath.toUtf8()));
        if (!projectPath.isValid()) {  // Might have been moved or deleted
            // Only a warning, we leave it up to the user to clear invalid projects
            NED_WARNING("Project at {} doesn't exist anymore", static_cast<std::string>(projectPath));
        }
        result.append(std::string(rawPath.toUtf8()));
    }
    return result;
}

void NauSettings::setRecentProjectDirectory(const NauDir& directory)
{
    QSettings settings;
    NED_ASSERT(directory.exists());
    settings.setValue(settingName(Setting::DefaultProjectLocation), directory.absolutePath());
}

NauDir NauSettings::recentProjectDirectory()
{
    QSettings settings;
    const QString defaultPath = defaultValue(Setting::DefaultProjectLocation).toString();
    const auto directory = NauDir(settings.value(settingName(Setting::DefaultProjectLocation), defaultPath).toString());
    if (!directory.exists()) {  // Reset invalid paths to default
        settings.remove(settingName(Setting::DefaultProjectLocation));
        return NauDir(defaultPath);
    }
    return directory;
}

void NauSettings::setRecentLauncherOutputDirectory(const NauDir& directory)
{
    QSettings settings;
    NED_ASSERT(directory.exists());
    settings.setValue(settingName(Setting::RecentLauncherOutputDirectory), directory.absolutePath());
}

NauDir NauSettings::recentLauncherOutputDirectory()
{
    QSettings settings;
    const NauDir defaultDir = NauDir(defaultValue(Setting::RecentLauncherOutputDirectory).toString());
    const QVariant dirData = settings.value(settingName(Setting::RecentLauncherOutputDirectory));
    if (!dirData.isValid()) {
        return defaultDir;
    }

    const NauDir directory = dirData.toString();
    if (!directory.exists()) {
        return defaultDir;
    }

    return directory;
}

// Currently unused
void NauSettings::setUndoRedoStackSize(size_t size)
{
    QSettings settings;
    settings.setValue(settingName(Setting::UndoRedoStackSize), size);
}

size_t NauSettings::undoRedoStackSize()
{
    QSettings settings;
    return static_cast<size_t>(settings.value(settingName(Setting::UndoRedoStackSize), defaultValue(Setting::RecentLauncherOutputDirectory)).toUInt());
}

void NauSettings::setDeviceId(const QString& deviceId)
{
    QSettings settings;
    settings.setValue(settingName(Setting::DeviceId), deviceId);
}

QString NauSettings::deviceId()
{
    QSettings settings;
    return static_cast<QString>(settings.value(settingName(Setting::DeviceId), defaultValue(Setting::DeviceId)).toString());
}

void NauSettings::setEditorLanguage(const QString& lang)
{
    QSettings settings;
    settings.setValue(settingName(Setting::Language), lang);
}

QString NauSettings::editorLanguage()
{
    QSettings settings;
    return settings.value(settingName(Setting::Language), defaultValue(Setting::Language)).toString();

}

#ifdef NAU_UNIT_TESTS
void NauSettings::clearRecentProjectPaths()
{
    removeSetting(NauSettings::Setting::RecentProjects);
}

NauDir NauSettings::defaultProjectDirectory()
{
    return defaultValue(Setting::DefaultProjectLocation).toString();
}

void NauSettings::clearRecentProjectDirectory()
{
    removeSetting(Setting::DefaultProjectLocation);
}

NauDir NauSettings::defaultLauncherOutputDirectory()
{
    return defaultValue(Setting::RecentLauncherOutputDirectory).toString();
}

void NauSettings::clearRecentLauncherOutputDirectory()
{
    removeSetting(Setting::RecentLauncherOutputDirectory);
}
#endif

QStringList NauSettings::recentProjectPathsRaw()
{
    QSettings settings;
    return settings.value(settingName(Setting::RecentProjects), defaultValue(Setting::RecentProjects)).toStringList();
}

void NauSettings::saveRecentProjectPathsRaw(QStringList& paths)
{
    QSettings settings;
    settings.setValue(settingName(Setting::RecentProjects), paths);
}

QString NauSettings::settingName(Setting key)
{
    static QHash<Setting, QString> settingNameByKey = {
        { Setting::RecentLauncherOutputDirectory, QStringLiteral("Launcher/RecentOutputDirectory") },
        { Setting::RecentProjects,                QStringLiteral("Editor/RecentProjects")          },
        { Setting::DefaultProjectLocation,        QStringLiteral("Editor/DefaultProjectLocation")  },
        { Setting::UndoRedoStackSize,             QStringLiteral("Editor/UndoRedoStackSize")       },
        { Setting::DeviceId,                      QStringLiteral("Editor/DeviceId")                },
        { Setting::Language,                      QStringLiteral("Editor/Language")                },
    };

    const auto settingNameIt = settingNameByKey.constFind(key);
    if (settingNameIt != settingNameByKey.constEnd()) {
        return settingNameIt.value();
    } else {
        NED_ERROR("Not implemented setting name for NauSettings::Setting::{}", magic_enum::enum_name(key));
        NED_ASSERT(!"Not implemented setting name for NauSettings::Setting");
        return QString();
    }
}

QVariant NauSettings::defaultValue(Setting key)
{
    static QHash<Setting, QVariant> defaultValueByKey = {
        { Setting::RecentLauncherOutputDirectory, NauDir(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)).absoluteFilePath(QStringLiteral("gameClean")) },
        { Setting::RecentProjects,                QStringList()                                                                                                             },
        { Setting::DefaultProjectLocation,        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)                                                       },
        { Setting::UndoRedoStackSize,             NauCommandStack::DefaultStackSize                                                                                         },
        { Setting::DeviceId,                      QString()                                                                                                                 },
        { Setting::Language,                      QStringLiteral("en")                                                                                                      }
    };

    const auto defaultValueIt = defaultValueByKey.constFind(key);
    if (defaultValueIt != defaultValueByKey.constEnd()) {
        return defaultValueIt.value();
    } else {
        NED_ERROR("Not implemented default value for NauSettings::Setting::{}", magic_enum::enum_name(key));
        NED_ASSERT(!"Not implemented default value for NauSettings::Setting");
        return QVariant();
    }
}

void NauSettings::removeSetting(Setting key)
{
    QSettings settings;
    settings.remove(settingName(key));
}
