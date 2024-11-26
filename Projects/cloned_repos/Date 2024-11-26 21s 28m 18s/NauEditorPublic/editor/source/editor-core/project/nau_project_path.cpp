// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_assert.hpp"
#include "nau_log.hpp"
#include "nau_project_path.hpp"
#include "nau_widget_utility.hpp"


// ** NauProjectPath

NauProjectPath::NauProjectPath(const std::string& path)
    : m_path(path)
{
}

NauProjectPath::NauProjectPath()
    : m_path("")
{
}

NauProjectPath& NauProjectPath::operator=(const NauProjectPath& other)
{
    m_path = other.m_path;
    return *this;
}

NauProjectPath& NauProjectPath::operator=(const NauProjectPath&& other)
{
    m_path = std::move(other.m_path);
    return *this;
}

NauProjectPath NauProjectPath::invalid()
{
    return NauProjectPath();
}

bool NauProjectPath::exists(const QString& base, const QString& name)
{
    NauFile projectFile(pathString(base, name));
    return projectFile.exists();
}

NauProjectPath NauProjectPath::create(const QString& base, const QString& name)
{
    NED_ASSERT(!base.isEmpty());
    NED_ASSERT(!name.isEmpty());
    QFileInfo baseInfo(base);
    NED_ASSERT(baseInfo.exists());

    // Check if project file already exists
    const auto pathText = pathString(base, name);
    const std::string pathString = pathText.toUtf8().constData();
    if (exists(base, name)) {
        NED_WARNING("Project at {} already exists!", pathString);
        return NauProjectPath(pathText.toUtf8().constData());
    }

    return NauProjectPath(pathText.toUtf8().constData());
}

QString NauProjectPath::userSettingsPathString() const
{
    return QString("%1.%2").arg(QString::fromUtf8(m_path.c_str())).arg(userSuffix);
}

QString NauProjectPath::pathString(const QString& base, const QString& name)
{
    return QString("%1/%2/%2.%3").arg(base).arg(name).arg(NauProjectPath::suffix);
}

NauDir NauProjectPath::root() const
{
    return QFileInfo(QString::fromUtf8(m_path)).dir();
}

bool NauProjectPath::isValid() const
{
    QFileInfo projectFile(QString::fromUtf8(m_path));
    return projectFile.exists() && projectFile.isFile() && projectFile.suffix() == suffix;
}

std::pair<bool, QString> NauProjectPath::createUserSettingsFileIfNeeded() const
{
    const QString path = userSettingsPathString();
    QFileInfo settingsFile(path);
    if (settingsFile.exists() && settingsFile.isFile()) return std::make_pair(true, path);  // Already exists

    // Create a new file
    NauFile userSettingsFile(path);
    if (!NED_PASSTHROUGH_ASSERT(userSettingsFile.open(QIODevice::WriteOnly))) {
        NED_ERROR("Failed to create user settings project path at {}", path);
        return std::make_pair(false, QString());
    }

    // Success
    userSettingsFile.close();
    return std::make_pair(true, path);
}

bool NauProjectPath::generate() const
{
    // Make project directory
    NauDir dir;
    if (!NED_PASSTHROUGH_ASSERT(dir.mkpath(root().path()))) {
        NED_ERROR("Failed to create project directory at {}", root().path().toUtf8().constData());
        return false;
    }

    // Make main project file
    NauFile projectFile(QString::fromStdString(m_path));
    if (!NED_PASSTHROUGH_ASSERT(projectFile.open(QIODevice::WriteOnly))) {
        NED_ERROR("Failed to create project path at {}", m_path);
        return false;
    }

    projectFile.close();
    return true;
}
