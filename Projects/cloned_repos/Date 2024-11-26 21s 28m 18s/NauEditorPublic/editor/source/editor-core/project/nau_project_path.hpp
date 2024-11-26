// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Project path

#pragma once

#include "nau/nau_editor_config.hpp"

#include <QObject>


// ** NauProjectPath

class NauDir;
class NauProjectPath;
using NauProjectPathList = QList<NauProjectPath>;

class NAU_EDITOR_API NauProjectPath
{
public:
    NauProjectPath(const std::string& path);
    NauProjectPath(const NauProjectPath&) = default;
    NauProjectPath& operator=(const NauProjectPath& other);
    NauProjectPath& operator=(const NauProjectPath&& other);
    
    static bool exists(const QString& base, const QString& name);
    static NauProjectPath create(const QString& base, const QString& name);
    static NauProjectPath invalid();

    // It's okay for a user to delete user settings file, so we need to be able to create it again
    std::pair<bool, QString> createUserSettingsFileIfNeeded() const;
    bool generate() const;

    NauDir root() const;
    bool isValid() const;

    // Custom conversion
    operator std::string() const { return m_path; }
    operator QString() const { return QString::fromUtf8(m_path.c_str()); }

    bool operator==(const NauProjectPath& other) const { return m_path == other.m_path; }

public:
    inline static const QString suffix = "nauproject";
    inline static const QString userSuffix = "user";

private:
    NauProjectPath();  // Used to build invalid path

    static QString pathString(const QString& base, const QString& name);
    QString userSettingsPathString() const;

private:
    std::string m_path;
};
