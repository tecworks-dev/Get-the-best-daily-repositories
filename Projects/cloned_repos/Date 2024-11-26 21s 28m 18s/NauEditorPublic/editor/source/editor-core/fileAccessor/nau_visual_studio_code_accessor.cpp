// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_visual_studio_code_accessor.hpp"
#include "nau_log.hpp"
#include "nau_widget_utility.hpp"

#include <QSettings>
#include <QProcess>
#include <QRegularExpression>
#include <QFileInfo>


// ** NauVisualStudioCodeAccessor

bool NauVisualStudioCodeAccessor::init()
{
    const QSettings vsCodeRegPath(R"(HKEY_CURRENT_USER\SOFTWARE\Classes\Applications\Code.exe\shell\open\command)",
        QSettings::NativeFormat);
    const auto keys = vsCodeRegPath.allKeys();
    if (keys.isEmpty()) {
        NED_ERROR("Failed to register source code accessor! VS Code not installed.");
        return false;
    }

    const QString vsCodeRegPathStr = vsCodeRegPath.value(keys[0]).toString();

    static const QRegularExpression regex("\"(.*)\" \".*\"");
    const QRegularExpressionMatch match = regex.match(vsCodeRegPathStr);
    if (match.hasMatch()) {
        m_IDEPath = match.captured(1);
    }

    bool isIDEExist = QFileInfo::exists(m_IDEPath);
    if (!isIDEExist) {
        NED_ERROR("Failed to register source code accessor! VS Code exe is missing.");
    }

    return isIDEExist;
}

bool NauVisualStudioCodeAccessor::openFile(const QString& path)
{   
    NauProcess vsCodeProc;

    vsCodeProc.setProgram(m_IDEPath);
    vsCodeProc.setArguments({ path });
    
    return vsCodeProc.startDetached();
}