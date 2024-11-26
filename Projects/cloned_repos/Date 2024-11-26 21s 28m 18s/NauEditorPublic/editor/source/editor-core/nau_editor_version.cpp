// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_editor_version.hpp"
#include "nau_widget_utility.hpp"
#include "nau_log.hpp"
#include "nau_assert.hpp"

#include <QCoreApplication>
#include <QStringList>
#include <QRegularExpression>


// ** NauEditorVersion

NauEditorVersion::NauEditorVersion(int major, int minor, int patch)
    : m_major(major)
    , m_minor(minor)
    , m_patch(patch)
{
}

NauEditorVersion NauEditorVersion::current()
{
    NauFile versionFile(QString(":/NauEditor/%1").arg(m_file.c_str()));

    // Open version file
    if (!versionFile.open(QIODevice::ReadOnly)) {
        NED_WARNING("Couldn't open {}", versionFile.fileName());
        return invalid();
    }

    // Read version as string
    const QString versionText = versionFile.readLine();
    if (versionText.isEmpty()) {
        NED_WARNING("Couldn't read {} file contents", versionFile.fileName());
        return invalid();
    }

    // Parse the version string
    const auto versionParts = versionText.split('.');
    if (versionParts.size() != 3) {
        NED_WARNING("Invalid version format: {}", versionText);
        return invalid();
    }

    return NauEditorVersion(
        versionParts.at(0).toInt(), 
        versionParts.at(1).toInt(), 
        versionParts.at(2).toInt()
    );
}

NauEditorVersion NauEditorVersion::invalid()
{
    return NauEditorVersion(0, 0, 0);
}

NauEditorVersion NauEditorVersion::fromText(const QString& version)
{
    const QStringList versionParts = version.split(QRegularExpression("[. ]"), Qt::SkipEmptyParts);
    if (!NED_PASSTHROUGH_ASSERT(versionParts.size() == 3 || versionParts.size() == 4)) {
        return NauEditorVersion::invalid();
    }

    bool success = false;
    const int major = versionParts[0].toInt(&success);
    const int minor = versionParts[1].toInt(&success);
    const int patch = versionParts[2].toInt(&success);
    if (!NED_PASSTHROUGH_ASSERT(success)) {
        return NauEditorVersion::invalid();
    }

    return NauEditorVersion(major, minor, patch);
}

std::string NauEditorVersion::asString() const
{
    return std::format("{}.{}.{} {}", m_major, m_minor, m_patch, m_stage);
}

QString NauEditorVersion::asQtString() const
{
    return QString::fromUtf8(asString().c_str());
}

bool NauEditorVersion::isValid() const
{
    return *this != NauEditorVersion::invalid();
}
