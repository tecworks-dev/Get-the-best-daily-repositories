// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Everything related to the version of the editor.

#pragma once

#include <string>
#include <QString>


// ** NauEditorVersion

class NauEditorVersion
{
public:
    static NauEditorVersion current();
    static NauEditorVersion invalid();
    static NauEditorVersion fromText(const QString& version);

public:
    NauEditorVersion(int major, int minor, int patch);

    std::string asString() const;
    QString asQtString() const;
    bool isValid() const;

    bool operator==(const NauEditorVersion& other) const = default;
    inline auto operator<=>(const NauEditorVersion& other) const
    {
        return std::tie(m_major, m_minor, m_patch) <=> std::tie(other.m_major, other.m_minor, other.m_patch);
    }

private:
    int m_major;
    int m_minor;
    int m_patch;

    #ifdef NAU_UNIT_TESTS
    inline static const std::string m_stage  = "test";
    inline static const std::string m_file   = "test_version.txt";
    #else
    inline static const std::string m_stage  = "beta";
    inline static const std::string m_file   = "version.txt";
    #endif
};
