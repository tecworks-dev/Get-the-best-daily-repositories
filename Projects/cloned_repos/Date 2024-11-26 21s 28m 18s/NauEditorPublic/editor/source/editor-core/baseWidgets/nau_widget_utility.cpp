// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_widget_utility.hpp"


// ** NauShortcut

NauKeySequence::NauKeySequence(QKeyCombination keyCombination)
    : QKeySequence(keyCombination)
{
}

NauKeySequence::NauKeySequence(StandardKey key)
    : QKeySequence(key)
{
}

NauShortcut::NauShortcut(const NauKeySequence& key, QObject* parent)
    : QShortcut(key, parent)
{
}

NauShortcut::NauShortcut(QKeySequence::StandardKey key, QObject* parent)
    : QShortcut(key, parent)
{
}


// ** NauDir

NauDir::NauDir(const QDir& dir)
    : QDir(dir)
{
}

NauDir::NauDir(const QString& path)
    : QDir(path)
{
}

NauDir::NauDir(const std::filesystem::path& path)
    : QDir(path)
{
}


// ** NauFile

NauFile::NauFile(const QString& name)
    : QFile(name)
{
}
