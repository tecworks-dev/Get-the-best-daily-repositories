// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Set of auxiliary Nau wrappers of base Qt classes.

#pragma once

#include "nau/nau_editor_config.hpp"

#include <QShortcut>
#include <QProcess>
#include <QDir>
#include <QFile>


// ** NauKeySequence
//
// The class in which key sequences will be stored in the future

class NAU_EDITOR_API NauKeySequence : public QKeySequence
{
public:
    NauKeySequence() = default;
    NauKeySequence(QKeyCombination keyCombination);
    NauKeySequence(StandardKey key);
};


// ** NauShortcut
//
// The class in which shortcuts will be stored in the future

class NAU_EDITOR_API NauShortcut : public QShortcut
{
public:
    NauShortcut() = default;
    NauShortcut(const NauKeySequence& key, QObject* parent);
    NauShortcut(QKeySequence::StandardKey key, QObject* parent);
};


// ** NauProcess
//
// Responsible for work and communication with external processes

class NAU_EDITOR_API NauProcess : public QProcess
{
public:
    NauProcess() = default;
};


// ** NauDir
//
// Allows work with paths

class NAU_EDITOR_API NauDir : public QDir
{
public:
    NauDir(const QDir& dir);
    NauDir(const QString& path = QString());
    NauDir(const std::filesystem::path &path);

};


// ** NauFile
//
// Allows manage the file

class NAU_EDITOR_API NauFile : public QFile
{
public:
    NauFile(const QString& name);
};
