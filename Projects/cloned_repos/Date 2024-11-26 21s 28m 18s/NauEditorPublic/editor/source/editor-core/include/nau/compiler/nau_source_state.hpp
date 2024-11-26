// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include <QString>


// ** NauSourceState
// 
// States of project scripts building.

enum class NauSourceState
{
    // Can't locate a functional build tools.
    NoBuildTools,

    // Scripts has been changed and modules must be recompiled.
    RecompilationRequired,

    // An error occurred since last compilation/build.
    CompilationError,

    // A system error related editor/engine work logic occurred.
    FatalError,

    // All sources and binaries is up to date.
    Success,
};


// ** NauSourceStateManifold
// A descriptor of the state of a compilation of the user scripts.
struct NauSourceStateManifold
{
    NauSourceState state;
    QString buildLogFileName;
    QString additionalMessage;
};


namespace Nau
{
    // Fill user-friendly info about compilation state.
    bool fillCompilationStateInfo(const NauSourceStateManifold& state,
        QString& briefInfo, QString& detailedInfo);
} 