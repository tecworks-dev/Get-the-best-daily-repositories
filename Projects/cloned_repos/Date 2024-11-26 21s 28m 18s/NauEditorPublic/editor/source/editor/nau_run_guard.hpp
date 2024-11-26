// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Used to allow only a single instance of the app to run at once

#pragma once

#include <memory>


// ** NauRunGuard

class NauRunGuardPrivate;
class NauRunGuard
{
public:
    NauRunGuard();
    ~NauRunGuard();

    bool tryToAcquire();

private:
    std::unique_ptr<NauRunGuardPrivate> m_impl;
};


namespace Nau
{
    // ** ShowAlreadyRunningWarning
    //
    // Shows a warning that another instance of the app is already running

    int ShowAlreadyRunningWarning(int argc, char* argv[]);
}
