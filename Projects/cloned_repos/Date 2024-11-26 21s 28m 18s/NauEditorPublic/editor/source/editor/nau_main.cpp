// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/app/nau_editor_application.hpp"
#include "nau_run_guard.hpp"

int main(int argc, char* argv[])
{
    // Don't allow to run multiple instances of the app at the same time
    NauRunGuard guard;
    if (!guard.tryToAcquire()) {
        // Put back on once we know how to properly shut down the engine:
        // Show a dialog notifying the user that another instance of the app is already running
        // and offer to terminate the other instance
        // return Nau::ShowAlreadyRunningWarning(argc, argv);
        return false;
    }

    NauEditorApplication application;

    if (!application.initialize(argc, argv)) {
        return 1;
    }

    const auto result = application.execute();

    return result;
}
