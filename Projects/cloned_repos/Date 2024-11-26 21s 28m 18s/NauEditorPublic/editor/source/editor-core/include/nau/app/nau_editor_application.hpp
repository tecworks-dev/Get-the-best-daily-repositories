// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor application class

#pragma once

#include "nau/nau_editor_config.hpp"
#include "nau/app/nau_qt_app.hpp"
#include "nau/app/nau_editor_window_interface.hpp"

#include "project/nau_project.hpp"

#include "nau/diag/logging.h"

#include <QCoreApplication>


// ** NauEditorApplication

class NAU_EDITOR_API NauEditorApplication
{
    Q_DECLARE_TR_FUNCTIONS(NauEditorApplication)

public:
    NauEditorApplication() = default;
    ~NauEditorApplication();

    bool initialize(int argc, char* argv[]);
    int execute();

    NauEditorApplication(NauEditorApplication const&) = delete;
    NauEditorApplication& operator=(NauEditorApplication const&) = delete;
    NauEditorApplication(NauEditorApplication&&) = delete;
    NauEditorApplication& operator=(NauEditorApplication&&) = delete;

private:
    NauProjectPtr loadProject();

    bool initQtApp(int argc, char* argv[]);
    bool initEditor(NauProjectPtr project);

    bool startupServices();
    bool shutdownServices();

private:
    bool m_initialized = false;
    nau::diag::Logger::SubscriptionHandle m_loggerHandle;
    std::unique_ptr<NauApp> m_app;
};
