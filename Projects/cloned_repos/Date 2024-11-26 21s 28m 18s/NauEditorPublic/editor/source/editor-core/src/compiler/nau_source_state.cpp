// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/compiler/nau_source_state.hpp"
#include "nau_log.hpp"
#include "magic_enum/magic_enum.hpp"

#include <QCoreApplication>
#include <QFileInfo>


namespace Nau
{
    bool fillCompilationStateInfo(const NauSourceStateManifold& state, QString& briefInfo, QString& detailedInfo)
    {
        switch (state.state)
        {
        case NauSourceState::NoBuildTools:
            briefInfo = QCoreApplication::translate("CompilationState", "No Build Tools");
            detailedInfo = QCoreApplication::translate("CompilationState", "Unable to found required tools to build the project sources. "
                "Please follow <a style=\"color:yellow\" href=\"https://nauengine.org/\">the documentation</a> to download and install the build tools.");
            return true;
        case NauSourceState::RecompilationRequired:
            briefInfo = QCoreApplication::translate("CompilationState", "Compilation required");
            detailedInfo = QCoreApplication::translate("CompilationState", "Scripts have been changed, and should be recompiled. "
                "Please restart the editor to compile and reload project modules");
            return true;
        case NauSourceState::CompilationError:
            briefInfo = QCoreApplication::translate("CompilationState", "Compilation failure");
            detailedInfo = QCoreApplication::translate("CompilationState", "Failed to compile or building. <br/>You can find compilation log in "
                "<a  style=\"color:yellow\" href=\"file:///%1\">%2</a>. <br/>Try to restart the editor").arg(state.buildLogFileName).arg(QFileInfo(state.buildLogFileName).fileName());
            return true;
        case NauSourceState::FatalError:
            briefInfo = QCoreApplication::translate("CompilationState", "Fatal error");
            detailedInfo = QCoreApplication::translate("CompilationState", "Unexpected fatal error occurred.</b> Try to restart the editor. "
                "More information in log files and in the console window");
            return true;
        case NauSourceState::Success:
            briefInfo = QCoreApplication::translate("CompilationState", "Code is up to date");
            detailedInfo = QCoreApplication::translate("CompilationState", "Project scripts were successfully built and loaded");
            return true;
        }

        NED_WARNING("Not implemented compilation state {}", magic_enum::enum_name<NauSourceState>(state.state));

        briefInfo = QCoreApplication::translate("CompilationState", "Unknown state");
        detailedInfo = QCoreApplication::translate("CompilationState", "");
        return false;
    }
}