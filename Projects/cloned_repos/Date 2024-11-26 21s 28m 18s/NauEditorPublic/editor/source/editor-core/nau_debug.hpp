// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// This file will store everything related to the debugging of the editor

#pragma once

#include "nau/nau_editor_config.hpp"

namespace Nau
{
    // ** DebugTools
    //
    // Namespace for storing tools for debugging the editor 

    namespace DebugTools
    {
        void NAU_EDITOR_API QtMessageHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg);
    };
};
