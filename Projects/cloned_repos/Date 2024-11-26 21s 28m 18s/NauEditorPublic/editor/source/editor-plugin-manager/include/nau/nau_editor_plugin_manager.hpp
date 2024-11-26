// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor modules loading & service provider functions

#pragma once

#include "nau/nau_editor_plugin_manager_config.hpp"
#include "nau/service/service_provider.h"

#include <string>


// ** Editor modules functions

namespace Nau::EditorModules
{
   NAU_EDITOR_PLUGIN_MANAGER_API void LoadEditorModules(const std::string& modulesListStr);
}


// ** Editor service provider

namespace Nau
{

    NAU_EDITOR_PLUGIN_MANAGER_API void SetDefaultEditorServiceProvider(nau::ServiceProvider::Ptr&&);

    NAU_EDITOR_PLUGIN_MANAGER_API bool HasEditorServiceProvider();

    NAU_EDITOR_PLUGIN_MANAGER_API nau::ServiceProvider& EditorServiceProvider();
}
