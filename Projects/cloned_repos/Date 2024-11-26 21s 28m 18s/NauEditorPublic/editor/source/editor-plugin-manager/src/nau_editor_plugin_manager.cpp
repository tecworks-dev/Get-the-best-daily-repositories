// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/nau_editor_plugin_manager.hpp"
#include "nau/module/module_manager.h"


// ** Editor modules functions

namespace Nau::EditorModules
{
    void LoadEditorModules(const std::string& modulesListStr)
    {
        nau::loadModulesList(modulesListStr.c_str());
    }
}


// ** Editor service provider

namespace Nau
{
    namespace
    {
        nau::ServiceProvider::Ptr& EditorServiceProviderInstanceRef()
        {
            static nau::ServiceProvider::Ptr s_serviceProvider;
            return (s_serviceProvider);
        }
    }  // namespace

    void SetDefaultEditorServiceProvider(nau::ServiceProvider::Ptr&& provider)
    {
        NAU_FATAL(!provider || !EditorServiceProviderInstanceRef(), "Editor service provider already set");
        EditorServiceProviderInstanceRef() = std::move(provider);
    }

    bool HasEditorServiceProvider()
    {
        return static_cast<bool>(EditorServiceProviderInstanceRef());
    }

    nau::ServiceProvider& EditorServiceProvider()
    {
        auto& instance = EditorServiceProviderInstanceRef();
        NAU_FATAL(instance);
        return *instance;
    }
}