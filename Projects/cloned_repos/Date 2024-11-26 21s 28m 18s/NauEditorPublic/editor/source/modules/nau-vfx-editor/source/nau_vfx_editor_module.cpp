// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_vfx_editor.hpp"
#include "nau/module/module.h"
#include "nau/nau_editor_plugin_manager.hpp"


// ** NauVFXEditorModule

struct NauVFXEditorModule : public nau::IModule
{
    nau::string getModuleName() override
    {
        return "NauVFXEditorModule";
    }

    void initialize() override
    {
        Nau::EditorServiceProvider().addClass<NauVFXEditor>();
    }

    void deinitialize() override
    {
    }
        
    void postInit() override
    {
    }
};

IMPLEMENT_MODULE(NauVFXEditorModule);
