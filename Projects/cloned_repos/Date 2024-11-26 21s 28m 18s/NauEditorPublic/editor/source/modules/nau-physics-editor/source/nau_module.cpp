// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/physics/nau_physics_editor.hpp"
#include "nau/module/module.h"
#include "nau/nau_editor_plugin_manager.hpp"


// ** NauPhysicsEditorModule

struct NauPhysicsEditorModule : public nau::IModule
{
    nau::string getModuleName() override
    {
        return "NauPhysicsEditorModule";
    }

    void initialize() override
    {
        Nau::EditorServiceProvider().addClass<NauPhysicsEditor>();
    }

    void deinitialize() override
    {
    }

    void postInit() override
    {
    }
};

IMPLEMENT_MODULE(NauPhysicsEditorModule);
