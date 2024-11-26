// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_gui_editor.hpp"
#include "nau/module/module.h"
#include "nau/nau_editor_plugin_manager.hpp"

namespace nau
{
    struct NauGuiEditorModule : public IModule
    {
        nau::string getModuleName() override
        {
            return "NauGuiEditorModule";
        }

        void initialize() override
        {
            Nau::EditorServiceProvider().addClass<NauGuiEditor>();
        }

        void deinitialize() override
        {
        }
        
        void postInit() override
        {
        }
    };
}  // namespace nau

IMPLEMENT_MODULE(nau::NauGuiEditorModule);
