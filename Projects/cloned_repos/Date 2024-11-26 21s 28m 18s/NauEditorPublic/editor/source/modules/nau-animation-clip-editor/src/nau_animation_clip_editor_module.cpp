// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/nau_animation_clip_editor.hpp"
#include "nau/module/module.h"
#include "nau/nau_editor_plugin_manager.hpp"

namespace nau
{
    struct NauAnimationClipEditorModule : public IModule
    {
        nau::string getModuleName() override
        {
            return "NauAnimationClipEditorModule";
        }

        void initialize() override
        {
            Nau::EditorServiceProvider().addClass<NauAnimationClipEditor>();
        }

        void deinitialize() override
        {
        }

        void postInit() override
        {
        }
    };
}  // namespace nau

IMPLEMENT_MODULE(nau::NauAnimationClipEditorModule);