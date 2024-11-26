// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_audio_editor.hpp"
#include "nau/module/module.h"
#include "nau/nau_editor_plugin_manager.hpp"


// ** NauAudioEditorModule

struct NauAudioEditorModule : public nau::IModule
{
    nau::string getModuleName() override
    {
        return "NauAudioEditorModule";
    }

    void initialize() override
    {
        Nau::EditorServiceProvider().addClass<NauAudioEditor>();
    }

    void deinitialize() override
    {
    }
        
    void postInit() override
    {
    }
};

IMPLEMENT_MODULE(NauAudioEditorModule);
