// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor engine instance managment functions

#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include "nau/editor-engine/nau_editor_engine_interface.hpp"

#include <memory>


// ** Editor engine service functions

namespace Nau
{
    NAU_EDITOR_ENGINE_API std::unique_ptr<NauEditorEngineInterface> CreateEditorEngine();
    NAU_EDITOR_ENGINE_API void SetCurrentEditorEngine(std::unique_ptr<NauEditorEngineInterface>&&);
    NAU_EDITOR_ENGINE_API NauEditorEngineInterface& EditorEngine();
}
