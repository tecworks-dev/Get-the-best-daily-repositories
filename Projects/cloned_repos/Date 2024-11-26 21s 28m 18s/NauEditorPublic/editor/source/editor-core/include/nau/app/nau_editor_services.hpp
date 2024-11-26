// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor window instance managment functions

#pragma once

#include "nau/nau_editor_config.hpp"

#include "nau/app/nau_editor_interface.hpp"
#include "project/nau_project.hpp"

#include <memory>


// ** Editor window service functions

namespace Nau
{
    NAU_EDITOR_API std::unique_ptr<NauEditorInterface> CreateEditor(NauProjectPtr project);
    NAU_EDITOR_API void SetDefaultEditor(std::unique_ptr<NauEditorInterface>&&);
    NAU_EDITOR_API NauEditorInterface& Editor();
}