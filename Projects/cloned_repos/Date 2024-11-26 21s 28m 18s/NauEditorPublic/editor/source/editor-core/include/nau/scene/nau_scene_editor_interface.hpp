// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Scene editor interface

#pragma once

#include "nau/assets/nau_asset_editor.hpp"
#include "nau/outliner/nau_outliner_client_interface.hpp"
#include "nau/scene/nau_editor_scene_manager_interface.hpp"

#include "nau/rtti/rtti_object.h"

#include "nau/app/nau_editor_window_interface.hpp"


// ** NauSceneEditorInterface

struct NAU_EDITOR_API NAU_ABSTRACT_TYPE NauSceneEditorInterface : virtual NauAssetEditorInterface
{
    NAU_INTERFACE(NauSceneEditorInterface, NauAssetEditorInterface)

public:
    virtual std::shared_ptr<NauEditorSceneManagerInterface> sceneManager() = 0;
    virtual std::shared_ptr<NauOutlinerClientInterface> outlinerClient() = 0;

    virtual void changeMode(bool isPlaymode) = 0;
};