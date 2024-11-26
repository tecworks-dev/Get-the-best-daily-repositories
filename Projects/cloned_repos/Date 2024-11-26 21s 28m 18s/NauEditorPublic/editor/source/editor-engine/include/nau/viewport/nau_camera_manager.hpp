// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Camera objects manager

#pragma once

#include "nau/nau_editor_engine_api.hpp"

#include <unordered_map>

#include "nau/scene/components/camera_component.h"
#include "nau/scene/camera/camera.h"
#include "nau/scene/internal/scene_manager_internal.h"
#include "nau/scene/scene_object.h"
#include "nau/math/math.h"


// ** NauEditorCameraManager

class NAU_EDITOR_ENGINE_API NauEditorCameraManager
{
public:
    NauEditorCameraManager() = default;

    static constexpr eastl::string_view MainCameraName{ "Camera.Main" };

    void setActiveCamera(nau::Ptr<nau::scene::ICameraControl> camera);
    nau::Ptr<nau::scene::ICameraControl> activeCamera();

private:
    nau::Ptr<nau::scene::ICameraControl> m_activeCamera;
};