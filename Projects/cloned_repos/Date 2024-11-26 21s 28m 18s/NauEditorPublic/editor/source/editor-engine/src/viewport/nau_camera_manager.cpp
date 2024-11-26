// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.S

#include "nau/viewport/nau_camera_manager.hpp"


// ** NauEditorCameraManager

void NauEditorCameraManager::setActiveCamera(nau::Ptr<nau::scene::ICameraControl> camera)
{
    m_activeCamera = camera;
    m_activeCamera->setRotation(nau::math::quat::identity());
}

nau::Ptr<nau::scene::ICameraControl> NauEditorCameraManager::activeCamera()
{
    return m_activeCamera;
}