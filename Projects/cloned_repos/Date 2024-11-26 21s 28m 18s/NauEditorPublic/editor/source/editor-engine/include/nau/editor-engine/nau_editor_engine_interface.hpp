// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor engine interface

#pragma once

#include "nau/nau_editor_engine_api.hpp"
#include "nau/viewport/nau_camera_manager.hpp"
#include "nau/viewport/nau_viewport_manager.hpp"

#include "nau/async/task_base.h"


// ** NauEditorEngineInterface

class NAU_EDITOR_ENGINE_API NauEditorEngineInterface
{
public:
    virtual ~NauEditorEngineInterface() = default;
    
    virtual std::shared_ptr <NauViewportManagerInterface> viewportManager() const = 0;
    virtual std::shared_ptr<NauEditorCameraManager> cameraManager() const = 0;

    virtual void tick() = 0;
    
    virtual bool initialize(const std::string& rootDir, const std::string& userDllDir, const QStringList& projectModules) = 0;
    virtual bool terminateSync() = 0;

    // Playmode functions
    virtual void startPlay() = 0;
    virtual void stopPlay() = 0;
    virtual void pausePlay(const bool isPaused) = 0;

    template<typename T>
    nau::Result<T> runTaskSync(const nau::async::Task<T>& task);
};


template<typename T>
nau::Result<T> NauEditorEngineInterface::runTaskSync(const nau::async::Task<T>& task)
{
    while (!task.isReady()) {
        tick();
    }

    return task.isRejected() ? task.getError() : task.asResult();
}
