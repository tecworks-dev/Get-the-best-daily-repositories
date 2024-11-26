// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Scene editor service

#pragma once

#include "nau/async/work_queue.h"
#include "nau/graphics/core_graphics.h"
#include "nau/runtime/async_disposable.h"
#include "nau/service/service.h"


// TODO: Unify service for all editors

// ** NauSceneEditorService

class NauSceneEditorService final : public nau::IServiceInitialization,
                                    public nau::IServiceShutdown
{
    NAU_RTTI_CLASS(NauSceneEditorService, nau::IServiceInitialization, nau::IServiceShutdown)
public:
    NauSceneEditorService();

private:
    nau::async::Task<> preInitService() override;

    nau::async::Task<> initService() override;

    nau::async::Task<> shutdownService() override;

private:
    std::thread m_thread;
    nau::WorkQueue::Ptr m_workQueue;
    nau::async::Task<> m_shutdownCompletedTask;
};
