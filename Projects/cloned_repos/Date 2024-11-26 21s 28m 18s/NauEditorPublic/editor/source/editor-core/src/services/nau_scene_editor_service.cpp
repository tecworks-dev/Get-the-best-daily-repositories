// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_scene_editor_service.hpp"

#include "nau/scene/nau_scene_editor_interface.hpp"

#include "nau/service/service.h"
#include "nau/nau_editor_plugin_manager.hpp"
#include "nau/threading/set_thread_name.h"

#include "nau/app/application.h"

#include "nau/nau_editor_plugin_manager.hpp"
#include "nau/app/nau_editor_services.hpp"

#include "nau/assets/nau_asset_editor.hpp"
#include "nau/assets/nau_asset_manager.hpp"



// ** CreateAssetEditors

void CreateAssetEditors(nau::async::TaskSource<>& readyTaskSource)
{
    auto classes = Nau::EditorServiceProvider().findClasses<NauAssetEditorInterface>();

    if (classes.empty()) {
        readyTaskSource.resolve();
        return;
    }

    for (auto registerClass : classes) {
        auto ctor = registerClass->getConstructor();
        NAU_FATAL(ctor);

        nau::Ptr<NauAssetEditorInterface> assetEditorPtr = ctor->invokeToPtr(nullptr, {});

        if (!assetEditorPtr) {
            NAU_FAILURE(assetEditorPtr->editorName().c_str());
            readyTaskSource.resolve();
            return;
        }

        Nau::EditorServiceProvider().addService(assetEditorPtr);
        assetEditorPtr->initialize(&Nau::Editor());
    }
}

void CreateAssetProcessor(nau::async::TaskSource<>& readyTaskSource)
{
    auto classes = Nau::EditorServiceProvider().findClasses<NauAssetFileProcessorInterface>();

    if (classes.empty()) {
        readyTaskSource.resolve();
        return;
    }

    auto ctor = classes.front()->getConstructor();
    NAU_FATAL(ctor);

    nau::Ptr<NauAssetFileProcessorInterface> assetProcessorPtr = ctor->invokeToPtr(nullptr, {});

    if (!assetProcessorPtr) {
        NAU_FAILURE("Failed to create asset processor");
        readyTaskSource.resolve();
        return;
    }

    Nau::EditorServiceProvider().addService(assetProcessorPtr);
}


// ** NauSceneEditorService

NauSceneEditorService::NauSceneEditorService() = default;

nau::async::Task<> NauSceneEditorService::initService()
{
    using namespace nau::async;

    TaskSource<> initCompletedTaskSource;
    Task<> initCompletedTask = initCompletedTaskSource.getTask();
    m_workQueue = nau::WorkQueue::create();

    m_thread = std::thread{
        [this, readyTaskSource = std::move(initCompletedTaskSource)]() mutable
        {
            TaskSource<> shutdownCompletedTaskSource;
            m_shutdownCompletedTask = shutdownCompletedTaskSource.getTask();
            nau::threading::setThisThreadName("Asset editors");
            Executor::setThisThreadExecutor(m_workQueue);

            scope_on_leave
            {
                shutdownCompletedTaskSource.resolve();
            };

            CreateAssetProcessor(readyTaskSource);
            CreateAssetEditors(readyTaskSource);

            readyTaskSource.resolve();
            readyTaskSource = nullptr;
        }};

    co_await initCompletedTask;
}

nau::async::Task<> NauSceneEditorService::shutdownService()
{
    NAU_FATAL(m_workQueue);

    m_workQueue->notify();
    m_thread.join();

    co_await m_shutdownCompletedTask;

}

nau::async::Task<> NauSceneEditorService::preInitService()
{
    return nau::async::makeResolvedTask();
}