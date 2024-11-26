// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor wrapper for engine.

#pragma once

#include "nau/editor-engine/nau_editor_engine_interface.hpp"
#include "nau/viewport/nau_camera_manager.hpp"

#include "nau/selection/nau_object_selection.hpp"

#include <eastl/string.h>
#include <eastl/vector.h>
#include <eastl/unordered_map.h>

#include <QTimer>

#include <nau/app/application.h>


// TODO: Implement with memento pattern
// ** NauEditorEngineSnapshot

class NauEditorEngineSnapshot
{
    struct Info
    {
        nau::math::Transform cameraTransform;
        NauBaseViewportControllerPtr viewportController;
    };

public:
    NauEditorEngineSnapshot() = default;
    ~NauEditorEngineSnapshot() = default;

    void takeShapshot(class NauEditorEngine* editorEngine);
    void restoreShapshot();

private:
    std::unique_ptr<Info> m_snapshotInfo;
    class NauEditorEngine* m_sourceEngine;
};
using NauEditorEngineSnapshotPtr = std::unique_ptr<NauEditorEngineSnapshot>;


// ** NauEditorEngine
// Engine wrapper with editor features

class NAU_EDITOR_ENGINE_API NauEditorEngine : public NauEditorEngineInterface
{
public:
    NauEditorEngine();
    ~NauEditorEngine();

    bool initialize(const std::string& rootDir, const std::string& userDllDir, const QStringList& projectModules) override;
    bool terminateSync() override;

    void tick() override;

    std::shared_ptr<NauViewportManagerInterface> viewportManager() const override;
    std::shared_ptr<NauEditorCameraManager> cameraManager() const override;

    // Playmode functions
    void startPlay() override;
    void stopPlay() override;
    void pausePlay(const bool flag) override;

    NauEditorEngine(NauEditorEngine const&) = delete;
    NauEditorEngine& operator=(NauEditorEngine const&) = delete;

private:
    void startEngineLoop(const std::string& rootDir);
    void stopEngineLoop();

    void stopEngine();

    void terminateEditorSystems();

private:
    eastl::unique_ptr<nau::Application> m_engineApplication;
    QTimer m_editorTickTimer;

    nau::async::Task<> m_editorDrawTask = nau::async::Task<>::makeResolved();

    std::shared_ptr<NauViewportManagerInterface> m_viewportManager;
    std::shared_ptr<NauEditorCameraManager> m_cameraManager;

    NauEditorEngineSnapshotPtr m_engineSnapshot;
    NauEditorMode m_editMode;
};
