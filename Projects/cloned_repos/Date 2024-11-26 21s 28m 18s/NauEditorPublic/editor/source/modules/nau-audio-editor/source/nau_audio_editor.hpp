// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "nau/assets/nau_asset_editor.hpp"
#include "nau/rtti/rtti_impl.h"

#include "nau/assets/nau_file_types.hpp"
#include "nau/assets/nau_asset_manager_client.hpp"
#include "nau/app/nau_editor_interface.hpp"
#include "project/nau_project.hpp"


#include "nau/audio/audio_engine.hpp"

#include <memory>


// ** NauAudioEditor

class NauAudioEditor final : public NauAssetEditorInterface
                           , public NauAssetManagerClientInterface
{
    NAU_CLASS_(NauAudioEditor, NauAssetEditorInterface)

public:
    NauAudioEditor();
    ~NauAudioEditor();

    void initialize(NauEditorInterface* mainEditor) override;
    void terminate() override;
    void postInitialize() override;
    void preTerminate() override;

    void createAsset(const std::string& assetPath) override;
    bool openAsset(const std::string& assetPath) override;
    bool saveAsset(const std::string& assetPath) override;

    std::string editorName() const override;
    NauEditorFileType assetType() const override;

    void handleSourceAdded(const std::string& path) override;
    void handleSourceRemoved(const std::string& path) override;

    nau::audio::AudioAssetPtr findAsset(const std::string& path);
    bool loadContainer(const std::string& path);
    void saveContainer(nau::audio::AudioAssetContainerPtr container);

private:
    NauAudioEditor(const NauAudioEditor&) = default;
    NauAudioEditor(NauAudioEditor&&) = default;
    NauAudioEditor& operator=(const NauAudioEditor&) = default;
    NauAudioEditor& operator=(NauAudioEditor&&) = default;

private:
    NauInspectorPage*    m_inspector = nullptr;
    NauEditorInterface*  m_mainEditor;
    NauDockManager* m_editorDockManger;

    NauDockWidget* m_audioDockWidget;
    NauInspectorPage*  m_audioInspector;

    nau::audio::IAudioEngine& m_engine;
};
