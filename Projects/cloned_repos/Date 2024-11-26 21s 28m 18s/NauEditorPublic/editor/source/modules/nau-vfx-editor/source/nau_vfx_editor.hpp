// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Main NauVFX editor class

#pragma once

#include "nau/assets/nau_asset_editor.hpp"
#include "nau/rtti/rtti_impl.h"

#include "nau/assets/nau_file_types.hpp"
#include "nau/assets/nau_asset_editor.hpp"
#include "nau/assets/nau_asset_manager_client.hpp"
#include "nau/app/nau_editor_interface.hpp"
#include "project/nau_project.hpp"

#include <memory>

#include "nau_dock_manager.hpp"
#include "nau_dock_widget.hpp"

#include "components/pages/nau_vfx_editor_page.hpp"
#include "nau/assets/nau_asset_watcher.hpp"

#include "usd_proxy/usd_property_proxy.h"
#include "usd_proxy/usd_prim_proxy.h"

#include "nau/inspector/nau_usd_inspector_client.hpp"
#include "nau/undo-redo/nau_usd_scene_undo_redo.hpp"


// ** NauVFXEditor

class NauVFXEditor final : public NauAssetEditorInterface, public NauAssetManagerClientInterface
{
    NAU_CLASS_(NauVFXEditor, NauAssetEditorInterface)
        
public:
    NauVFXEditor();
    ~NauVFXEditor();
    
    void initialize(NauEditorInterface* mainEditor) override;
    void terminate() override;
    void postInitialize() override;
    void preTerminate() override;
    
    void createAsset(const std::string & assetPath) override;
    bool openAsset(const std::string & assetPath) override;
    bool saveAsset(const std::string & assetPath) override;
    
    virtual std::string editorName() const override;
    virtual NauEditorFileType assetType() const override;
    
    void handleSourceAdded(const std::string& sourcePath) override;
    void handleSourceRemoved(const std::string& sourcePath) override;

private:
    bool openAssetInNewWindow(const QString& assetPath);
    void loadVFXData(const QString& assetPath, NauInspectorPage& inspector);

private:
    NauVFXEditor(const NauVFXEditor&) = default;
    NauVFXEditor(NauVFXEditor&&) = default;
    NauVFXEditor & operator=(const NauVFXEditor&) = default;
    NauVFXEditor & operator=(NauVFXEditor&&) = default;
    
private:
    NauEditorInterface* m_mainEditor;
    NauDockManager* m_editorDockManger;

    NauInspectorPage* m_mainInspector;

    NauDockWidget* m_dwMVFXPropertyPanel;
    NauInspectorPage* m_inspectorWithVFX;

    std::shared_ptr<NauUsdInspectorClient> m_inspectorClient;
    NauUsdSceneUndoRedoSystemPtr m_sceneUndoRedoSystem;

    pxr::UsdStageRefPtr m_vfxAsset;
    std::string m_vfxAssetPath;
};