// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Material editing classes

#pragma once

#include "nau/rtti/rtti_impl.h"
#include "nau/assets/nau_asset_editor.hpp"
#include "nau/assets/nau_asset_manager_client.hpp"
#include "nau/inspector/nau_usd_inspector_client.hpp"
#include "nau/inspector/nau_inspector.hpp"
#include "nau/undo-redo/nau_usd_scene_undo_redo.hpp"
#include "nau/app/nau_editor_interface.hpp"
#include "nau_dock_manager.hpp"
#include "pxr/usd/usd/stage.h"


class NauMaterialPreview;


// ** NauMaterialEditor
//
// Material editor instance. Include needed widgets, material editing logic etc.

class NauMaterialEditor final : public NauAssetEditorInterface
                              , public NauAssetManagerClientInterface

{
    NAU_CLASS_(NauMaterialEditor, NauAssetEditorInterface)

public:
    NauMaterialEditor();

    // TODO: Implement
    void initialize(NauEditorInterface* mainEditor) override;
    void terminate() override;
    void postInitialize() override;
    void preTerminate() override;

    // NauAssetEditorInterface overrides
    void createAsset(const std::string& assetPath) override;
    bool openAsset(const std::string& assetPath) override;
    bool saveAsset(const std::string& assetPath) override;

    [[nodiscard]] std::string editorName() const override;
    [[nodiscard]] NauEditorFileType assetType() const override;

    // NauAssetManagerClientInterface overrides
    void handleSourceAdded(const std::string& path) override;
    void handleSourceRemoved(const std::string& path) override;

private:
    bool openAssetInNewWindow(const QString& assetPath);

    void loadMaterialData(const QString& assetPath, NauInspectorPage& inspector);

private:
    NauEditorInterface* m_mainEditor;
    NauDockManager* m_editorDockManger;
    NauInspectorPage* m_mainInspector;

    NauMaterialPreview* m_preview = nullptr;
    NauInspectorPage* m_inspectorWithMaterial;
    NauDockWidget* m_dwMaterialPropertyPanel = nullptr;

    std::shared_ptr<NauUsdInspectorClient> m_inspectorClient;
    NauUsdSceneUndoRedoSystemPtr m_sceneUndoRedoSystem;

    pxr::UsdStageRefPtr m_materialAsset;
    std::string m_materialAssetPath;
};
