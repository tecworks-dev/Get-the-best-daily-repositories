// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "nau/assets/nau_asset_editor.hpp"
#include "nau/rtti/rtti_impl.h"

#include "nau/assets/nau_file_types.hpp"
#include "nau/app/nau_editor_interface.hpp"
#include "project/nau_project.hpp"
#include "nau/physics/nau_physics_material_edit_widget.hpp"

#include <pxr/usd/usd/stage.h>

#include <QCoreApplication>


// ** NauPhysicsEditor

class NauPhysicsEditor : public NauAssetEditorInterface
                       , public NauAssetManagerClientInterface
{
    NAU_CLASS_(NauPhysicsEditor, NauAssetEditorInterface)

    Q_DECLARE_TR_FUNCTIONS(NauPhysicsEditor)

public:
    void initialize(NauEditorInterface* editorWindow) override;
    void preTerminate() override {};
    void terminate() override {};
    void postInitialize() override;

    void createAsset(const std::string& assetPath) override;
    bool openAsset(const std::string& assetPath) override;
    bool saveAsset(const std::string& assetPath) override;

    std::string editorName() const override;
    NauEditorFileType assetType() const override;

    void handleSourceAdded(const std::string& sourcePath) override;
    void handleSourceRemoved(const std::string& sourcePath) override;

private:
    NauDockManager* m_editorDockManger = nullptr;
    NauDockWidget* m_editorDockWidget = nullptr;
    NauEditorInterface* m_mainEditor = nullptr;
    NauPhysicsMaterialEditWidget* m_editorWidget = nullptr;

    PXR_NS::UsdStageRefPtr m_stage;
};
