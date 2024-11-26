// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "nau/assets/nau_asset_editor.hpp"
#include "nau/rtti/rtti_impl.h"

#include "nau/assets/nau_file_types.hpp"
#include "project/nau_project.hpp"

#include <memory>

#include "nau_dock_manager.hpp"
#include "nau_dock_widget.hpp"

#include "components/pages/nau_input_editor_page.hpp"

#include "usd_proxy/usd_property_proxy.h"
#include "usd_proxy/usd_prim_proxy.h"
#include "nau_input_asset_watcher.hpp"


// ** NauInputEditor

class NauInputEditor final : public NauAssetEditorInterface
{
    NAU_CLASS_(NauInputEditor, NauAssetEditorInterface)

public:
    explicit NauInputEditor() noexcept;

    // TODO: Implement
    void initialize(NauEditorInterface* mainEditor) override;
    void terminate() override;
    void postInitialize() override;
    void preTerminate() override;

    void createAsset(const std::string& assetPath) override;
    bool openAsset(const std::string& assetPath) override;
    bool saveAsset(const std::string& assetPath) override;

    [[nodiscard]] std::string editorName() const override;
    [[nodiscard]] NauEditorFileType assetType() const override;


private:
    void handleRemovedAction(const std::string& actionFilePath);
    void reset();

private:
    NauDockManager* m_dockManager;
    NauDockWidget* m_editorDockWidget;
    NauInputEditorPage* m_editorWidget;

    PXR_NS::UsdStageRefPtr m_stage;

    std::unique_ptr<NauInputAssetWatcher> m_assetWatcher;
};
