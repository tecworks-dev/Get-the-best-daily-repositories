// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Interfaces for editors implementation

#pragma once

#include "nau/assets/nau_asset_editor.hpp"
#include "nau/nau_editor_config.hpp"

#include "fileAccessor/nau_file_accessor.hpp"

#include <string>
#include <memory>


// ** NauAssetEditorAccessor
//
// Asset editor accessor implementation

class NAU_EDITOR_API NauAssetEditorAccessor : public NauFileAccessorInterface
{
public:
    explicit NauAssetEditorAccessor(NauAssetEditorInterface* assetEditor);

    bool init() override;
    bool openFile(const QString& assetPath) override;

private:
    NauAssetEditorInterface* m_assetEditorPtr;
};