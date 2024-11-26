// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor file access system

#pragma once

#include "nau/nau_editor_config.hpp"

#include "nau/assets/nau_file_types.hpp"
#include "nau_file_accessor.hpp"
#include "nau/assets/nau_asset_editor_accessor.hpp"
#include "nau/assets/nau_asset_manager.hpp"

#include <QHash>
#include <QString>


// ** NauFileAccess
//
// Stores and calls file accessors for supported editor file types (NauEditorFileType)

class NAU_EDITOR_API NauFileAccess
{
    using NauFileAccessorMap = QHash<NauEditorFileType, std::shared_ptr<NauFileAccessorInterface>>;

public:
    NauFileAccess() = delete;

    template <NauEditorFileType type, std::derived_from<NauFileAccessorInterface> T, typename... Args>
    static void registerGenericAccessor(Args&&... args)
    {
        warnIfContains(type);
        m_fileAccessors[type] = std::make_shared<T>(std::forward<Args>(args)...);
    }

    static void registerAssetAccessor(NauEditorFileType assetType, std::shared_ptr<NauAssetEditorAccessor> assetAccessorPtr);
    static void registerExternalAccessors();

    static bool openFile(const QString& path, NauEditorFileType type);
    static void setAssetManager(NauAssetManagerInterface* assetManager);

private:
    static void warnIfContains(NauEditorFileType assetType);

private:
    static NauFileAccessorMap m_fileAccessors;
    static NauAssetManagerInterface* m_assetManager;
};