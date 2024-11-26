// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/assets/nau_asset_editor_accessor.hpp"
#include "nau_log.hpp"
#include "nau_assert.hpp"
#include <QFileInfo>
#include <QDirIterator>


// ** NauAssetEditorAccessor

NauAssetEditorAccessor::NauAssetEditorAccessor(NauAssetEditorInterface* assetEditor)
    : m_assetEditorPtr(std::move(assetEditor))
{

}

bool NauAssetEditorAccessor::init()
{
    return true;
}

bool NauAssetEditorAccessor::openFile(const QString& assetPath)
{
    if (m_assetEditorPtr == nullptr) {
        NED_ERROR("Asset editor accessor not inited for file {}", assetPath);
        return false;
    }

    return m_assetEditorPtr->openAsset(assetPath.toUtf8().constData());
}