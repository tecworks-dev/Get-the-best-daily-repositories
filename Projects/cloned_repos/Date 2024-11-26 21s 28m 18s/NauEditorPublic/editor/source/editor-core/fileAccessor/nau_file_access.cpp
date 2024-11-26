// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_file_access.hpp"
#include "nau_log.hpp"

#include "magic_enum/magic_enum.hpp"
#include "fileAccessor/nau_visual_studio_code_accessor.hpp"

#include <QFileInfo>
#include <QDirIterator>


// ** NauFileAccess

NauFileAccess::NauFileAccessorMap NauFileAccess::m_fileAccessors;
NauAssetManagerInterface* NauFileAccess::m_assetManager = nullptr;

void NauFileAccess::registerAssetAccessor(NauEditorFileType assetType, std::shared_ptr<NauAssetEditorAccessor> assetAccessorPtr)
{
    if (m_fileAccessors.contains(assetType)) {
        NED_ERROR("Attempt to re-register asset editor accessor for {} type!", magic_enum::enum_name(assetType));
        return;
    }

    m_fileAccessors[assetType] = std::move(assetAccessorPtr);
}

void NauFileAccess::registerExternalAccessors()
{
    // Script access register
    auto vsCodeAccessor = std::make_shared<NauVisualStudioCodeAccessor>();
    if (vsCodeAccessor->init()) {
        m_fileAccessors[NauEditorFileType::Script] = vsCodeAccessor;
    }
}

bool NauFileAccess::openFile(const QString& path, NauEditorFileType type)
{
    NED_TRACE("NauFileAccess: open file at \"{}\", type={}", path.toUtf8().constData(), magic_enum::enum_name(type));

    auto itFileAccessor = m_fileAccessors.find(type);
    if (itFileAccessor == m_fileAccessors.end()) {
        NED_ERROR("There are no file accessor for file {}!", path.toUtf8().constData());
        return false;
    }

    std::string sourcePath = m_assetManager->sourcePathFromAsset(path.toUtf8().constData());
    if (sourcePath.empty()) {
        // To-Do: it's a temporary workaround. All files including scripts should have a meta file.
        sourcePath = path.toStdString();
    }

    bool result = itFileAccessor.value()->openFile(sourcePath.c_str());
    if (!result) {
        NED_ERROR("Failed to open file {}.", sourcePath.c_str());
    } 

    return result;
}

void NauFileAccess::setAssetManager(NauAssetManagerInterface* assetManager)
{
    NauFileAccess::m_assetManager = assetManager;
}

void NauFileAccess::warnIfContains(NauEditorFileType type)
{
    if (m_fileAccessors.contains(type)) {
        NED_WARNING("Overwriting file accessor for {}", magic_enum::enum_name(type));
    }
}
