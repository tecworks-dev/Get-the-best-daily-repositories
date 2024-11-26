// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Thumbnail manager interface

#pragma once

#include "nau/nau_editor_config.hpp"
#include <filesystem>
#include "nau/assets/asset_meta_info.h"


// ** NauThumbnailManagerInterface

class NAU_EDITOR_API NauThumbnailManagerInterface
{
public:
    virtual bool exists(const std::filesystem::path& assetPath) const = 0;
    virtual std::pair<bool, std::filesystem::path> thumbnailForAsset(const std::filesystem::path& assetPath) = 0;
    virtual std::pair<bool, std::filesystem::path> generateThumbnail(const nau::AssetMetaInfoBase& assetMetaInfoBase) = 0;
};

using NauThumbnailManagerInterfacePtr = std::shared_ptr<NauThumbnailManagerInterface>;