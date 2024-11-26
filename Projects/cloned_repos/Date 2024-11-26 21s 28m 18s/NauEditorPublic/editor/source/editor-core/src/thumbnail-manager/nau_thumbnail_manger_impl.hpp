// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Thumbnail manager implementation

#pragma once

#include "nau/thumbnail-manager/nau_thumbnail_manger.hpp"
#include "nau_thumbnail_generator.hpp"

//#include "nau/assets/nau_asset_manager.hpp"
#include "nau_project_browser_item_type_resolver.hpp"

#include <unordered_map>
#include "nau/assets/nau_asset_manager.hpp"


// ** NauThumbnailManager
// Thumbnail manager implementation

class NauThumbnailManager : public NauThumbnailManagerInterface
{
    using GeneratorsMap = std::unordered_map<NauEditorFileType, NauThumbnailGeneratorInterfacePtr>;
    using ThumbnailMap = std::unordered_map<std::filesystem::path, std::filesystem::path>;

public:
    NauThumbnailManager(const std::filesystem::path& folder, NauProjectBrowserItemTypeResolverInterfacePtr typeResolver);
    
    bool exists(const std::filesystem::path& assetPath) const override;
    std::pair<bool, std::filesystem::path> thumbnailForAsset(const std::filesystem::path& assetPath) override;
    std::pair<bool, std::filesystem::path> generateThumbnail(const nau::AssetMetaInfoBase& assetMetaInfoBase) override;

private:
    void registerGenerators();

    void generateThumbnails();
    void generateThumbnailsForTypedAssets(NauEditorFileType type, const eastl::vector<nau::AssetMetaInfoBase>& assets);

    std::filesystem::path buildThumbnailPath(const std::filesystem::path& assetPath);

    std::pair<bool, std::filesystem::path> makeGeneratorResult(const std::filesystem::path& path);
    std::pair<bool, std::filesystem::path> makeGeneratorErrorResult();

private:
    NauProjectBrowserItemTypeResolverInterfacePtr m_typeResolver;

    ThumbnailMap m_thumbnailMap;
    std::filesystem::path m_thumbnailsFolder;

    GeneratorsMap m_generators;
};