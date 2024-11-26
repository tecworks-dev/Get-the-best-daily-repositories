// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Thumbnail generators for assets

#pragma once

#include <filesystem>
#include <memory>


// ** NauThumbnailGeneratorInterface

class NauThumbnailGeneratorInterface
{
public:
    virtual bool generate(const std::filesystem::path& destPath, const std::filesystem::path& source) = 0;

    static std::string_view thumbnailExtension();
};

using NauThumbnailGeneratorInterfacePtr = std::shared_ptr<NauThumbnailGeneratorInterface>;


// ** NauTextureThumbnailGenerator

class NauTextureThumbnailGenerator : public NauThumbnailGeneratorInterface
{
public:
    bool generate(const std::filesystem::path& destPath, const std::filesystem::path& source) override;
};