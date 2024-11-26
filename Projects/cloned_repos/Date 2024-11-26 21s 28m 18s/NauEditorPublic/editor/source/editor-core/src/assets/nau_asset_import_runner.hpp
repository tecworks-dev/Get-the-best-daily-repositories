// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Asset import processes manager

#pragma once

#include "baseWidgets/nau_widget_utility.hpp"

#include <filesystem>


// ** NauAssetImportRunner
// TODO: Made a common class for process running
// Now there is a lot of common code with game build process starting

class NauAssetImportRunner
{
public:
    NauAssetImportRunner(const std::filesystem::path& projectPath, const std::filesystem::path& assetToolPath);
    void run(const std::optional<std::filesystem::path>& assetPath);

private:
    std::filesystem::path m_assetToolPath;
    std::filesystem::path m_projectDir;
    std::unique_ptr<NauProcess> m_importProcess;

    bool m_isRunning = false;
};
