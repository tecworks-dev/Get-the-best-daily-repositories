// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Asset manager client interface

#pragma once

#include "nau/nau_editor_config.hpp"
#include <string>


// ** NauAssetManagerClientInterface

class NAU_EDITOR_API NauAssetManagerClientInterface
{
public:
    virtual ~NauAssetManagerClientInterface() = default;

    virtual void handleSourceAdded(const std::string& sourcePath) = 0;
    virtual void handleSourceRemoved(const std::string& sourcePath) = 0;
};
