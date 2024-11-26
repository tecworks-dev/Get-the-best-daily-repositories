// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Usd prims selection container

#pragma once

#include "nau/nau_usd_asset_editor_common_config.hpp"

#include "nau/nau_delegate.hpp"
#include <pxr/usd/usd/prim.h>


// ** NauUsdNormalizedContainer

using NauUsdPrimsSelection = std::vector<pxr::UsdPrim>;

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdNormalizedContainer
{
public:
    NauUsdNormalizedContainer(const NauUsdPrimsSelection& selection);
    ~NauUsdNormalizedContainer() = default;

    const NauUsdPrimsSelection& data() const;
    size_t count() const;

private:
    NauUsdPrimsSelection m_normalizedData;
    size_t m_allPrimsCount;
};


// ** NauUsdSelectionContainer

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdSelectionContainer
{
public:
    NauUsdSelectionContainer() = default;
    ~NauUsdSelectionContainer() = default;
    
    const NauUsdPrimsSelection& selection() const;
    std::string lastSelectedPath() const;

    void clear(bool notify = true);
    void updateFromPaths(pxr::UsdStageRefPtr currentScene, const std::vector<std::string>& paths);
    void addFromPath(pxr::UsdStageRefPtr currentScene, const pxr::SdfPath& path);

public:
    NauDelegate<const NauUsdPrimsSelection&> selectionChangedDelegate;

private:
     NauUsdPrimsSelection m_selectedPrims;
};

using NauUsdSelectionContainerPtr = std::shared_ptr<NauUsdSelectionContainer>;
