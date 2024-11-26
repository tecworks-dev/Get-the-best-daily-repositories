// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/selection/nau_usd_selection_container.hpp"
#include "nau/utils/nau_usd_editor_utils.hpp"


// ** NauUsdNormalizedContainer

NauUsdNormalizedContainer::NauUsdNormalizedContainer(const NauUsdPrimsSelection& data)
{
    m_normalizedData = NauUsdEditorUtils::normalizedPrimsList(data);

    // TODO: Try to get prims with child count in normalizedPrimsList function
    pxr::SdfPathSet primPaths;
    NauUsdEditorUtils::collectPrimsWithChildrensPaths(m_normalizedData, primPaths);

    m_allPrimsCount = primPaths.size();
}

const NauUsdPrimsSelection& NauUsdNormalizedContainer::data() const
{
    return m_normalizedData;
}

size_t NauUsdNormalizedContainer::count() const
{
    return m_allPrimsCount;
}


// ** NauUsdSelectionContainer

const NauUsdPrimsSelection& NauUsdSelectionContainer::selection() const
{
    return m_selectedPrims;
}

std::string NauUsdSelectionContainer::lastSelectedPath() const
{
    if (m_selectedPrims.empty()) {
        return std::string();
    }

    return m_selectedPrims.back().GetPath().GetString();
}

void NauUsdSelectionContainer::clear(bool notify)
{
    m_selectedPrims.clear();

    if (notify) {
        selectionChangedDelegate.broadcast(m_selectedPrims);
    }
}

void NauUsdSelectionContainer::updateFromPaths(pxr::UsdStageRefPtr currentScene, const std::vector<std::string>& paths)
{
    m_selectedPrims.clear();

    for (const auto& path : paths) {
        auto selectedPrim = currentScene->GetPrimAtPath(pxr::SdfPath(path));
        m_selectedPrims.push_back(selectedPrim);
    }

    selectionChangedDelegate.broadcast(m_selectedPrims);
}

void NauUsdSelectionContainer::addFromPath(pxr::UsdStageRefPtr currentScene, const pxr::SdfPath& path)
{
    auto selectedPrim = currentScene->GetPrimAtPath(pxr::SdfPath(path));
    m_selectedPrims.push_back(selectedPrim);

    selectionChangedDelegate.broadcast(m_selectedPrims);
}