// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/relationship.h>

#include "nau/input_system.h"


// ** NauInputAssetWatcher

class NauInputAssetWatcher
{
public:
    NauInputAssetWatcher();

    void addAsset(const std::string& assetPath);
    void removeAsset(const std::string& assetPath);
    void updateAsset(const std::string& assetPath);

    void makeAssetCurrent(const std::string& assetPath);

public:
    bool isAssetAdded(const std::string& assetPath);

    bool saveToFile(const std::string& path);
    bool loadFromFile(const std::string& path);

private:
    nau::IInputSignal* createSignal(const pxr::UsdPrim& signalPrim);
    nau::IInputSignal* createModifierChain(const pxr::UsdPrim& bindPrim);

    nau::IInputSignal* combineWithModifier(nau::IInputSignal* mainSignal, nau::IInputSignal* modifierSignal);

private:
    void parseAllBinds();

    void removeAllBind();
    void removeAllForBind(const pxr::UsdPrim& bindPrim);

    void parseBind(const pxr::UsdPrim& bindPrim, nau::IInputAction::Type actionType, bool isAxis = false);

    bool validateAndExtractBindInfo(const pxr::UsdPrim& prim, int& bindId, pxr::SdfPathVector& signalPaths);
    bool isAttributeExistAndValid(const pxr::UsdPrim& prim, const std::string& propertyName, std::string& property);

private:
    using ModifierCreator = std::function<nau::IInputSignal* (nau::IInputSystem&, const pxr::UsdPrim&)>;
    std::unordered_map<std::string, ModifierCreator> m_modifierFactory;

private:
    PXR_NS::UsdStageRefPtr m_stage;

    nau::IInputSystem* m_inputSystem;

    std::unordered_set<std::string> m_assetsPath;
    std::unordered_map<int, eastl::shared_ptr<nau::IInputAction>> m_actions;
};