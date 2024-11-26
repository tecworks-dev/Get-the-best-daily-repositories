// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Prim creator implementations

#pragma once

#include "nau/nau_usd_asset_editor_common_config.hpp"
#include "nau/prim-factory/nau_usd_prim_factory.hpp"


// ** NauDefaultUsdPrimCreator

class NAU_USD_ASSET_EDITOR_COMMON_API NauDefaultUsdPrimCreator : public NauUsdPrimCreatorAbstract
{
protected:
    pxr::UsdPrim createPrimInternal(pxr::UsdStageWeakPtr stage, const pxr::SdfPath& path, const pxr::TfToken& typeName) override;
};


// ** NauResourceUsdPrimCreator

class NAU_USD_ASSET_EDITOR_COMMON_API NauResourceUsdPrimCreator : public NauUsdPrimCreatorAbstract
{
public:
    NauResourceUsdPrimCreator(const std::string& defaultAssetPath, const pxr::SdfPath& primPath);

protected:
    pxr::UsdPrim createPrimInternal(pxr::UsdStageWeakPtr stage, const pxr::SdfPath& path, const pxr::TfToken& typeName) override;

private:
    std::string m_defaultAssetPath;
    pxr::SdfPath m_primPath;
};

// ** NauResourceUsdPrimCreator

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPrimComponentCreator : public NauUsdPrimCreatorAbstract
{
public:
    NauUsdPrimComponentCreator(const std::string& typeName);

protected:
    pxr::UsdPrim createPrimInternal(pxr::UsdStageWeakPtr stage, const pxr::SdfPath& path, const pxr::TfToken& typeName) override;

private:
    std::string m_typeName;
};
