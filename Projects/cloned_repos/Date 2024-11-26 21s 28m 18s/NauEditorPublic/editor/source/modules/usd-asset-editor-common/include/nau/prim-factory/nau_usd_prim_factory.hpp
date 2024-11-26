// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Usd prim factory

#pragma once

#include "nau/nau_usd_asset_editor_common_config.hpp"

#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/prim.h"


// ** NauUsdPrimCreatorAbstract

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPrimCreatorAbstract
{
public:
    // TODO: Wrap part of the arguements to PrimCreationContext
    pxr::UsdPrim createPrim(pxr::UsdStageWeakPtr stage, const pxr::SdfPath& path, const pxr::TfToken& typeName,
        const std::string& displayName, const pxr::GfMatrix4d& initialTransform, bool isComponent);

protected:
    virtual pxr::UsdPrim createPrimInternal(pxr::UsdStageWeakPtr stage, const pxr::SdfPath& path, const pxr::TfToken& typeName) = 0;
};
using NauUsdPrimCreatorAbstractPtr = std::shared_ptr<NauUsdPrimCreatorAbstract>;


// ** NauUsdPrimFactory

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPrimFactory
{
public:
    static NauUsdPrimFactory& instance();

    void addCreator(const std::string& primType, NauUsdPrimCreatorAbstractPtr creator);

    // TODO: Wrap part of the arguements to PrimCreationContext
    pxr::UsdPrim createPrim(pxr::UsdStageWeakPtr stage, const pxr::SdfPath& path, const pxr::TfToken& typeName,
        const std::string& displayName, const pxr::GfMatrix4d& initialTransform, bool isComponent);

    typedef bool (*primFilter)(const std::string&);

    std::vector<std::string> registeredAllPrimCreators() const;
    std::vector<std::string> registeredPrimCreators(primFilter filter) const;

private:
    std::map<std::string, NauUsdPrimCreatorAbstractPtr> m_creators;
};
