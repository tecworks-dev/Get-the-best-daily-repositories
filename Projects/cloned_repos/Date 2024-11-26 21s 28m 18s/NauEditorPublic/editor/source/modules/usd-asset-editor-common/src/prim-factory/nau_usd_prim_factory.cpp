// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/prim-factory/nau_usd_prim_factory.hpp"
#include "nau_log.hpp"
#include "nau/utils/nau_usd_editor_utils.hpp"

#include "pxr/usd/usdGeom/xformCache.h"


// ** NauUsdPrimCreatorAbstract

pxr::UsdPrim NauUsdPrimCreatorAbstract::createPrim(pxr::UsdStageWeakPtr stage, const pxr::SdfPath& path, const pxr::TfToken& typeName,
    const std::string& displayName, const pxr::GfMatrix4d& initialTransform, bool isComponent)
{
    pxr::UsdPrim prim = createPrimInternal(stage, path, typeName);

    prim.SetDisplayName(displayName);

    auto xformable = PXR_NS::UsdGeomXformable(prim);
    if (xformable) {
        // Block objectchanged event during transform setting when creating object
        PXR_NS::TfNotice::Block block;
        PXR_NS::UsdGeomXformOp transformOp = NauUsdPrimUtils::forceAddTransformOpOrder(xformable);
        transformOp.Set(initialTransform);
    }

    // Set component kind
    if (isComponent) {
        prim.SetKind(pxr::TfToken("component"));
    }

    return prim;
}


// ** NauUsdPrimFactory

NauUsdPrimFactory& NauUsdPrimFactory::instance()
{
    static NauUsdPrimFactory instance;
    return instance;
}

void NauUsdPrimFactory::addCreator(const std::string& primType, NauUsdPrimCreatorAbstractPtr creator)
{
    if (primType.empty()) {
        NED_ERROR("Usd prim factory: trying to add creator for empty type.");
        return;
    }

    if (!creator) {
        NED_ERROR("Usd prim factory: trying to add empty creator.");
        return;
    }

    m_creators[primType] = creator;
}

pxr::UsdPrim NauUsdPrimFactory::createPrim(pxr::UsdStageWeakPtr stage, const pxr::SdfPath& path, const pxr::TfToken& typeName,
        const std::string& displayName, const pxr::GfMatrix4d& initialTransform, bool isComponent)
{
    NauUsdPrimCreatorAbstractPtr creator = m_creators[typeName.GetString()];
    return creator->createPrim(stage, path, typeName, displayName, initialTransform, isComponent);
}

std::vector<std::string> NauUsdPrimFactory::registeredAllPrimCreators() const
{
    std::vector<std::string> types;

    for (const auto& resourceType : m_creators) {
        types.push_back(resourceType.first);
    }

    return types;
}

std::vector<std::string> NauUsdPrimFactory::registeredPrimCreators(primFilter filter) const
{
    std::vector<std::string> types;

    for (const auto& resourceType : m_creators) {
        if (filter(resourceType.first)) {
            types.push_back(resourceType.first);
        }
    }

    return types;
}
