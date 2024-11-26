// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/undo-redo/nau_usd_scene_commands.hpp"
#include "nau/prim-factory/nau_usd_prim_factory.hpp"
#include "nau/utils/nau_usd_editor_utils.hpp"
#include "nau_log.hpp"

#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/xform.h"
#include "pxr/usd/sdf/namespaceEdit.h"
#include "pxr/usd/usdGeom/xformCache.h"

#include <filesystem>


// ** NauAbstractUsdSceneCommand

NauAbstractUsdSceneCommand::NauAbstractUsdSceneCommand(pxr::UsdStageRefPtr scene)
    : m_currentScene(scene)
{
}


// ** NauCommandCreateUsdPrim

NauCommandCreateUsdPrim::NauCommandCreateUsdPrim(
    pxr::UsdStageRefPtr scene,
    const std::string& path,
    const std::string& typeName,
    const std::string& name,
    const PXR_NS::GfMatrix4d& initialTransform,
    bool isComponent
    )
    : NauAbstractUsdSceneCommand(scene)
    , m_path(path)
    , m_typeName(typeName)
    , m_displayName(name)
    , m_initialTransform(initialTransform)
    , m_isComponent(isComponent)
{
}

void NauCommandCreateUsdPrim::execute()
{
    NauUsdPrimFactory::instance().createPrim(m_currentScene, pxr::SdfPath(m_path), pxr::TfToken(m_typeName), m_displayName, m_initialTransform, m_isComponent);

    NED_DEBUG("Execute object creation command, path: {}", m_path);
}

void NauCommandCreateUsdPrim::undo()
{
    NED_DEBUG("Undo object creation command, path: {}", m_path);

    m_currentScene->RemovePrim(pxr::SdfPath(m_path));
    
}

NauCommandDescriptor NauCommandCreateUsdPrim::description() const
{
    return {
        .id = id,
        .name = "Object creation",
        .objectId = m_path.c_str()
    };
}


// ** NauCommandRemoveUsdPrim

NauCommandRemoveUsdPrim::NauCommandRemoveUsdPrim(pxr::UsdStageRefPtr scene, const std::string& path)
    : NauAbstractUsdSceneCommand(scene)
    , m_path(path)
{
    auto usdPrim = scene->GetPrimAtPath(pxr::SdfPath(path));
    m_displayName = usdPrim.GetDisplayName();
    m_typeName = usdPrim.GetTypeName().GetString();
}

void NauCommandRemoveUsdPrim::execute()
{
    m_currentScene->RemovePrim(pxr::SdfPath(m_path));

    NED_DEBUG("Execute object remove command, path: {}", m_path);
}

void NauCommandRemoveUsdPrim::undo()
{
    auto newPrim = m_currentScene->DefinePrim(pxr::SdfPath(m_path), pxr::TfToken(m_typeName.c_str()));
    newPrim.SetDisplayName(m_displayName);

    NED_DEBUG("Undo object remove command, path: {}", m_path);
    
}

NauCommandDescriptor NauCommandRemoveUsdPrim::description() const
{
    return {
        .id = id,
        .name = "Object removing",
        .objectId = m_path.c_str()
    };
}


// ** NauCommandRenameUsdPrim

NauCommandRenameUsdPrim::NauCommandRenameUsdPrim(
    pxr::UsdStageRefPtr scene,
    const std::string& path,
    const std::string& newName
    )
    : NauAbstractUsdSceneCommand(scene)
    , m_path(path)
    , m_newName(newName)
{
    m_oldName = m_currentScene->GetPrimAtPath(pxr::SdfPath(path)).GetDisplayName();
}

void NauCommandRenameUsdPrim::execute()
{
    auto prim = m_currentScene->GetPrimAtPath(pxr::SdfPath(m_path));
    prim.SetDisplayName(m_newName);

    NED_DEBUG("Execute object rename command, path: {}", m_path);
}

void NauCommandRenameUsdPrim::undo()
{
    auto prim = m_currentScene->GetPrimAtPath(pxr::SdfPath(m_path));
    prim.SetDisplayName(m_oldName);

    NED_DEBUG("Undo object rename command, path: {}", m_path);
    
}

NauCommandDescriptor NauCommandRenameUsdPrim::description() const
{
    return {
        .id = id,
        .name = "Object renaming",
        .objectId = m_path.c_str()
    };
}


// ** NauCommandReparentUsdPrim

NauCommandReparentUsdPrim::NauCommandReparentUsdPrim(pxr::UsdStageRefPtr scene,
    const std::string& currentPath, const std::string& newPath)
    : NauAbstractUsdSceneCommand(scene)
    , m_oldPath(currentPath)
    , m_newPath(newPath)
{
}

void NauCommandReparentUsdPrim::execute()
{
    if (reparentPrim(m_oldPath, m_newPath)) {
        NED_DEBUG("UndoRedo: Executed object reparent command successfully, from: {} to {}",
            m_oldPath.GetString(), m_newPath.GetString());
        return;
    }

    NED_WARNING("UndoRedo: Failed to reparent object from: {} to {}",
        m_oldPath.GetString(), m_newPath.GetString());
}

void NauCommandReparentUsdPrim::undo()
{
    if (reparentPrim(m_newPath, m_oldPath)) {
        NED_DEBUG("UndoRedo: Undo object reparent command successfully, from: {} to {}",
            m_newPath.GetString(), m_oldPath.GetString());
        return;
    }

    NED_WARNING("UndoRedo: Failed to reparent object from: {} to {}",
        m_newPath.GetString(), m_oldPath.GetString());
}

NauCommandDescriptor NauCommandReparentUsdPrim::description() const
{
    return {
        .id = id,
        .name = "Object reparent",
        .objectId = m_oldPath.GetString()
    };
}

bool NauCommandReparentUsdPrim::reparentPrim(PXR_NS::SdfPath& oldPath, PXR_NS::SdfPath& newPath)
{
    using namespace PXR_NS;

    const auto oldXform = NauUsdPrimUtils::worldPrimTransform(m_currentScene->GetPrimAtPath(oldPath));

    SdfBatchNamespaceEdit edit;
    edit.Add(SdfNamespaceEdit::Reparent(oldPath, newPath, SdfNamespaceEdit::AtEnd));

    if (!m_currentScene->GetRootLayer()->Apply(edit)) {
        return false;
    }

    newPath = newPath.AppendPath(SdfPath(oldPath.GetName()));
    oldPath = oldPath.GetParentPath();

    UsdPrim prim = m_currentScene->GetPrimAtPath(SdfPath(newPath));

    UsdGeomXformCache cache;
    NauUsdPrimUtils::setPrimTransform(prim, oldXform * cache.GetParentToWorldTransform(prim).GetInverse());

    return true;
}


// ** NauCommandChangeUsdPrimProperty

NauCommandChangeUsdPrimProperty::NauCommandChangeUsdPrimProperty(
    pxr::UsdStageRefPtr scene, 
    const PXR_NS::SdfPath& path, 
    const PXR_NS::TfToken& propName, 
    const PXR_NS::VtValue& value
    )
    : NauAbstractUsdSceneCommand(scene)
    , m_path(path)
    , m_propName(propName)
    , m_value(value)
{
    if(auto proxyProp = proxyProperty(path, propName))
        proxyProp->getValue(&m_initialValue);
}

NauCommandChangeUsdPrimProperty::NauCommandChangeUsdPrimProperty(
    pxr::UsdStageRefPtr scene, 
    const PXR_NS::SdfPath& path, 
    const PXR_NS::TfToken& propName, 
    const PXR_NS::VtValue& initialValue,
    const PXR_NS::VtValue& newValue
    )
    : NauAbstractUsdSceneCommand(scene)
    , m_path(path)
    , m_propName(propName)
    , m_value(newValue)
    , m_initialValue(initialValue)
{
}

UsdProxy::UsdProxyPropertyPtr NauCommandChangeUsdPrimProperty::proxyProperty(const PXR_NS::SdfPath& path, const PXR_NS::TfToken& propName)
{
    auto prim = m_currentScene->GetPrimAtPath(path);
    auto proxyPrim = UsdProxy::UsdProxyPrim(prim);
    return proxyPrim.getProperty(propName);
}

void NauCommandChangeUsdPrimProperty::execute()
{
    auto proxyProp = proxyProperty(m_path, m_propName);
    if (proxyProp)
    {
        proxyProp->setValue(m_value);
        NED_DEBUG("Execute object property changing command, path: {}", m_path.GetString());
    }
    else
    {
        NED_WARNING("Execute object property changing command failed, path: {}", m_path.GetString());
    }
}

void NauCommandChangeUsdPrimProperty::undo()
{
    auto proxyProp = proxyProperty(m_path, m_propName);
    proxyProp->setValue(m_initialValue);

    NED_DEBUG("Undo object property changing command, path: {}", m_path.GetString());
}

NauCommandDescriptor NauCommandChangeUsdPrimProperty::description() const
{
    return {
        .id = id,
        .name = "Object property changing",
        .objectId = m_path.GetString().c_str()
    };
}


// ** NauCommandChangeUsdPrimAssetReference

NauCommandChangeUsdPrimAssetReference::NauCommandChangeUsdPrimAssetReference(pxr::UsdStageRefPtr scene, const PXR_NS::SdfPath& path, const PXR_NS::VtValue& newValue)
    : NauAbstractUsdSceneCommand(scene)
    , m_path(path)
    , m_value(newValue)
{
}

NauCommandChangeUsdPrimAssetReference::NauCommandChangeUsdPrimAssetReference(pxr::UsdStageRefPtr scene, const PXR_NS::SdfPath& path, const PXR_NS::VtValue& initialValue, const PXR_NS::VtValue& newValue)
    : NauAbstractUsdSceneCommand(scene)
    , m_path(path)
    , m_value(newValue)
    , m_initialValue(initialValue)
{
}

void NauCommandChangeUsdPrimAssetReference::execute()
{
    setPrimReference(m_value);

    NED_DEBUG("Execute asset reference changing command, path: {}", m_path.GetString());
}

void NauCommandChangeUsdPrimAssetReference::undo()
{
    setPrimReference(m_initialValue);

    NED_DEBUG("Undo asset reference changing command, path: {}", m_path.GetString());
}

NauCommandDescriptor NauCommandChangeUsdPrimAssetReference::description() const
{
    return {
        .id = id,
        .name = "Asset reference changing",
        .objectId = m_path.GetString().c_str()
    };
}

void NauCommandChangeUsdPrimAssetReference::setPrimReference(PXR_NS::VtValue value)
{
    auto prim = m_currentScene->GetPrimAtPath(m_path);
    auto typeName = prim.GetTypeName().GetString();
    prim.GetReferences().ClearReferences();

    PXR_NS::SdfReference ref = value.Get<PXR_NS::SdfReference>();

    prim.GetReferences().AddReference(ref);
    prim.Load();
}
