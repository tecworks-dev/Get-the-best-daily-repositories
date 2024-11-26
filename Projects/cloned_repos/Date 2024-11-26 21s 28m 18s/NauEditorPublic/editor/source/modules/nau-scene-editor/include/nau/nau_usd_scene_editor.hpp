// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Interface for usd scene editor

#pragma once

#include "nau/nau_usd_scene_editor_config.hpp"
#include "nau/selection/nau_usd_selection_container.hpp"
#include "nau/nau_usd_scene_synchronizer.hpp"

#include "nau/scene/nau_scene_editor_interface.hpp"
#include "nau/app/nau_editor_interface.hpp"

#include "nau/rtti/rtti_impl.h"


// ** NauUsdSceneEditorInterface

class NAU_USD_SCENE_EDITOR_API NAU_ABSTRACT_TYPE NauUsdSceneEditorInterface : public NauSceneEditorInterface
{
    NAU_INTERFACE(NauUsdSceneEditorInterface, NauSceneEditorInterface)

public:
    virtual NauUsdSelectionContainerPtr selectionContainer() const noexcept = 0;
    virtual const NauUsdSceneSynchronizer& sceneSynchronizer() const noexcept = 0;

    virtual PXR_NS::UsdPrim createPrim(const PXR_NS::SdfPath& parentPath, const PXR_NS::TfToken& name, const PXR_NS::TfToken& typeName, bool isComponent, std::string& uniquePath) = 0;
};
