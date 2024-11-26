// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_view_attributes.hpp"


QStringList NauResourceSelector::resourceList(NauStringViewAttributeType type)
{
    QStringList resources;

    //if (type == NauStringViewAttributeType::MODEL) {
    //    resources = NauEngineResourceAPI::getLoadedModels();
    //} else if (type == NauStringViewAttributeType::MATERIAL) {
    //    resources = NauEngineResourceAPI::getLoadedMaterials();
    //} else if (type == NauStringViewAttributeType::TEXTURE) {
    //    resources = NauEngineResourceAPI::getLoadedTextures();
    //}

    return resources;
}