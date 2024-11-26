// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport selection utils

#pragma once

#include "nau/nau_editor_engine_api.hpp"

#include "nau/nau_delegate.hpp"
#include "nau/scene/scene_object.h"

#include "nau/math/math.h"

class QMouseEvent;


// ** NauSelectionContainer
//
// Container with selected objects that refers to objects from editor scene

// TODO: When the project folder structure changes, you should use this alias wherever appropriate
using NauSelectionList = std::vector<nau::scene::SceneObject*>;

class NAU_EDITOR_ENGINE_API NauSelectionContainer
{
public:
    NauSelectionContainer() = default;
    const NauSelectionList& selection() const { return m_selectedObjects; }
    void clearSelection(bool notify = true);

    bool isSelected(nau::scene::SceneObject* object);
    void addToSelection(nau::scene::SceneObject* object);
    void removeFromSelection(nau::scene::SceneObject* object);

    NauDelegate<const NauSelectionList&> selectionChangedDelegate;

    void onSelectionChanged(const NauSelectionList& newSelectionIds);

private:

    // TODO: Transfer to weak pointers references
    NauSelectionList m_selectedObjects;
};


// ** NauSelectionProxy
//
// Provides api for managing selection container
//
//class NauSelectionProxy
//{
//public:
//    NauSelectionProxy() = delete;
//    static void addToSelection(NauSceneObjectPtr entity, bool concatenate = true);
//    static void addToSelection(const NauSelectionList& entities, bool concatenate = true);
//    static void removeFromSelection(NauSceneObjectPtr entity);
//    static void clearSelection();
//    static const NauSelectionList& getSelectionGroup();
//
//    static std::shared_ptr<NauSelectionList> getSelectionGroupPtr();
//    static NauSelectionList getTransformableSelectionGroup();
//};