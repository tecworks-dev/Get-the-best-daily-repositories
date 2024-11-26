// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/selection/nau_object_selection.hpp"
#include "nau/editor-engine/nau_editor_engine_services.hpp"
#include "nau/viewport/nau_viewport_utils.hpp"
#include "nau/nau_constants.hpp"

#include "nau/diag/assertion.h"
#include "nau/debugRenderer/debug_render_system.h"
#include "nau/service/service_provider.h"
#include "nau/scene/scene_manager.h"

#include <imgui.h>

#include <QMouseEvent>


// ** NauViewportSelectionUtils

static nau::scene::SceneObject* hitObjectFromScreenSpace(float hitX, float hitY)
{
    nau::math::vec3 worldDirection{};
    nau::math::vec3 worldTransform{};

    const nau::math::vec2 screenPoint = { hitX, hitY };

    Nau::Utils::screenToWorld(screenPoint, worldTransform, worldDirection);

    // TODO: Get object that are in line of sight
    float minRayDistance = FLT_MAX;
    nau::scene::SceneObject* selectedObject = nullptr;

    auto activeScene = nau::getServiceProvider().get<nau::scene::ISceneManager>().getActiveScenes()[0];
    for (const auto& object : activeScene->getRoot().getChildObjects(true)) {
        
        const auto transform = object->getWorldTransform();

        // TODO: Use engine object OBB or AABB
        nau::math::AABB objectBBox = nau::math::AABB({-1,-1,-1 }, { 1,1,1 });
        objectBBox.Transform(transform.getMatrix());
        
        const nau::math::Ray ray(worldTransform, worldDirection);

        float rayDistance = 0.f;
        if (nau::math::rayIntersectsAABB(ray, objectBBox, &rayDistance)) {
            // If the first object got under the cast - just select it
            // If not - compare ray distances
            if (!selectedObject || (rayDistance < minRayDistance)) {
                selectedObject = object;
                minRayDistance = rayDistance;
            }
        }
    }

    return selectedObject;
}


// ** NauSelectionContainer

void NauSelectionContainer::clearSelection(bool notify)
{
    m_selectedObjects.clear();

    if (notify) {
        selectionChangedDelegate.broadcast(m_selectedObjects);
    }
}

bool NauSelectionContainer::isSelected(nau::scene::SceneObject* object)
{
    for (const auto& selectedObject : m_selectedObjects) {
        if (selectedObject->getUid() == object->getUid()) {
            return true;
        }
    }

    return false;
}

void NauSelectionContainer::addToSelection(nau::scene::SceneObject* object)
{
    if (isSelected(object)) {
        return;
    }

    m_selectedObjects.push_back(object);
    selectionChangedDelegate.broadcast(m_selectedObjects);
}

void NauSelectionContainer::removeFromSelection(nau::scene::SceneObject* object)
{
    NAU_ASSERT(isSelected(object));
    // delete even elements from the vector
    m_selectedObjects.erase(std::remove_if(m_selectedObjects.begin(), m_selectedObjects.end(), [&object](nau::scene::SceneObject* selected) {
        return selected->getUid() == object->getUid();
    }));

    selectionChangedDelegate.broadcast(m_selectedObjects);
}

void NauSelectionContainer::onSelectionChanged(const NauSelectionList& newSelection)
{
    m_selectedObjects.clear();
    m_selectedObjects = newSelection;

    selectionChangedDelegate.broadcast(m_selectedObjects);
}