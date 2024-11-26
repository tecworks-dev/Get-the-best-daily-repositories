// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport scene object tools

#pragma once

#include "nau/gizmo/nau_gizmo.hpp"
#include "nau/selection/nau_usd_selection_container.hpp"

#include <eastl/vector.h>
#include <memory>

#include <QMouseEvent>


// ** NauObjectTools

// TODO: create general interface for gizmo tools and object tools
class NauObjectTools final
{
public:
    NauObjectTools(NauUsdSelectionContainerPtr container);
    ~NauObjectTools() noexcept;

    void handleMouseInput(QMouseEvent* mouseEvent, float dpi);

    bool isUsing() const;

    void updateBasis();

private:
    [[nodiscard]]
    static std::string getComponentTypeName(const pxr::UsdPrim& prim);

    nau::math::mat4 transform(const nau::math::mat4& transform, const nau::math::vec3& delta) const;
    void applyDeltaToSelection(nau::math::vec3 delta);

    void init();
    void terminate();

    void startUse();
    void stopUse();

    void onSelectionChanged(const NauUsdPrimsSelection& selection);

    void createObjectGizoms(const pxr::UsdPrim& prim);
    void addCameraGizmo(const pxr::UsdPrim& prim);

private:
    NauCallbackId m_transformToolCallbackId;

    std::vector<std::unique_ptr<NauGizmoAbstract>> m_gizmos;
    std::unordered_map<std::string, pxr::GfMatrix4d> m_originalSelectedObjectsTransform;

    NauUsdSelectionContainerPtr m_container;

    NauCallbackId m_startedUsingCallbackId;
    NauCallbackId m_applyDeltaCallbackId;
    NauCallbackId m_stoppedUsingCallbackId;
    NauCallbackId m_selectionChangedCallbackId;
};
