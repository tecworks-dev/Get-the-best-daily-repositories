// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport scene objects transfrom tools

#pragma once

#include "nau/gizmo/nau_gizmo.hpp"
#include "nau/selection/nau_usd_selection_container.hpp"

#include <eastl/vector.h>
#include <memory>

#include <QMouseEvent>


class QMouseEvent;


// ** NauTransformTool
//
// Base transform tool class. Contains gizmo controls functional

class NauTransformTool
{
public:
    NauTransformTool(std::unique_ptr<NauGizmoAbstract>&& gizmo, NauUsdSelectionContainerPtr container);
    virtual ~NauTransformTool();
    
    void handleMouseInput(QMouseEvent* mouseEvent, float dpi);

    bool isUsing() const { return m_gizmo->isUsing(); }

    void updateBasis();

    void setCoordinateSpace(GizmoCoordinateSpace space);

protected:
    virtual nau::math::mat4 transform(const nau::math::mat4& transform, const nau::math::vec3& delta) const = 0;
    virtual void applyDeltaToSelection(nau::math::vec3 delta);

private:
    void init();
    void terminate();

    void startUse();
    void stopUse();

    void onSelectionChanged(const NauUsdPrimsSelection& selection);

public:
    // TODO: Make it non-static
    inline static NauDelegate<const pxr::SdfPath&, const pxr::GfMatrix4d&, const pxr::GfMatrix4d&> StopUsingDelegate;

protected:
    std::unique_ptr<NauGizmoAbstract> m_gizmo;
    std::unordered_map<std::string, pxr::GfMatrix4d> m_originalSelectedObjectsTransform;

    NauUsdSelectionContainerPtr m_container;

private:
    NauCallbackId m_startedUsingCallbackId;
    NauCallbackId m_applyDeltaCallbackId;
    NauCallbackId m_stoppedUsingCallbackId;
    NauCallbackId m_selectionChangedCallbackId;
};


// ** NauTranslateTool
//
// Translation tool implementation. Contains translation from delta functional

class NauTranslateTool : public NauTransformTool
{
public:
    NauTranslateTool(NauUsdSelectionContainerPtr container);
protected:
    virtual nau::math::mat4 transform(const nau::math::mat4& transform, const nau::math::vec3& delta) const override;
};


// ** NauRotateTool
//
// Rotation tool implementation. Contains rotation from delta functional

class NauRotateTool : public NauTransformTool
{
public:
    NauRotateTool(NauUsdSelectionContainerPtr container);

protected:
    virtual nau::math::mat4 transform(const nau::math::mat4& transform, const nau::math::vec3& delta) const override;
};


// ** NauScaleTool
//
// Scaling tool implementation. Contains scaling from delta functional

class NauScaleTool : public NauTransformTool
{
public:
    NauScaleTool(NauUsdSelectionContainerPtr container);

protected:
    virtual nau::math::mat4 transform(const nau::math::mat4& transform, const nau::math::vec3& delta) const override;
};
