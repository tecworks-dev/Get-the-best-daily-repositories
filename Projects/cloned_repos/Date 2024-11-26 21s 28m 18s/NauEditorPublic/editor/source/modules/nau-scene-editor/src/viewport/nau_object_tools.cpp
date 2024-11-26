// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_object_tools.hpp"
#include "nau_transform_tools.hpp"
#include "nau/nau_constants.hpp"
#include "nau/math/nau_matrix_math.hpp"
#include "nau/math/math.h"
#include "nau/gizmo/nau_camera_gizmo.hpp"
#include "nau/NauComponentSchema/nauComponent.h"

#include "nau/utils/nau_usd_editor_utils.hpp"
#include "usd_proxy/usd_prim_proxy.h"

#include <pxr/base/gf/frustum.h>
#include <pxr/usd/usdGeom/camera.h>
#include <algorithm>


// ** NauObjectTools

NauObjectTools::NauObjectTools(NauUsdSelectionContainerPtr container)
    : m_applyDeltaCallbackId()
    , m_stoppedUsingCallbackId()
    , m_container(container)
{
    init();
}

NauObjectTools::~NauObjectTools() noexcept
{
    terminate();
}

void NauObjectTools::handleMouseInput(QMouseEvent* mouseEvent, float dpi)
{
    for (auto& gizmo : m_gizmos) {
        gizmo->handleMouseInput(mouseEvent, dpi);
    }
}

bool NauObjectTools::isUsing() const
{
    for (auto& gizmo : m_gizmos) {
        if (gizmo->isUsing()) {
            return true;
        }
    }
    return false;
}

void NauObjectTools::updateBasis()
{
    for (auto& gizmo : m_gizmos) {
        if (gizmo->isActive()) {
            auto primTransform = NauUsdPrimUtils::worldPrimTransform(m_container->selection().back());
            const nau::math::mat4 transform = NauUsdEditorMathUtils::gfMatrixToNauMatrix(primTransform);
            gizmo->setBasis(transform);
        }
    }
}

std::string NauObjectTools::getComponentTypeName(const pxr::UsdPrim& prim)
{
    std::string typeName;
    if (pxr::UsdNauComponent component{ prim }) {
        component.GetComponentTypeNameAttr().Get(&typeName);
    }
    return typeName;
}

nau::math::mat4 NauObjectTools::transform(const nau::math::mat4& transform, const nau::math::vec3& delta) const
{
    nau::math::mat4 newTransform = transform;
    newTransform.setCol3({
        std::clamp(newTransform[3][0] + delta.getX(), NAU_POSITION_MIN_LIMITATION, NAU_POSITION_MAX_LIMITATION),
        std::clamp(newTransform[3][1] + delta.getY(), NAU_POSITION_MIN_LIMITATION, NAU_POSITION_MAX_LIMITATION),
        std::clamp(newTransform[3][2] + delta.getZ(), NAU_POSITION_MIN_LIMITATION, NAU_POSITION_MAX_LIMITATION),
        newTransform[3][3]
    });

    return newTransform;
}

void NauObjectTools::applyDeltaToSelection(nau::math::vec3 delta)
{
    for (auto selected : m_container->selection()) {
        auto primTransform = NauUsdPrimUtils::localPrimTransform(selected);
        const nau::math::mat4 oldTransform = NauUsdEditorMathUtils::gfMatrixToNauMatrix(primTransform);
        const pxr::GfMatrix4d newTransform = NauUsdEditorMathUtils::nauMatrixToGfMatrix(transform(oldTransform, delta));
        NauUsdPrimUtils::setPrimTransform(selected, newTransform);
    }
}

void NauObjectTools::init()
{
    auto selection = m_container->selection();

    for (auto& gizmo : m_gizmos) {
        // set gizmo pivot if there is a selected object
        if (!selection.empty()) {
            // get pivot from first object from selection
            auto primTransform = NauUsdPrimUtils::worldPrimTransform(selection.back());
            const nau::math::mat4 transform = NauUsdEditorMathUtils::gfMatrixToNauMatrix(primTransform);
            gizmo->activate(transform);
        }

        m_startedUsingCallbackId = gizmo->startedUsing.addCallback([this]() {
            startUse();
        });

        // Called on every gizmo update, doesn't generate an undo step
        m_applyDeltaCallbackId = gizmo->deltaUpdated.addCallback([this](nau::math::vec3 delta) {
            applyDeltaToSelection(delta);
        });

        // Called at the end of gizmo interactions, generates an undo step
        m_stoppedUsingCallbackId = gizmo->stoppedUsing.addCallback([this]() {
            stopUse();
        });
    }

    auto selectionChangedLambda = [this](const NauUsdPrimsSelection& selection) {
        onSelectionChanged(selection);
    };
    m_selectionChangedCallbackId = m_container->selectionChangedDelegate.addCallback(selectionChangedLambda);

    m_transformToolCallbackId = NauTransformTool::StopUsingDelegate.addCallback([this](const pxr::SdfPath& primPath, const pxr::GfMatrix4d& originalTransform, const pxr::GfMatrix4d& newTransform) {
        const nau::math::mat4 transform = NauUsdEditorMathUtils::gfMatrixToNauMatrix(newTransform);
        for (auto& gizmo: m_gizmos) {
            gizmo->setBasis(transform);
        }
    });
}

void NauObjectTools::terminate()
{
    NauTransformTool::StopUsingDelegate.deleteCallback(m_transformToolCallbackId);
    for (auto& gizmo : m_gizmos) {
        gizmo->deactivate();
        gizmo->startedUsing.deleteCallback(m_startedUsingCallbackId);
        gizmo->deltaUpdated.deleteCallback(m_applyDeltaCallbackId);
        gizmo->stoppedUsing.deleteCallback(m_stoppedUsingCallbackId);
        m_container->selectionChangedDelegate.deleteCallback(m_selectionChangedCallbackId);
    }
}

void NauObjectTools::startUse()
{
    m_originalSelectedObjectsTransform.clear();

    for (auto selected : m_container->selection()) {
        m_originalSelectedObjectsTransform[selected.GetPrimPath().GetString()] = NauUsdPrimUtils::localPrimTransform(selected);;
    }
}

void NauObjectTools::stopUse()
{
    m_originalSelectedObjectsTransform.clear();
}

void NauObjectTools::onSelectionChanged(const NauUsdPrimsSelection& selection)
{
    for (auto& gizmo : m_gizmos) {
        gizmo->deactivate();
    }
    m_gizmos.clear();
    if (!selection.empty()) {
        const pxr::UsdPrim& selectedPrim = selection.back();
        createObjectGizoms(selectedPrim);
    }
    for (auto& gizmo : m_gizmos) {
        auto primTransform = NauUsdPrimUtils::worldPrimTransform(selection.back());
        const nau::math::mat4 transform = NauUsdEditorMathUtils::gfMatrixToNauMatrix(primTransform);
        gizmo->activate(transform);
    }
}

void NauObjectTools::createObjectGizoms(const pxr::UsdPrim& prim)
{
    constexpr std::string_view CAMERA_TYPE = "nau::scene::CameraComponent";
    std::string typeName = getComponentTypeName(prim);
    if (typeName == CAMERA_TYPE) {
        addCameraGizmo(prim);
    }
}

void NauObjectTools::addCameraGizmo(const pxr::UsdPrim& prim)
{
    auto getCameraParams = [prim]() -> std::vector<nau::math::Point3> {

        constexpr std::string_view FOV_NAME = "FieldOfView";
        constexpr std::string_view FAR_NAME = "ClipFarPlane";
        constexpr std::string_view NEAR_NAME = "ClipNearPlane";

        float fovValue = 60.f;
        float nearValue = 0.f;
        float farValue = 1000.f;
        pxr::VtValue value;

        UsdProxy::UsdProxyPrim proxyPrim{ prim };
        for (auto&& [_, property] : proxyPrim.getProperties()) {
            const std::string& propertyName = property->getName().GetString();
            property->getValue(&value);
            if (propertyName == FOV_NAME) {
                fovValue = value.Get<float>();
            } else if (propertyName == FAR_NAME) {
                farValue = value.Get<float>();
            } else if (propertyName == NEAR_NAME) {
                nearValue = value.Get<float>();
            }
        }
        pxr::GfCamera camera;
        camera.SetClippingRange({ nearValue, farValue });
        camera.SetPerspectiveFromAspectRatioAndFieldOfView(16.f / 9.f, fovValue, pxr::GfCamera::FOVHorizontal);

        const std::vector<pxr::GfVec3d> frustumPoints = camera.GetFrustum().ComputeCorners();
        std::vector<nau::math::Point3> frustumPointsNew;
        frustumPointsNew.reserve(frustumPoints.size());
        for (const auto& point : frustumPoints) {
            frustumPointsNew.emplace_back(point[0], point[1], point[2]);
        }
        return frustumPointsNew;
    };
    auto&& gizmo = std::make_unique<NauCameraGizmo>(std::move(getCameraParams()));
    gizmo->setCallback(getCameraParams);
    m_gizmos.emplace_back(std::move(gizmo));
}
