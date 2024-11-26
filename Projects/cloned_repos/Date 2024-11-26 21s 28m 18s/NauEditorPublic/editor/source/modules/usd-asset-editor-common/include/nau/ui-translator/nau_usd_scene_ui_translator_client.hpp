// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Translator client classes

#pragma once

#include "nau/nau_usd_asset_editor_common_config.hpp"

#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/sdf/notice.h>


// ** NauUITranslatorProxyNotice

class NAU_USD_ASSET_EDITOR_COMMON_API NauUITranslatorProxyNotice
{
public:
    NauUITranslatorProxyNotice(pxr::UsdNotice::ObjectsChanged const& notice);

    const pxr::SdfPathVector& resyncedPaths() const;
    const pxr::SdfPathVector& infoChanges() const;

    pxr::UsdStageWeakPtr stage() const;

private:
    pxr::UsdStageWeakPtr m_stage;

    pxr::SdfPathVector m_resyncedPaths;
    pxr::SdfPathVector m_infoChangesPaths;
};


// ** NauUITranslatorClientInterface

class NAU_USD_ASSET_EDITOR_COMMON_API NauUITranslatorClientInterface
{
public:
    virtual ~NauUITranslatorClientInterface() = default;
    virtual void handleNotice(NauUITranslatorProxyNotice const& notice) = 0; 
};

using NauUITranslatorClientInterfacePtr = std::shared_ptr<NauUITranslatorClientInterface>;