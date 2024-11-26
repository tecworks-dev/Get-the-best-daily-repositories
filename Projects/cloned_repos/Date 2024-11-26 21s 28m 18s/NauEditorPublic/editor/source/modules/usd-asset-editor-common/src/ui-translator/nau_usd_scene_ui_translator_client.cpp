// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/ui-translator/nau_usd_scene_ui_translator_client.hpp"


// ** NauUITranslatorProxyNotice

NauUITranslatorProxyNotice::NauUITranslatorProxyNotice(pxr::UsdNotice::ObjectsChanged const& notice)
    : m_stage(notice.GetStage())
{
    for (const auto& path : notice.GetResyncedPaths()) {
        m_resyncedPaths.push_back(path);
    }

    for (const auto& path : notice.GetChangedInfoOnlyPaths()) {
        m_infoChangesPaths.push_back(path);
    }
}

const pxr::SdfPathVector& NauUITranslatorProxyNotice::resyncedPaths() const
{
    return m_resyncedPaths;
}

const pxr::SdfPathVector& NauUITranslatorProxyNotice::infoChanges() const
{
    return m_infoChangesPaths;
}

pxr::UsdStageWeakPtr NauUITranslatorProxyNotice::stage() const
{
    return m_stage;
}