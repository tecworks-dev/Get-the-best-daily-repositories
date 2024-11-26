// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Translator between Usd scene events and translator clients

#pragma once

#include "nau/nau_usd_asset_editor_common_config.hpp"

#include "nau_usd_scene_ui_translator_client.hpp"
#include "usd_proxy/usd_stage_watcher.h"

#include <memory>


// ** NauUsdSceneUITranslator

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdSceneUITranslator
{
public:
    friend class NauUITranslatorNotificationBlock;
    friend class NauUITranslatorNotificationAccumulator;

    NauUsdSceneUITranslator(const PXR_NS::UsdStageRefPtr& stage);
    ~NauUsdSceneUITranslator() = default;

    void addClient(const NauUITranslatorClientInterfacePtr& client);

private:
    void blockNotifications(bool block);
    void accumulateNotifications(bool accumulate);

    std::vector<pxr::UsdNotice::ObjectsChanged> recomposeEvents(const std::vector<pxr::UsdNotice::ObjectsChanged>& events);

private:
    bool m_blockNotifications = false;
    bool m_accmulateEvents = false;
    std::vector<NauUITranslatorProxyNotice> m_accumulatedEvents;

    std::vector<NauUITranslatorClientInterfacePtr> m_clients;

    std::unique_ptr<UsdProxy::StageObjectChangedWatcher> m_stageWatcher;
};


// ** NauUITranslatorBlock

class NAU_USD_ASSET_EDITOR_COMMON_API NauUITranslatorNotificationBlock
{
public:
    NauUITranslatorNotificationBlock(NauUsdSceneUITranslator* translator);
    ~NauUITranslatorNotificationBlock();

private:
    NauUsdSceneUITranslator* m_translator;
};


// ** NauUITranslatorNotificationAccumulator

class NAU_USD_ASSET_EDITOR_COMMON_API NauUITranslatorNotificationAccumulator
{
public:
    NauUITranslatorNotificationAccumulator(NauUsdSceneUITranslator* translator);
    ~NauUITranslatorNotificationAccumulator();

private:
    NauUsdSceneUITranslator* m_translator;
};