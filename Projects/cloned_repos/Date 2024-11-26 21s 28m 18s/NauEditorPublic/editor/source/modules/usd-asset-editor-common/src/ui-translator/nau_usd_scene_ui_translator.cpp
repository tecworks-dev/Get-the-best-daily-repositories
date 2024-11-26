// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/ui-translator/nau_usd_scene_ui_translator.hpp"


// ** NauUITranslatorNotificationBlock

NauUITranslatorNotificationBlock::NauUITranslatorNotificationBlock(NauUsdSceneUITranslator* translator)
    : m_translator(translator)
{
    m_translator->blockNotifications(true);
}

NauUITranslatorNotificationBlock::~NauUITranslatorNotificationBlock()
{
    m_translator->blockNotifications(false);
}


// ** NauUITranslatorNotificationAccumulator

NauUITranslatorNotificationAccumulator::NauUITranslatorNotificationAccumulator(NauUsdSceneUITranslator* translator)
    : m_translator(translator)
{
    m_translator->accumulateNotifications(true);
}

NauUITranslatorNotificationAccumulator::~NauUITranslatorNotificationAccumulator()
{
    m_translator->accumulateNotifications(false);
}


// ** NauUsdSceneUITranslator

NauUsdSceneUITranslator::NauUsdSceneUITranslator(const PXR_NS::UsdStageRefPtr& stage)
{
    m_stageWatcher = std::make_unique<UsdProxy::StageObjectChangedWatcher>(stage, [this](pxr::UsdNotice::ObjectsChanged const& notice) {
        if (m_blockNotifications) {
            return;
        }

        NauUITranslatorProxyNotice proxyNotice(notice);
        if (m_accmulateEvents) {
            m_accumulatedEvents.push_back(proxyNotice);
            return;
        }

        for (const auto& client : m_clients) {
            client->handleNotice(proxyNotice);
        }
    });
}

void NauUsdSceneUITranslator::addClient(const NauUITranslatorClientInterfacePtr& client)
{
    m_clients.push_back(client);
}

void NauUsdSceneUITranslator::blockNotifications(bool block)
{
    m_blockNotifications = block;
}

void NauUsdSceneUITranslator::accumulateNotifications(bool accumulate)
{
    m_accmulateEvents = accumulate;
    if (accumulate) {
        return;
    }

    // Handle accumulated events
    for (const NauUITranslatorProxyNotice& notice : m_accumulatedEvents) {
        for (const auto& client : m_clients) {
            client->handleNotice(notice);
        }
    }
    m_accumulatedEvents.clear();
}

std::vector<pxr::UsdNotice::ObjectsChanged> NauUsdSceneUITranslator::recomposeEvents(const std::vector<pxr::UsdNotice::ObjectsChanged>& events)
{
    std::vector<pxr::UsdNotice::ObjectsChanged> recomposed;
    // TODO: implement events recomposing - delete duplicated events
    return recomposed;
}