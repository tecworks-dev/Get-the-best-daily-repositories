// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_audio_container_sources_view.hpp"
#include "nau_audio_sound_view.hpp"
#include "nau/inspector/nau_usd_inspector_widgets.hpp"

#include "baseWidgets/nau_static_text_label.hpp"
#include "themes/nau_theme.hpp"
#include "nau_log.hpp"

#include "nau/audio/audio_service.hpp"
#include "nau/shared/file_system.h"


// ** NauAudioContainerSourcesView

NauAudioContainerSourcesView::NauAudioContainerSourcesView(nau::audio::AudioAssetContainerPtr container, QWidget* parent)
    : NauWidget(parent)
    , m_container(container)
    , m_spoiler(new NauSpoiler(tr("Sources")))
    , m_sources(new NauScrollWidgetVertical(this))
{
    auto layoutMain = new NauLayoutVertical(this);

    auto assetSelector = new NauInspectorUSDResourceComboBox(NauInspectorUSDResourceComboBox::ShowFileNameMode, this);
    assetSelector->setFixedHeight(36);
    assetSelector->setPlaceholderText(tr("Audio source"));
    assetSelector->setAssetTypes({ NauEditorFileType::RawAudio, NauEditorFileType::AudioContainer });
    layoutMain->addWidget(assetSelector);

    connect(assetSelector, QOverload<const QString&>::of(&NauInspectorUSDResourceComboBox::eventSelectionChanged), 
        [this, assetSelector](const QString& itemText)
    {
        auto& assetDb = nau::getServiceProvider().get<nau::IAssetDB>();
        const auto uid = assetSelector->getCurrentUid();
        const auto USDPath = assetDb.findAssetMetaInfoByUid(uid).nausdPath;
        std::filesystem::path sourcePath = nau::FileSystemExtensions::resolveToNativePathContentFolder(USDPath.c_str());
        const auto asset = findAsset(sourcePath.replace_extension().string());

        if (!asset) {
            NED_DEBUG("Trying to use unregistered audio asset");
            return;
        }

        // Self check
        if (asset->name() == m_container->name()) {
            NED_ERROR("Can't include a container inside itself!");
            return;
        }

        // Recursion check
        if (checkRecursion(asset)) {
            NED_ERROR("Can't include this container because it will lead to recursion!");
            return;
        }

        // Empty check
        // TODO: temporary fix, fix properly in engine!
        if (!asset->instantiate()) {
            return;
        }

        // Add
        m_container->add(asset);
        updateView();
    });
    
    m_spoiler->hide();  // TODO: Fix nested spoilers
    m_sources->setFixedHeight(300);  // TODO: Figure out why it is not resizing
    layoutMain->addSpacing(12);
    layoutMain->addWidget(m_sources);

    updateView();
}

NauAudioContainerSourcesView::~NauAudioContainerSourcesView() = default;

void NauAudioContainerSourcesView::updateView()
{
    m_sources->layout()->clear();

    if (m_container->empty()) {  // Show empty view
        auto emptyLabel = new NauStaticTextLabel(tr("Add sources by pressing \"+\""), this);
        emptyLabel->setFont(Nau::Theme::current().fontPrimaryButton());
        m_sources->addWidget(emptyLabel);
    } else {  // Show sources
        int number = 1;
        for (auto asset : *m_container) {
            auto assetView = new NauAudioSoundView(asset, this, std::format("{}.", number++));
            m_sources->layout()->addWidget(assetView);
            m_sources->layout()->addSpacing(4);
            connect(assetView, &NauAudioSoundView::eventDelete, [this](nau::audio::AudioAssetPtr asset) {
                m_container->remove(asset);
                updateView();
            });
        }
    }

    m_sources->layout()->addStretch(1);
    m_sources->updateGeometry();
    update();
}

bool NauAudioContainerSourcesView::checkRecursion(const nau::audio::AudioAssetPtr asset) const
{
    auto container = findContainer(asset->name().c_str());
    if (!container) {
        return false;
    }

    for (const auto asset : *container) {
        if (m_container->name() == asset->name()) {
            return true;
        }
        if (checkRecursion(asset)) {
            return true;
        }
    }
    return false;
}

nau::audio::AudioAssetContainerPtr NauAudioContainerSourcesView::findContainer(const std::string& name) const
{
    auto& engine = nau::getServiceProvider().get<nau::audio::AudioService>().engine();
    const auto containers = engine.containerAssets();
    const auto itContainer = std::find_if(containers.begin(), containers.end(), [name](nau::audio::AudioAssetContainerPtr container) {
        return std::filesystem::path(container->name().c_str()) == std::filesystem::path(name);
    });
    return itContainer != containers.end() ? *itContainer : nullptr;
}

nau::audio::AudioAssetPtr NauAudioContainerSourcesView::findAsset(const std::string& name) const
{
    auto& engine = nau::getServiceProvider().get<nau::audio::AudioService>().engine();
    const auto assets = engine.assets();
    const auto itAsset = std::find_if(assets.begin(), assets.end(), [name](nau::audio::AudioAssetPtr _asset) {
        return std::filesystem::path(_asset->name().c_str()) == std::filesystem::path(name);
    });
    return itAsset != assets.end() ? *itAsset : nullptr;
}