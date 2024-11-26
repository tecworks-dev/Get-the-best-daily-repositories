// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_audio_container_view.hpp"
#include "nau_audio_container_sources_view.hpp"
#include "nau_audio_playback_view.hpp"

#include "nau/inspector/nau_inspector.hpp"
#include "baseWidgets/nau_static_text_label.hpp"
#include "baseWidgets/nau_buttons.hpp"
#include "nau_log.hpp"
#include "themes/nau_theme.hpp"

#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/sdf/assetPath.h"
#include "usd_proxy/usd_proxy.h"


// ** NauAudioContainerView

NauAudioContainerView::NauAudioContainerView(nau::audio::AudioAssetContainerPtr asset, QWidget* parent)
    : NauWidget(parent)
    , m_container(asset)
    , m_kindDropdown(new NauComboBox(this))
    , m_header(new NauInspectorPageHeader("", ""))
{
    // Header
    auto layout = new NauLayoutVertical(this);
    m_header->changeTitle(QFileInfo(asset->name().c_str()).baseName().toStdString());
    m_header->changeIcon(":/Inspector/icons/audio/audio-container-inspector-header.png");
    m_header->findChild<NauPrimaryButton*>()->hide();  // Temp hack
    layout->addWidget(m_header);

    // Playback view
    auto playbackView = new NauAudioPlaybackView(m_container, this);
    playbackView->setFixedHeight(40);
    layout->addWidget(playbackView);

    // Separator
    auto separator = new QFrame;
    separator->setStyleSheet(QString("background-color: #141414;"));
    separator->setFrameShape(QFrame::HLine);
    separator->setFixedHeight(1);
    layout->addWidget(separator);
    layout->addSpacing(12);

    // Playback kind
    auto kindWidget = new NauWidget(this);
    auto kindLayout = new NauLayoutHorizontal(kindWidget);
    layout->addWidget(kindWidget);

    // Playback kind: label
    auto kindLabel = new NauStaticTextLabel(tr("Playback type"), this);
    kindLabel->setFont(Nau::Theme::current().fontTitleBarMenuItem());
    kindLabel->setColor(ColorSecondary);
    kindLayout->addWidget(kindLabel);

    // Playback kind: dropdown
    const auto items = nau::EnumTraits<nau::audio::AudioContainerKind>().getStrValues();
    for (const auto item : items) {
        m_kindDropdown->addItem(std::string(item).c_str());
    }
    kindLayout->addWidget(m_kindDropdown);
    connect(m_kindDropdown, &NauComboBox::currentTextChanged, [this](const QString& currentSelection) {
        if (const auto kind = nau::EnumTraits<nau::audio::AudioContainerKind>().parse(currentSelection.toStdString())) {
            m_container->setKind(*kind);
        }
    });

    // Sources
    auto sourcesView = new NauAudioContainerSourcesView(m_container, this);
    layout->addSpacing(10);
    layout->addWidget(sourcesView);

    // Subscribe to container updates
    m_container->subscribe(this, [this] {
        onContainerChange();
    });

    updateView();
}

NauAudioContainerView::~NauAudioContainerView()
{
    m_container->unsubscribe(this);
}

void NauAudioContainerView::onContainerChange()
{
    emit eventContainerChanged();
    updateView();
}

void NauAudioContainerView::updateView()
{
    // Container kind
    QSignalBlocker blocker(m_kindDropdown);
    const auto kind = nau::EnumTraits<nau::audio::AudioContainerKind>().toString(m_container->kind());
    m_kindDropdown->setCurrentText(std::string(kind).c_str());

    if (m_container->empty()) {
        return;
    }

    const std::chrono::milliseconds duration = m_container->instantiate()->duration();
    m_header->changeSubtitle(QString("%1 s").arg(duration.count() / 1000).toStdString());
    update();
}
