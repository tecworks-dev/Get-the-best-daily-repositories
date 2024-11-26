// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_spoiler.hpp"

#include "nau/audio/audio_engine.hpp"


// ** NauAudioContainerSourcesView

class NauAudioContainerSourcesView : public NauWidget
{
    Q_OBJECT

public:
    NauAudioContainerSourcesView(nau::audio::AudioAssetContainerPtr container, QWidget* parent);
    ~NauAudioContainerSourcesView();

signals:
    void eventChanged();

private:
    void updateView();
    bool checkRecursion(const nau::audio::AudioAssetPtr asset) const;
    
    nau::audio::AudioAssetContainerPtr findContainer(const std::string& name) const;
    nau::audio::AudioAssetPtr findAsset(const std::string& name) const;

private:
    nau::audio::AudioAssetContainerPtr  m_container;

    NauSpoiler*               m_spoiler;  // TODO: need to fix nested spoilers
    NauScrollWidgetVertical*  m_sources;  // Remove once spoilers are fixed
};
