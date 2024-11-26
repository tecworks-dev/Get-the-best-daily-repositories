// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include <baseWidgets/nau_widget.hpp>

#include "nau/audio/audio_engine.hpp"


// ** NauAudioContainerView

class NauInspectorPageHeader;

class NauAudioContainerView : public NauWidget
{
    Q_OBJECT

public:
    NauAudioContainerView(nau::audio::AudioAssetContainerPtr asset, QWidget* parent);
    ~NauAudioContainerView();

signals:
    void eventContainerChanged();

private:
    void onContainerChange();
    void updateView();

private:
    nau::audio::AudioAssetContainerPtr   m_container;

    NauComboBox*             m_kindDropdown;
    NauInspectorPageHeader*  m_header;

    inline static constexpr NauColor ColorSecondary = NauColor(128, 128, 128);
};
