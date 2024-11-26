// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_static_text_label.hpp"

#include "nau/audio/audio_engine.hpp"

#include "nau_audio_playback_view.hpp"


// ** NauAudioSourceTimeView

class NauAudioSourceTimeView : public NauWidget
{
    Q_OBJECT

public:
    NauAudioSourceTimeView(NauWidget* parent);

    void setTime(const NauAudioCursorInfo& info);

private:
    static QString millisecondsToTime(std::chrono::milliseconds ms);

private:
    inline static const NauColor ColorOn  = NauColor(255, 255, 255);
    inline static const NauColor ColorOff = NauColor(128, 128, 128);

    NauStaticTextLabel* m_position;
    NauStaticTextLabel* m_duration;
};


// ** NauAudioSoundView

class NauAudioSoundView : public NauWidget
{
    Q_OBJECT

public:
    NauAudioSoundView(nau::audio::AudioAssetPtr asset, QWidget* parent, const std::string namePrefix = "");
    ~NauAudioSoundView();

protected:
    void paintEvent(QPaintEvent* event) override;

signals:
    void eventPlay();
    void eventStop();
    void eventDelete(nau::audio::AudioAssetPtr sound);

private:
    void updateView();

private:
    inline static constexpr auto Height = 68;
    inline static constexpr auto ColorBackground = NauColor(34, 34, 34);

private:
    nau::audio::AudioAssetPtr   m_asset;
    NauAudioPlaybackView*       m_playbackView;
    NauAudioSourceTimeView*     m_timeView;
};
