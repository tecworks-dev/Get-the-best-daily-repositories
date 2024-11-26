// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "baseWidgets/nau_widget.hpp"

#include "nau/audio/audio_engine.hpp"

#include <QTimer>


// ** NauPlayButton

class NauAudioPlayButton : public NauWidget
{
    Q_OBJECT

public:

    enum State
    {
        Stopped = 0,
        Playing,
        Paused
    };

    NauAudioPlayButton(NauWidget* parent);

    void setState(State state);
    State state() const;

signals:
    void eventPressed();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    // Style
    inline static constexpr int IconSize  = 16;
    inline static constexpr int Margin    = 4;

    // Icons
    const QPixmap m_iconPlay;
    const QPixmap m_iconPause;

    // State
    State  m_state;
    bool   m_hovered;
};


// ** NauAudioCursorInfo

struct NauAudioCursorInfo
{
    std::chrono::milliseconds position;
    std::chrono::milliseconds duration;
};

// ** NauAudioPlaybackView

class NauAudioPlaybackView : public NauWidget
{
    Q_OBJECT

public:
    NauAudioPlaybackView(nau::audio::AudioAssetPtr asset, QWidget* parent);
    ~NauAudioPlaybackView();
    
    NauAudioCursorInfo cursor() const;

signals:
    void eventCursorChanged();
    
private slots:
    void startUpdates();
    void stopUpdates();

private:
    void updateView();
    void reinstantiate();

private:
    nau::audio::AudioAssetPtr   m_asset;
    nau::audio::AudioSourcePtr  m_source;
    std::chrono::milliseconds   m_duration;

    NauAudioPlayButton*  m_play;
    QTimer*              m_cursorPollTimer;
};
