// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_audio_playback_view.hpp"

#include "nau_log.hpp"
#include "nau_color.hpp"

#include <QRandomGenerator>


// ** NauPlayButton

NauAudioPlayButton::NauAudioPlayButton(NauWidget* parent)
    : NauWidget(parent)
    , m_iconPlay(":/audio/icons/audio/audio-play.png")
    , m_iconPause(":/audio/icons/audio/audio-pause.png")
    , m_state(State::Stopped)
    , m_hovered(false)
{
    setFixedSize(IconSize + Margin, IconSize + Margin);
}

void NauAudioPlayButton::setState(State state)
{
    if (m_state == state) return;

    m_state = state;
    update();
}

NauAudioPlayButton::State NauAudioPlayButton::state() const
{
    return m_state;
}

void NauAudioPlayButton::paintEvent(QPaintEvent* event)
{
    NauPainter painter(this);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    
    QPixmap icon = m_state == Playing ? m_iconPause : m_iconPlay;
    if (m_hovered) {
        Nau::paintPixmap(icon, Qt::white);
    }
    painter.drawPixmap(QRect(Margin, Margin, IconSize, IconSize), icon);
}

void NauAudioPlayButton::mouseReleaseEvent(QMouseEvent* event)
{
    emit eventPressed();
}

void NauAudioPlayButton::enterEvent(QEnterEvent* event)
{
    m_hovered = true;
    update();
}

void NauAudioPlayButton::leaveEvent(QEvent* event)
{
    m_hovered = false;
    update();
}


// ** NauAudioPlaybackView

NauAudioPlaybackView::NauAudioPlaybackView(nau::audio::AudioAssetPtr asset, QWidget* parent)
    : m_asset(asset)
    , m_source(nullptr)
    , m_duration(std::chrono::milliseconds(0))
    , m_play(new NauAudioPlayButton(this))
    , m_cursorPollTimer(new QTimer(this))
{
    m_asset->subscribe(this, [this] {
        reinstantiate();
    });

    auto layout = new QHBoxLayout(this);

    connect(m_play, &NauAudioPlayButton::eventPressed, [this] {
        if (!m_source) {
            NED_WARNING("Attempting to play an empty audio source.");
            return;
        }
        if (m_source->isAtEnd()) reinstantiate();
        if (const auto state = m_play->state(); state == NauAudioPlayButton::Playing) {
            m_source->pause();
            m_play->setState(NauAudioPlayButton::Paused);
            stopUpdates();
        } else {  // Paused or stopped
            m_source->play();
            m_play->setState(NauAudioPlayButton::Playing);
            startUpdates();
        }
    });

    m_cursorPollTimer->callOnTimeout([this] {
        if (!m_source) {
            NED_DEBUG("Trying to poll audio source playback when the source is empty!");
            return;
        }

        // Randomize the next update so the time intervals don't look chopped
        m_cursorPollTimer->setInterval(QRandomGenerator::global()->bounded(50, 150));
        emit eventCursorChanged();
    });

    reinstantiate();
}

NauAudioCursorInfo NauAudioPlaybackView::cursor() const
{
    return {
        .position = m_source ? m_source->position() : std::chrono::milliseconds(0),
        .duration = m_source ? m_duration : std::chrono::milliseconds(0)
    };
}

void NauAudioPlaybackView::updateView()
{
    update();
}

void NauAudioPlaybackView::startUpdates()
{
    m_cursorPollTimer->start();
}

void NauAudioPlaybackView::stopUpdates()
{
    m_cursorPollTimer->stop();
    emit eventCursorChanged();
}

void NauAudioPlaybackView::reinstantiate()
{
    m_source = m_asset->instantiate();

    if (!m_source) {
        emit eventCursorChanged();
        return;  // Empty container
    }

    m_duration = m_source->duration();

    m_source->setEndCallback([this] {
        m_play->setState(NauAudioPlayButton::Stopped);
        // This is coming from the audio thread and the timer needs to be stopped from the GUI thread
        QMetaObject::invokeMethod(this, "stopUpdates", Qt::AutoConnection);
    });

    m_play->setState(NauAudioPlayButton::Stopped);
    emit eventCursorChanged();
}

NauAudioPlaybackView::~NauAudioPlaybackView()
{
    m_asset->unsubscribe(this);
}
