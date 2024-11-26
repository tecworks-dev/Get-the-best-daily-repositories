// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Animation timeline playback widget

#pragma once

#include "baseWidgets/nau_widget.hpp"


class NauTimelineButton;


// ** NauTimelinePlayback
// Use the Timeline Playback Controls to play the Timeline instance and to control the location of the Timeline Playhead.

class NauTimelinePlayback : public NauWidget
{
    Q_OBJECT

public:
    NauTimelinePlayback(NauWidget* parent);

    [[nodiscard]]
    NauTimelineButton& recordButton() const noexcept;
    [[nodiscard]]
    NauToolButton& timelineStartButton() const noexcept;
    [[nodiscard]]
    NauToolButton& previousKeyButton() const noexcept;
    [[nodiscard]]
    NauToolButton& timelinePlayButton() const noexcept;
    [[nodiscard]]
    NauToolButton& timelineStopButton() const noexcept;
    [[nodiscard]]
    NauToolButton& timelinePauseButton() const noexcept;
    [[nodiscard]]
    NauToolButton& nextKeyButton() const noexcept;
    [[nodiscard]]
    NauToolButton& timelineEndButton() const noexcept;

    void reset();
    void setCurrentTime(float time) noexcept;

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    NauTimelineButton* m_recordButton;
    NauToolButton* m_timelineStartButton;
    NauToolButton* m_previousFrameButton;
    NauToolButton* m_timelinePlayButton;
    NauToolButton* m_timelinePauseButton;
    NauToolButton* m_timelineStopButton;
    NauToolButton* m_nextFrameButton;
    NauToolButton* m_timelineEndButton;
    NauTimeEdit* m_time;
};