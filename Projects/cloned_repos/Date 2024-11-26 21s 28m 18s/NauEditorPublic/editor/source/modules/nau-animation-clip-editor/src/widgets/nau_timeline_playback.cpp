// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/widgets/nau_timeline_playback.hpp"
#include "nau/widgets/nau_common_timeline_widgets.hpp"

#include "themes/nau_theme.hpp"
#include "nau_assert.hpp"


// ** NauTimelinePlayback

NauTimelinePlayback::NauTimelinePlayback(NauWidget* parent)
    : NauWidget(parent)
    , m_recordButton(new NauTimelineButton(this))
    , m_timelineStartButton(new NauToolButton(this))
    , m_previousFrameButton(new NauToolButton(this))
    , m_timelinePlayButton(new NauToolButton(this))
    , m_timelinePauseButton(new NauToolButton(this))
    , m_timelineStopButton(new NauToolButton(this))
    , m_nextFrameButton(new NauToolButton(this))
    , m_timelineEndButton(new NauToolButton(this))
    , m_time(new NauTimeEdit(this))
{
    constexpr QSize TIME_SIZE{ 72, 24 };
    constexpr QSize BUTTON_SIZE{ 32, 32 };

    const QString BUTTON_STYLE_SHEET = "background-color: 0#00000000";
    const std::vector<NauIcon> icons = Nau::Theme::current().iconsTimelinePlayback();

    const std::array buttons{
        m_timelineStartButton, m_previousFrameButton, m_timelinePlayButton,
        m_timelinePauseButton, m_timelineStopButton, m_nextFrameButton, m_timelineEndButton };

    NED_ASSERT(buttons.size() == icons.size());

    m_recordButton->setBackgroundColor(Nau::Theme::current().paletteTimelineRecord().color(NauPalette::Role::Background));
    m_recordButton->setStyleSheet(BUTTON_STYLE_SHEET);
    m_recordButton->setFixedSize(BUTTON_SIZE);
    m_recordButton->setDisabled(true);

    m_timelinePauseButton->hide();
    m_timelinePauseButton->setToolTip(tr("Pause"));
    m_timelineStopButton->hide();
    m_timelineStopButton->setToolTip(tr("Stop"));
    m_timelinePlayButton->setToolTip(tr("Play"));
    m_timelineStartButton->setToolTip(tr("To start"));
    m_previousFrameButton->setToolTip(tr("Previous frame"));
    m_nextFrameButton->setToolTip(tr("Next frame"));
    m_timelineEndButton->setToolTip(tr("To end"));

    m_time->setTime(QTime(0, 0, 0, 0));
    m_time->setDisplayFormat("mm:ss:zzz");
    m_time->setFixedSize(TIME_SIZE);
    m_time->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    auto* layout = new NauLayoutHorizontal;
    layout->setContentsMargins(16, 8, 16, 8);
    layout->setSpacing(4);
    layout->addWidget(m_recordButton);
    for (size_t index = 0, size = buttons.size(); index < size; ++index) {
        auto* button = buttons[index];
        button->setIcon(icons[index]);
        button->setStyleSheet(BUTTON_STYLE_SHEET);
        button->setFixedSize(BUTTON_SIZE);
        layout->addWidget(button);
    }
    layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding));
    layout->addWidget(m_time, 0, Qt::AlignRight);

    setLayout(layout);
    setFixedHeight(48);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    QObject::connect(m_timelinePlayButton, &QAbstractButton::pressed, [this]() {
        m_timelinePlayButton->hide();
        m_timelinePauseButton->show();
        m_timelineStopButton->show();
    });
    QObject::connect(m_timelinePauseButton, &QAbstractButton::pressed, [this]() {
        m_timelinePauseButton->hide();
        m_timelinePlayButton->show();
    });
    QObject::connect(m_timelineStopButton, &QAbstractButton::pressed, [this]() {
        m_timelinePauseButton->hide();
        m_timelineStopButton->hide();
        m_timelinePlayButton->show();
    });
}

NauTimelineButton& NauTimelinePlayback::recordButton() const noexcept
{
    NED_ASSERT(m_recordButton);
    return *m_recordButton;
}

NauToolButton& NauTimelinePlayback::timelineStartButton() const noexcept
{
    NED_ASSERT(m_timelineStartButton);
    return *m_timelineStartButton;
}

NauToolButton& NauTimelinePlayback::previousKeyButton() const noexcept
{
    NED_ASSERT(m_previousFrameButton);
    return *m_previousFrameButton;
}

NauToolButton& NauTimelinePlayback::timelinePlayButton() const noexcept
{
    NED_ASSERT(m_timelinePlayButton);
    return *m_timelinePlayButton;
}

NauToolButton& NauTimelinePlayback::timelineStopButton() const noexcept
{
    NED_ASSERT(m_timelineStopButton);
    return *m_timelineStopButton;
}

NauToolButton& NauTimelinePlayback::timelinePauseButton() const noexcept
{
    NED_ASSERT(m_timelinePauseButton);
    return *m_timelinePauseButton;
}

NauToolButton& NauTimelinePlayback::nextKeyButton() const noexcept
{
    NED_ASSERT(m_nextFrameButton);
    return *m_nextFrameButton;
}

NauToolButton& NauTimelinePlayback::timelineEndButton() const noexcept
{
    NED_ASSERT(m_timelineEndButton);
    return *m_timelineEndButton;
}

void NauTimelinePlayback::reset()
{
    m_timelineStopButton->pressed();
}

void NauTimelinePlayback::setCurrentTime(float time) noexcept
{
    QTime timeData{ 0, 0, 0, 0 };
    timeData = timeData.addMSecs(static_cast<int>(time * 1000.f));
    m_time->setTime(timeData);
}

void NauTimelinePlayback::paintEvent(QPaintEvent* event)
{
    const NauPalette palette = Nau::Theme::current().paletteTimelineTrackList();
    QPainter painter{ this };
    painter.fillRect(QRectF(0, 0, width(), height()), palette.color(NauPalette::Role::BackgroundHeader));
}

