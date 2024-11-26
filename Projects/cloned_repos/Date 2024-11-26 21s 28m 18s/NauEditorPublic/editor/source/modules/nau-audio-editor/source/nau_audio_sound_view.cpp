// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_audio_sound_view.hpp"
#include "nau_audio_playback_view.hpp"

#include "baseWidgets/nau_static_text_label.hpp"
#include "themes/nau_theme.hpp"

#include <QPainterPath>
#include <QTime>


// ** NauAudioSourceTimeView

NauAudioSourceTimeView::NauAudioSourceTimeView(NauWidget* parent)
    : NauWidget(parent)
    , m_position(new NauStaticTextLabel(""))
    , m_duration(new NauStaticTextLabel(""))
{
    auto layout = new NauLayoutVertical(this);
    auto layoutWidgets = new NauLayoutHorizontal();
    layout->addSpacing(5);
    layout->addLayout(layoutWidgets);

    const auto prepareWidget = [layoutWidgets](NauStaticTextLabel* label) {
        label->setFont(Nau::Theme::current().fontTitleBarMenuItem());
        label->setColor(ColorOff);
        layoutWidgets->addWidget(label);
    };

    prepareWidget(m_position);
    layoutWidgets->addSpacing(7);
    prepareWidget(m_duration);
}

void NauAudioSourceTimeView::setTime(const NauAudioCursorInfo& info)
{
    m_duration->setText(millisecondsToTime(info.duration));
    m_position->setText(millisecondsToTime(info.position));
    m_position->setColor((info.position == std::chrono::milliseconds(0))
        || (info.position == info.duration) ? ColorOff : ColorOn);
    update();
}

QString NauAudioSourceTimeView::millisecondsToTime(std::chrono::milliseconds msTotal)
{
    const auto duration = msTotal.count();
    const auto ms = duration % 999;
    const auto ss = (duration / 1000) % 60;
    const auto mm = (duration / 1000 / 60) % (60 * 60);
    [[maybe_unused]] const auto hh = (duration / 1000 / 60 / 60) % (60 * 60 * 60);   // Currently, we only show minutes

    return QTime(0, mm, ss, ms).toString("mm:ss.zzz");
}


// ** NauAudioSoundView

NauAudioSoundView::NauAudioSoundView(nau::audio::AudioAssetPtr asset, QWidget* parent, const std::string namePrefix)
    : NauWidget(parent)
    , m_asset(asset)
    , m_playbackView(new NauAudioPlaybackView(m_asset, this))
    , m_timeView(new NauAudioSourceTimeView(this))
{
    m_asset->subscribe(this, [this] {
        updateView();
    });

    setFixedHeight(Height);

    auto layoutMain = new NauLayoutVertical(this);
    layoutMain->setContentsMargins(8, 8, 8, 8);

    auto widgetTop = new NauWidget(this);
    auto layoutTop = new NauLayoutHorizontal(widgetTop);
    layoutMain->addWidget(widgetTop);

    // Name
    const QFileInfo info(asset->name().c_str());
    auto labelName = new NauStaticTextLabel(std::format("   {} {}", namePrefix, info.fileName().toStdString()).c_str(), this);
    labelName->setFixedHeight(24);
    labelName->setFont(Nau::Theme::current().fontLoggerLevel());
    auto layoutName = new NauLayoutVertical;
    layoutName->addSpacing(5);
    layoutName->addWidget(labelName);
    layoutTop->addLayout(layoutName);
    layoutTop->addStretch(1);

    // Time
    layoutTop->addWidget(m_timeView);

    // Close button
    auto buttonClose = new NauPushButton(this);
    buttonClose->setStyleSheet("QToolButton { background-color: transparent }");
    buttonClose->setIcon(QIcon(":/audio/icons/audio/audio-container-close.png"));
    buttonClose->setIconSize(QSize(24, 24));
    buttonClose->setFlat(true);
    layoutTop->addWidget(buttonClose);

    connect(buttonClose, &NauPushButton::pressed, [this] {
        emit eventDelete(m_asset);
    });

    // Playback panel
    m_playbackView->setFixedHeight(28);
    layoutMain->addWidget(m_playbackView);

    // Time updates
    connect(m_playbackView, &NauAudioPlaybackView::eventCursorChanged, this, &NauAudioSoundView::updateView);
    updateView();
}

void NauAudioSoundView::paintEvent(QPaintEvent* event)
{
    NauPainter painter(this);

    painter.setRenderHint(QPainter::Antialiasing);
    
    QPainterPath path;
    path.addRoundedRect(rect(), 4, 4);   
    painter.fillPath(path, ColorBackground);
}

void NauAudioSoundView::updateView()
{
    m_timeView->setTime(m_playbackView->cursor());
    update();
}

NauAudioSoundView::~NauAudioSoundView()
{
    m_asset->unsubscribe(this);
}
