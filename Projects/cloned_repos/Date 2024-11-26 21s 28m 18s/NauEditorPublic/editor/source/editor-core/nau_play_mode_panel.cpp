// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include <nau_play_mode_panel.hpp>
#include "themes/nau_theme.hpp"


// ** NauPlayModePanel

NauPlayModePanel::NauPlayModePanel(QWidget* parent)
    : NauWidget(parent)
    , m_layout(new NauLayoutHorizontal(this))
    , m_playButton(new NauToolButton())
    , m_stopButton(new NauToolButton())
    , m_pauseButton(new NauToolButton())
    , m_startLaunchButton(new NauToolButton())
    , m_stopLaunchButton(new NauToolButton())
{
    // Setup Layout
    m_layout->setSpacing(0);
    m_layout->setAlignment(Qt::AlignLeft);

    const auto setupButtonStyle = [](NauToolButton* button) {
        button->setStyleSheet("QToolButton { margin: 0px; border-radius: 2px; }"
                              "QToolButton:hover{ background-color: #1B1B1B; }"
                              "QToolButton:checked{ background-color: #3143E5; }");
    };

    const auto setupButton = [&setupButtonStyle](NauToolButton* button, const QString& name, const QIcon& icon, const QString& tooltip) {
        button->setObjectName(name + "Button");
        button->setIcon(icon);
        button->setToolTip(tooltip);
        button->setAutoRaise(true);
        button->setEnabled(false);
        button->setContentsMargins(8, 8, 8, 8);
        button->setFixedSize(32, 32);

        setupButtonStyle(button);
    };

    setupButton(m_playButton       , "play"       , Nau::Theme::current().iconPlay()         , tr("Launch this scene in play mode"));
    setupButton(m_stopButton       , "stop"       , Nau::Theme::current().iconStop()         , tr("Stop play mode"));
    setupButton(m_pauseButton      , "pause"      , Nau::Theme::current().iconPause()        , tr("Pause play mode"));
    setupButton(m_startLaunchButton, "startLaunch", Nau::Theme::current().iconBuildSettings(), tr("Open build settings"));
    setupButton(m_stopLaunchButton , "stopLaunch" , Nau::Theme::current().iconStop()         , tr("Stop building"));

    m_pauseButton->setCheckable(true);

    connect(m_playButton, &NauToolButton::clicked, this, &NauPlayModePanel::buttonPlayPressed);
    connect(m_stopButton, &NauToolButton::clicked, this, &NauPlayModePanel::buttonStopPressed);
    connect(m_pauseButton, &NauToolButton::toggled, this, &NauPlayModePanel::buttonPausePressed);
    connect(m_startLaunchButton, &NauToolButton::clicked, this, &NauPlayModePanel::buttonStartLaunchPressed);
    connect(m_stopLaunchButton, &NauToolButton::clicked, this, &NauPlayModePanel::buttonStopLaunchPressed);

    // Add buttons on layout
    m_layout->addWidget(m_playButton);
    m_layout->addSpacing(8);
    m_layout->addWidget(m_stopButton);
    m_layout->addSpacing(8);
    m_layout->addWidget(m_pauseButton);

    constexpr QColor lineColor(52, 52, 52);
    constexpr int lineWidth(2);

    auto splitLine = new NauLineWidget(lineColor, lineWidth, Qt::Orientation::Vertical, this);
    splitLine->setFixedSize(32, 32);
    m_layout->addWidget(splitLine);

    m_layout->addWidget(m_startLaunchButton);
    m_layout->addSpacing(8);
    m_layout->addWidget(m_stopLaunchButton);

    // TODO: Temporarily disabled until NED-548 is completed
    m_playButton->setDisabled(true);
    m_stopButton->setDisabled(true);
    m_pauseButton->setDisabled(true);
}

void NauPlayModePanel::buttonPlayPressed() const
{
    m_playButton->setEnabled(false);
    m_stopButton->setEnabled(true);
    m_pauseButton->setEnabled(true);
    m_startLaunchButton->setEnabled(false);
    m_stopLaunchButton->setEnabled(false);

    emit playSimulation();
}

void NauPlayModePanel::buttonStopPressed() const
{
    m_pauseButton->setChecked(false);

    m_stopButton->setEnabled(false);
    m_pauseButton->setEnabled(false);

    emit stopSimulation();
}

void NauPlayModePanel::buttonPausePressed(const bool checked)
{
    emit pauseSimulation(m_pauseButton->isChecked());
}

void NauPlayModePanel::buttonStartLaunchPressed() const
{
    m_playButton->setEnabled(false);
    m_startLaunchButton->setEnabled(false);
    m_stopLaunchButton->setEnabled(true);

    emit startLauchProject();
}

void NauPlayModePanel::buttonStopLaunchPressed() const
{
    m_playButton->setEnabled(true);
    m_startLaunchButton->setEnabled(true);
    m_stopLaunchButton->setEnabled(false);

    emit stopLauchProject();
}

void NauPlayModePanel::buttonPlayRequested() const
{
    m_playButton->setEnabled(true);
    m_startLaunchButton->setEnabled(true);
    m_stopLaunchButton->setEnabled(false);
}

void NauPlayModePanel::buttonLaunchRequested() const
{
    m_playButton->setEnabled(true);
    m_startLaunchButton->setEnabled(true);
    m_stopLaunchButton->setEnabled(false);
}

void NauPlayModePanel::setLaunchButtonsVisible(bool visible) const
{
    m_startLaunchButton->setVisible(visible);
    m_stopLaunchButton->setVisible(visible);
}
