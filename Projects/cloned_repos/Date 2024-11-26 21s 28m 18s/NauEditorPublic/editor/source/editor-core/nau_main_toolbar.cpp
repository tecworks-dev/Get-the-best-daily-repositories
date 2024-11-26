// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_main_toolbar.hpp"
#include "nau_assert.hpp"
#include "themes/nau_theme.hpp"
#include "nau_icon.hpp"


// ** NauMainToolbar

NauMainToolbar::NauMainToolbar(QWidget* parent)
    : NauToolbarBase(parent)
{
    addSection(NauToolbarSection::Left);  // Empty section for proper alignment

    // Play mode & export
    auto centerSection = addSection(NauToolbarSection::Center);
    m_buttonPlay = centerSection->addButton(Nau::Theme::current().iconPlay(), tr("Launch this scene in play mode"), this, &NauMainToolbar::handlePlay);
    m_buttonPause = centerSection->addButton(Nau::Theme::current().iconPause(), tr("Pause play mode"), this, &NauMainToolbar::handlePause);
    m_buttonStop = centerSection->addButton(Nau::Theme::current().iconStop(), tr("Stop play mode"), this, &NauMainToolbar::handleStop);
    centerSection->addSeparator();
    m_buttonExport = centerSection->addButton(Nau::Theme::current().iconBuildSettings(), tr("Open build settings"), this, &NauMainToolbar::eventExport);

    // Undo/redo
    auto rightSection = addSection(NauToolbarSection::Right);
    rightSection->addButton(Nau::Theme::current().iconUndo(), tr("Undo"), this, &NauMainToolbar::eventUndo);
    rightSection->addButton(Nau::Theme::current().iconRedo(), tr("Redo"), this, &NauMainToolbar::eventRedo);
    m_buttonHistory = rightSection->addButton(Nau::Theme::current().iconHistory(), tr("History"), [this] {
        NED_ASSERT(m_buttonHistory);
        emit eventHistory(*m_buttonHistory);
    });
    rightSection->addSeparator();
    auto buttonSettings = rightSection->addButton(Nau::Theme::current().iconSettings(), tr("Settings"), this, &NauMainToolbar::eventSettings);

    // Default state
    m_buttonPause->setCheckable(true);
    m_buttonPause->hide();
    m_buttonStop->setEnabled(false);
    buttonSettings->setDisabled(true);  // TODO: add settings
}

void NauMainToolbar::handleLaunchState(bool started)
{
    m_buttonPlay->setEnabled(!started);
    m_buttonExport->setEnabled(!started);
}

void NauMainToolbar::handlePlay()
{
    m_buttonPlay->hide();
    m_buttonPause->show();
    m_buttonStop->setEnabled(true);
    emit eventPlay();
}

void NauMainToolbar::handlePause()
{
    emit eventPause(m_buttonPause->isChecked());
}

void NauMainToolbar::handleStop()
{
    m_buttonPlay->show();
    m_buttonPause->hide();
    m_buttonPause->setChecked(false);
    m_buttonStop->setEnabled(false);
    emit eventStop();
}