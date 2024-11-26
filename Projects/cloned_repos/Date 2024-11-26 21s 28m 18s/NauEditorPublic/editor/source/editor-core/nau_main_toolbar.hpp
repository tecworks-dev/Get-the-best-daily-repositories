// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Main toolbar

#pragma once

#include "baseWidgets/nau_toolbar.hpp"


// ** NauMainToolbar

class NAU_EDITOR_API NauMainToolbar : public NauToolbarBase
{
    Q_OBJECT

public:
    NauMainToolbar(QWidget* parent);

signals:
    void eventPlay();
    void eventPause(bool pause);
    void eventStop();
    void eventExport();

    void eventUndo();
    void eventRedo();
    void eventHistory(const NauToolButton& button);
    void eventSettings();

public slots:
    void handleLaunchState(bool started);
    void handleStop();

private slots:
    void handlePlay();
    void handlePause();

private:
    NauToolButton* m_buttonPlay         = nullptr;
    NauToolButton* m_buttonPause        = nullptr;
    NauToolButton* m_buttonStop         = nullptr;
    NauToolButton* m_buttonExport       = nullptr;
    NauToolButton* m_buttonHistory      = nullptr;
};
