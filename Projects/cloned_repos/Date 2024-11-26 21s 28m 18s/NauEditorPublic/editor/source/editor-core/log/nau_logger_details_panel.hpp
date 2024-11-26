// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A panel with details about selected log message.

#pragma once

#include "baseWidgets/nau_label.hpp"
#include "nau_shortcut_hub.hpp"
#include "baseWidgets/nau_widget.hpp"


// ** NauLoggerDetailsPanel

class NauLoggerDetailsPanel : public QScrollArea
{
    Q_OBJECT

public:
    NauLoggerDetailsPanel(NauShortcutHub* shortcutHub, NauWidget* parent = nullptr);
    void loadMessageInfo(const QModelIndex& index);

signals:
    void eventCloseRequested();

private:
    void createUi(NauShortcutHub* shortcutHub);
    void clearUi();
    void handleCopyMessageTextSelectionRequest();

private:
    NauLabel* m_titleIcon = nullptr;
    NauLabel* m_titleLevel = nullptr;
    NauLabel* m_titleSource = nullptr;
    NauLabel* m_labelLevel = nullptr;
    NauLabel* m_labelLogger = nullptr;
    NauLabel* m_labelDate = nullptr;
    NauLabel* m_labelMessage = nullptr;
    NauLabel* m_labelDetails = nullptr;

    NauWidget* m_tagsContainer = nullptr;
};
