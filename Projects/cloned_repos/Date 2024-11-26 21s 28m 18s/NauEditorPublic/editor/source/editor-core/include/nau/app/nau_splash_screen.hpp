// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor application startup splash screen

#pragma once

#include "nau/nau_editor_config.hpp"
#include "baseWidgets/nau_label.hpp"
#include "baseWidgets/nau_widget.hpp"
#include <QSplashScreen>


// ** NauEditorSplashScreen

class NAU_EDITOR_API NauEditorSplashScreen : public NauFrame
{
    Q_OBJECT
public:
    explicit NauEditorSplashScreen(const QString& projectName, int stepCount);

    // Display specified message on this splash, but keep progress unchanged.
    void showMessage(const QString& message);

    // Display specified message on this splash, and advance progress on specified step.
    void advance(const QString& message, int step = 1);

private:
    QString getProgressPercentage() const;

private:
    NauLabel* m_messageLabel = nullptr;
    NauProgressBar* m_progressBar = nullptr;
};
