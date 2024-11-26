// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Animation timeline parameters widget

#pragma once

#include "baseWidgets/nau_widget.hpp"


// ** NauTimelineModesWidget

class NauTimelineModesWidget final : public QTabBar
{
    Q_OBJECT
public:
    NauTimelineModesWidget(NauWidget* parent);

signals:
    void eventDopeSheetModeEnabled();
    void eventCurveModeEnabled();

protected:
    QSize tabSizeHint(int index) const override;
    void paintEvent(QPaintEvent* event) override;
};


// ** NauTimelineParameters

class NauTimelineParameters : public NauWidget
{
    Q_OBJECT

public:
    NauTimelineParameters(NauWidget* parent);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    NauDoubleSpinBox* m_step;
    NauSpinBox* m_duration;
};