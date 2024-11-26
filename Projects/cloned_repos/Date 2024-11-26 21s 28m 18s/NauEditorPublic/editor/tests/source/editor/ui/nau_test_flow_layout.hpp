// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include <QObject>
#include <QWidget>
#include "nau_flow_layout.hpp"

class NauFlowLayoutUITests : public QObject
{

    Q_OBJECT

private slots:
    void getMarginFromNewLayout();
    void getHorizontalSpacingFromNewLayout();
    void getVerticalSpacingFromNewLayout();
    void checkExpandingDirectionsFromNewLayout();
    void checkNewLayoutHasNoWidgets();
    void addWidgetToLayoutTest();
};