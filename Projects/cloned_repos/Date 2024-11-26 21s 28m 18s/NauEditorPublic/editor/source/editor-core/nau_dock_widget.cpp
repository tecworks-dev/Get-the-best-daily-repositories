// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_dock_widget.hpp"

NauDockWidget::NauDockWidget(const QString& title, NauWidget* parent)
    : ads::CDockWidget(title, parent)
{
    setStyleSheet("background-color: #282828");

    // Do not leave this widget without explicitly setObjectName.
    // ADS by default sets object name with a title.
    // ObjectName will be used as a key in serialization of state 
    // the docking system which may lead to a problem with the localization.
    setObjectName("DockingCommonWidget");
}
