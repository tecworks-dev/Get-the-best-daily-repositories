// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
// 
// Wrapper around CDockWidget from Advanced Docking System(ads).

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "DockWidget.h"


// ** NauDockWidget

class NAU_EDITOR_API NauDockWidget : public ads::CDockWidget
{
public:
    NauDockWidget(const QString& title, NauWidget* parent);
};
