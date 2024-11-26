// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A checkbox is used in the filters.

#pragma once

#include "baseWidgets/nau_widget.hpp"


// ** NauFilterCheckBox

class NAU_EDITOR_API NauFilterCheckBox : public NauCheckBox
{
public:
    NauFilterCheckBox(const QString& text, QWidget* parent);

protected:
    // User can't explicitly set to the partially checked state.
    virtual void nextCheckState() override;
};
