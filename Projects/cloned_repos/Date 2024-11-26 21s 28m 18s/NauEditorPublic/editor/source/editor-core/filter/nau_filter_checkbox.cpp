// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_filter_checkbox.hpp"


// ** NauFilterCheckBox

NauFilterCheckBox::NauFilterCheckBox(const QString& text, QWidget* parent)
    : NauCheckBox(text, parent)
{
}

void NauFilterCheckBox::nextCheckState()
{
    setCheckState(checkState() == Qt::CheckState::Checked ? Qt::CheckState::Unchecked : Qt::CheckState::Checked);
}
