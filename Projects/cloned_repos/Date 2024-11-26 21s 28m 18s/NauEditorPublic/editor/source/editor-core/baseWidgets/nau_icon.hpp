// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file..
//
// Describes class that represents a set of pixmap associated with various states.
// Such pixmaps are used by Qt widgets to show an icon representing a particular action.

#pragma once

#include "nau/nau_editor_config.hpp"

#include <QIcon>


// ** NauIcon

class NAU_EDITOR_API NauIcon : public QIcon
{
public:
    NauIcon() = default;
    NauIcon(const QPixmap &pixmap);
    explicit NauIcon(const QString &fileName); 
};