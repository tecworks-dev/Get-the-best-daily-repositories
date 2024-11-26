// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Asset preview provider

#pragma once

#include "nau/nau_editor_config.hpp"
#include <QIcon>
#include <QString>


// ** NauAssetPreview

class NAU_EDITOR_API NauAssetPreview
{
public:
    NauAssetPreview() = delete;

    static bool hasPreview(const QString& assetPath);
    static QIcon assetPreview(const QString& assetPath);
};
