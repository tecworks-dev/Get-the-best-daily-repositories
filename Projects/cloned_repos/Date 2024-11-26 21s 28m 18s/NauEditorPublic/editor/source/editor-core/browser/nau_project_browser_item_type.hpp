// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Classification of files in the project browser.

#pragma once

#include "nau/nau_editor_config.hpp"

#include "nau/assets/nau_file_types.hpp"

#include <QString>
#include <QMap>


namespace Nau
{
    QString itemTypeToString(NauEditorFileType type);
};

// ** NauProjectBrowserItemTypeResolverInterface

class NAU_EDITOR_API NauProjectBrowserItemTypeResolverInterface
{
public:
    virtual ~NauProjectBrowserItemTypeResolverInterface() = default;

    struct ItemTypeDescription
    {
        NauEditorFileType type;
        QString shortName;
        QString description;
    };

    // Returns resolved information about type for the specified extension.
    // extension is trimmed, with no dots, maybe empty, e.g: completeExtension - "vromfs.bin", "dxp",
    // Impementations should return NauEditorFileType::Unrecognized if fails to resolve.
    virtual ItemTypeDescription resolve(const QString& filePath, const std::optional<std::string> primPath = std::nullopt) const = 0;
};

using NauProjectBrowserItemTypeResolverInterfacePtr = std::shared_ptr<NauProjectBrowserItemTypeResolverInterface>;

