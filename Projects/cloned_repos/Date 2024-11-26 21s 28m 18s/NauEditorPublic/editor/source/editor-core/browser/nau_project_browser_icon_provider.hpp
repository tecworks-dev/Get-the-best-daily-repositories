// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A default icon provider for the project browser.

#pragma once

#include "nau_project_browser_item_type.hpp"

#include <QAbstractFileIconProvider>


// ** NauProjectBrowserIconProvider

class NAU_EDITOR_API NauProjectBrowserIconProvider : public QAbstractFileIconProvider
{
public:
    explicit NauProjectBrowserIconProvider(
        std::vector<std::shared_ptr<NauProjectBrowserItemTypeResolverInterface>> itemTypeResolvers);

    virtual QIcon icon(QAbstractFileIconProvider::IconType type) const override;
    virtual QIcon icon(const QFileInfo &info) const override;

private:
    std::vector<std::shared_ptr<NauProjectBrowserItemTypeResolverInterface>> m_itemTypeResolvers;
    const QIcon m_defaultDirIcon;
    const QIcon m_defaultFileIcon;

    QMap<NauEditorFileType, QIcon> m_iconsRepository;
};
