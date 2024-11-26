// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "nau/assets/nau_file_types.hpp"

#include <QMimeData>
#include <QModelIndexList>

#include <optional>
#include <vector>


// ** NauProjectBrowserDragContext
// A wrapper for encapsulating information about drag from the project browser.

class NauProjectBrowserDragContext
{
public:
    NauProjectBrowserDragContext(std::vector<std::pair<NauEditorFileType, QString>> assetDragList);
    void toMimeData(QMimeData& mimeData) const;

    const std::vector<std::pair<NauEditorFileType, QString>>& assetDragList() const {
        return m_assetDragList;
    }

    static std::optional<NauProjectBrowserDragContext> fromMimeData(const QMimeData& mimeData);
    static QString mimeType();

private:
    const std::vector<std::pair<NauEditorFileType, QString>> m_assetDragList;
};