// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
// 
// Utilities are used to perform drag&drop from/into/within the world outliner.

#pragma once

#include <QString>
#include <QMimeData>

#include <optional>
#include <string>
#include <vector>


// ** NauOutlinerDragContext
// A wrapper for encapsulating information about drag from the world outliner.

class NauOutlinerDragContext
{
public:
    NauOutlinerDragContext(std::vector<QString> objects);
    void toMimeData(QMimeData& mimeData) const;

    const std::vector<QString>& guids() const;

    static std::optional<NauOutlinerDragContext> fromMimeData(const QMimeData& mimeData);
    static QString mimeType();

private:
    const std::vector<QString> m_guids;
};