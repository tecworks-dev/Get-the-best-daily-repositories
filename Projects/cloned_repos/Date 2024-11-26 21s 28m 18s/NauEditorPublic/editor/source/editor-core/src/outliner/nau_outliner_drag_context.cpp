// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/outliner/nau_outliner_drag_context.hpp"

#include <QTextStream>
#include <QBuffer>


// ** NauOutlinerDragContext

NauOutlinerDragContext::NauOutlinerDragContext(std::vector<QString> guids)
    : m_guids(std::move(guids))
{
}

void NauOutlinerDragContext::toMimeData(QMimeData& mimeData) const
{
    QBuffer buffer;
    buffer.open(QIODevice::WriteOnly);

    QTextStream stream(&buffer);

    for (auto const& guid : m_guids) {
        stream << guid << Qt::endl;
    }

    mimeData.setData(mimeType(), buffer.data());
}

const std::vector<QString>& NauOutlinerDragContext::guids() const
{
    return m_guids;
}

std::optional<NauOutlinerDragContext> NauOutlinerDragContext::fromMimeData(const QMimeData& mimeData)
{
    if (!mimeData.hasFormat(mimeType())) {
        return std::nullopt;
    }

    QTextStream stream(mimeData.data(mimeType()));
    std::vector<QString> result;

    while (!stream.atEnd()) {
        result.emplace_back(stream.readLine());
    }

    return NauOutlinerDragContext{result};
}

QString NauOutlinerDragContext::mimeType()
{
    return QStringLiteral("application/9c74a648-27ef-4c5c-b49d-fe6a53459d55");
}