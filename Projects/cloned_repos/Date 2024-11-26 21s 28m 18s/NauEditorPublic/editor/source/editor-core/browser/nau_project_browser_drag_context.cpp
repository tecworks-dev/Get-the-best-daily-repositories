// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_project_browser_drag_context.hpp"

#include <QTextStream>
#include <QBuffer>


// ** NauProjectBrowserDragContext

NauProjectBrowserDragContext::NauProjectBrowserDragContext(
    std::vector<std::pair<NauEditorFileType, QString>> assetDragList)
    : m_assetDragList(std::move(assetDragList))
{
}

void NauProjectBrowserDragContext::toMimeData(QMimeData& mimeData) const
{
    QBuffer buffer;
    buffer.open(QIODevice::WriteOnly);

    QTextStream stream(&buffer);

    for (auto const&[type, url] : m_assetDragList) {
        stream << static_cast<int>(type) << Qt::endl;
        stream << url << Qt::endl;
    }

    mimeData.setData(mimeType(), buffer.data());
}

std::optional<NauProjectBrowserDragContext> NauProjectBrowserDragContext::fromMimeData(const QMimeData& mimeData)
{
    if (!mimeData.hasFormat(mimeType())) {
        return std::nullopt;
    }

    QTextStream stream(mimeData.data(mimeType()));
    int type{};
    QString fileName;
    std::vector<std::pair<NauEditorFileType, QString>> result;

    while (!stream.atEnd())
    {
        stream >> type >> fileName;
        result.emplace_back(std::make_pair(static_cast<NauEditorFileType>(type), fileName));
    }

    return NauProjectBrowserDragContext{result};
}

QString NauProjectBrowserDragContext::mimeType()
{
    return QStringLiteral("application/88def878-f93b-4bc4-8932-dee216e88359");
}