// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_color.hpp"

#include <QBitmap>
#include <QPainter>


namespace Nau
{
    void paintPixmap(QPixmap& pixmap, const QColor& color)
    {
        QPainter painter(&pixmap);
        painter.setCompositionMode(QPainter::CompositionMode_SourceIn);
        painter.setBrush(color);
        painter.setPen(color);
        painter.drawRect(pixmap.rect());
    }

    QPixmap paintPixmap(const QString& path, const QColor& color)
    {
        QPixmap pixmap = QPixmap(path);
        paintPixmap(pixmap, color);
        return pixmap;
    }
    
    QPixmap paintPixmapCopy(QPixmap pixmap, const QColor& color)
    {
        paintPixmap(pixmap, color);
        return pixmap;
    }

    QIcon paintIcon(const QString& path, QColor color)
    {
        return QIcon(paintPixmap(path, color));
    }
}


// ** NauColor

NauColor::NauColor(Qt::GlobalColor color)
    : QColor(color)
{
}

NauColor::NauColor(QRgb rgb)
    : QColor(rgb)
{
}


// ** NauBrush

NauBrush::NauBrush(const QColor& color, Qt::BrushStyle bs)
    : QBrush(color, bs)
{
}


// ** NauPen

NauPen::NauPen(const NauColor& color)
    : QPen(color)
{
}

NauPen::NauPen(const NauBrush& brush, double width, Qt::PenStyle style, Qt::PenCapStyle cap, Qt::PenJoinStyle join)
    : QPen(brush, width, style, cap, join)
{
}
