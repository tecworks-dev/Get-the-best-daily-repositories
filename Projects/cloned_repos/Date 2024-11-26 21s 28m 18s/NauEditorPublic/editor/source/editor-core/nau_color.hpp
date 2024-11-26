// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Helper color classes and functions

#pragma once

#include "nau/nau_editor_config.hpp"

#include <QColor>
#include <QPen>
#include <QBrush>
#include <QIcon>
#include <QPixmap>
#include <QString>


namespace Nau
{
    void     NAU_EDITOR_API paintPixmap(QPixmap& pixmap, const QColor& color);
    QPixmap  NAU_EDITOR_API paintPixmap(const QString& path, const QColor& color);
    QPixmap  NAU_EDITOR_API paintPixmapCopy(QPixmap pixmap, const QColor& color);
    QIcon    NAU_EDITOR_API paintIcon(const QString& path, QColor color);
}


// ** NauColor

class NAU_EDITOR_API NauColor : public QColor
{
public:
    // TODO: Maybe add a constructor from USD color type?
    NauColor() = default;
    NauColor(Qt::GlobalColor color);
    constexpr NauColor(int r, int g, int b, int a = 255) noexcept
        : QColor(r, g, b, a) {}
    NauColor(QRgb rgb);

    inline QString hex() const { return name(); }
};


// ** NauBrush

class NAU_EDITOR_API NauBrush : public QBrush
{
public:
    NauBrush() = default;
    NauBrush(const QColor& color, Qt::BrushStyle bs = Qt::SolidPattern);
};


// ** NauPen

class NAU_EDITOR_API NauPen : public QPen
{
public:
    NauPen() = default;
    NauPen(const NauColor& color);
    NauPen(const NauBrush& brush, double width, Qt::PenStyle style = Qt::SolidLine, Qt::PenCapStyle cap = Qt::SquareCap, Qt::PenJoinStyle join = Qt::BevelJoin);
};