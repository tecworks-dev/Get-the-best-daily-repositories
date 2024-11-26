// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Class displays the headers used in item views such as the QTableView and QTreeView classes.
// Implemented as a customization of QHeaderView.

#pragma once

#include "nau_widget.hpp"
#include "nau_label.hpp"
#include "nau_icon.hpp"
#include "nau_color.hpp"
#include "nau_plus_enum.hpp"
#include "scene/nau_world.hpp"

#include <unordered_set>


// ** NauHeaderView

class NAU_EDITOR_API NauHeaderView : public QHeaderView
{
    Q_OBJECT
public:
    NauHeaderView(Qt::Orientation orientation, QWidget* parent = nullptr);
  
    void setSectionHideable(int logicalIndex, bool hidable);
    void setBackgroundBrush(const NauBrush& brush);
    void setForegroundColor(const NauColor& color);
    void setForegroundBrightText(const NauColor& color);

    void setSpacing(int spacing);
    void setContentMargin(int aleft, int atop, int aright, int abottom);

    // Set palette for this header view.
    // Fetches brush NauPalette::Role::BackgroundHeader as a background and
    // text color NauPalette::Role::ForegroundHeader.
    void setPalette(const NauPalette& palette);

    void setColumnHighlighted(int logicalIndex, bool highlighted = true);

signals:
    void eventColumnVisibleToggled(int logicalIndex, bool visible);

protected:
    void paintSection(QPainter* painter, const QRect& rect, int logicalIndex) const override;
    void contextMenuEvent(QContextMenuEvent* event) override;

    virtual QSize sortIndicatorSize() const;
    virtual void fillContextMenu(NauMenu& menu);

private:
    NauBrush m_brush;
    NauColor m_color;
    NauColor m_brightColor;

    int m_spacing = 4;
    QMargins m_contentMargin{8, 8, 4, 8};

    NauPalette m_palette;

    std::unordered_set<int> m_unhideableSections;
    std::unordered_set<int> m_highlightedColumns;
};
