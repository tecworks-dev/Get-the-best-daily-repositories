// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A flow(right to left, left to right) layout.

#pragma once 

#include "baseWidgets/nau_widget.hpp"

#include <QLayout>
#include <QRect>
#include <QStyle>


// ** NauFlowLayout

class NauFlowLayout : public QLayout
{
public:
    // Only QtLayoutDirection::LeftToRight, QtLayoutDirection::RightToLeft allowed.
    NauFlowLayout(Qt::LayoutDirection direction, int margin, int hSpacing, int vSpacing, NauWidget* parent = nullptr);
    ~NauFlowLayout();

    virtual void addItem(QLayoutItem* item) override;
    virtual Qt::Orientations expandingDirections() const override;
    virtual bool hasHeightForWidth() const override;
    virtual int heightForWidth(int) const override;
    virtual int count() const override;
    virtual QLayoutItem* itemAt(int index) const override;
    virtual QSize minimumSize() const override;
    virtual void setGeometry(const QRect& rect) override;
    virtual QSize sizeHint() const override;
    virtual QLayoutItem* takeAt(int index) override;

    void addWidget(int atIndex, QWidget* widget);
    int horizontalSpacing() const;
    int verticalSpacing() const;

private:
    int doLayout(const QRect& rect, bool testOnly) const;

    QSize calcSpaces(const QWidget* widget) const;
    int doLayoutLeftToRight(const QRect& rect, const QRect& effectiveRect, bool testOnly) const;
    int doLayoutRightToLeft(const QRect& rect, const QRect& effectiveRect, bool testOnly) const;

    int smartSpacing(QStyle::PixelMetric pm) const;

private:
    const Qt::LayoutDirection m_layoutDirection;
    int m_hSpace;
    int m_vSpace;

    QList<QLayoutItem*> m_itemList;
};
