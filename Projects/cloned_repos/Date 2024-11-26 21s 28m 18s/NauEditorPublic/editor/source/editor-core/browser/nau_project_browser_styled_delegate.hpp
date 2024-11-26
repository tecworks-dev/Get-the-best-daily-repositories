// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A set of delegates for views in the project browser.

#pragma once

#include "nau_tree_view_item_delegate.hpp"
#include <QStyledItemDelegate>


// ** NauProjectBasicDelegate
// Implements common funcionality of all delegates in the project browser.

class NAU_EDITOR_API NauProjectBasicDelegate : public NauTreeViewItemDelegate
{
    Q_OBJECT
public:
    explicit NauProjectBasicDelegate(QObject* parent = nullptr);

    // In any view a user cannot change the extension of a file.
    virtual void setEditorData(QWidget *editor, const QModelIndex &index) const override;
    virtual void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const override;
};


// ** NauProjectContentViewTileDelegate

class NAU_EDITOR_API NauProjectContentViewTileDelegate : public NauProjectBasicDelegate
{
    Q_OBJECT
public:
    explicit NauProjectContentViewTileDelegate(QObject* parent = nullptr);

    // Sets value of scale of items. Valid values between [0.0..1.0].
    void setScale(float value);

    virtual void paint(QPainter* painter,
        const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    
    virtual QSize sizeHint(const QStyleOptionViewItem& option, 
        const QModelIndex& index) const override;

    virtual QWidget* createEditor(QWidget *parent, const QStyleOptionViewItem &option,
        const QModelIndex &index) const override;

    virtual void updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem& option,
        const QModelIndex& index) const override;

private:
    QRect calculateTextRect(const QStyleOptionViewItem& option) const;
    std::pair<QColor, QBrush> getBackgroundBrush(const QStyleOptionViewItem& option) const;

private:
    QMargins m_spacing;
    QMargins m_marging;

    const QSize m_minSizeHint{64, 64};
    const QSize m_maxSizeHint{144, 144};
    const QPixmap m_lockPix;

    QSize m_sizeHint;
};


// ** NauProjectContentViewTableDelegate

class NAU_EDITOR_API NauProjectContentViewTableDelegate : public NauProjectBasicDelegate
{
public:
    explicit NauProjectContentViewTableDelegate(QObject* parent = nullptr);
    QString displayText(const QVariant &value, const QLocale &locale) const override;
};


// ** NauProjectTreeDelegate

class NAU_EDITOR_API NauProjectTreeDelegate : public NauTreeViewItemDelegate
{
public:
    explicit NauProjectTreeDelegate(QObject* parent = nullptr);
};
