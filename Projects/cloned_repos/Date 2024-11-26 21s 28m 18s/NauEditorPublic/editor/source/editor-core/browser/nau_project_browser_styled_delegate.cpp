// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_project_browser_styled_delegate.hpp"
#include "nau_assert.hpp"
#include "nau_widget.hpp"
#include "nau_plus_enum.hpp"
#include "nau_project_browser_file_system_model.hpp"
#include "themes/nau_theme.hpp"

#include <QPainter>


// ** NauProjectBasicDelegate

NauProjectBasicDelegate::NauProjectBasicDelegate(QObject* parent)
    : NauTreeViewItemDelegate(parent)
{
}

void NauProjectBasicDelegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
    const QFileInfo fi{index.data(NauProjectBrowserFileSystemModel::FilePathRole).toString()};
    if (fi.isDir()) {
        QStyledItemDelegate::setEditorData(editor, index);
        return;
    }

    // We don't want user to change the extension of any file.
    if (auto lineEditor = dynamic_cast<QLineEdit*>(editor)) {
        lineEditor->setText(fi.completeBaseName());
    }
}

void NauProjectBasicDelegate::setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
    const QFileInfo fi{index.data(NauProjectBrowserFileSystemModel::FilePathRole).toString()};
    if (fi.isDir()) {
        QStyledItemDelegate::setModelData(editor, model, index);
        return;
    }

    if (auto lineEditor = dynamic_cast<QLineEdit*>(editor)) {
        QString newFileName = lineEditor->text();
        if (!fi.suffix().isEmpty())
            newFileName += "." + fi.suffix();

        model->setData(index, newFileName, Qt::EditRole);
        return;
    }

    QStyledItemDelegate::setModelData(editor, model, index);
}


// ** NauProjectContentViewTileDelegate

NauProjectContentViewTileDelegate::NauProjectContentViewTileDelegate(QObject* parent)
    : NauProjectBasicDelegate(parent)
    , m_spacing{ 2, 2, 2, 2 }
    , m_marging{ 2, 2, 2, 2 }
    , m_lockPix{":/UI/icons/lock.png"}
{
    setScale(0.5);
}

void NauProjectContentViewTileDelegate::setScale(float value)
{
    m_sizeHint = m_minSizeHint + std::clamp(value, 0.0f, 1.0f) * (m_maxSizeHint - m_minSizeHint);
    emit sizeHintChanged(QModelIndex());
}

void NauProjectContentViewTileDelegate::paint(QPainter* painter,
    const QStyleOptionViewItem& option, const QModelIndex& index) const
{
   if (index.column() != 0) 
   {
        QStyledItemDelegate::paint(painter, option, index);
        return;
   }

    const QRect boundary = option.rect - m_spacing;
    const auto boundaryFrameWidth = 1.0f;
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing);
    painter->setRenderHint(QPainter::TextAntialiasing);
    painter->setRenderHint(QPainter::SmoothPixmapTransform);

    // Draw boundary frame.
    const auto[color, brush] = getBackgroundBrush(option);
    painter->setBrush(brush);
    painter->setPen(QPen(color, boundaryFrameWidth));
    painter->drawRoundedRect(boundary, 2, 2);

    // Draw resource name.
    const QRect textRect = calculateTextRect(option);
    const QString text = option.fontMetrics.elidedText(index
        .data(Qt::DisplayRole).toString(), option.textElideMode, textRect.width());
 
    painter->setPen(Qt::white);
    painter->drawText(textRect, text, { option.displayAlignment });

    // Draw icon.
    const QIcon px = index.data(NauProjectBrowserFileSystemModel::FileIconRole).value<QIcon>();

    if (!px.isNull()) {
        QRect pxRect = boundary - m_marging;
        pxRect.setBottom(textRect.top());
        px.paint(painter, pxRect);
    }

    if (!index.flags().testFlag(Qt::ItemIsEditable)) {
        const QRect lockPixRect = QRect{ boundary.topLeft(), 0.20 * boundary.size() } -
            boundaryFrameWidth * QMargins{1, 1, 1, 1};

        painter->drawPixmap(lockPixRect, m_lockPix);
    }

    painter->restore();
}

QSize NauProjectContentViewTileDelegate::sizeHint(
    const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    if (index.column() == 0) return m_sizeHint;

    return QStyledItemDelegate::sizeHint(option, index);
}

QWidget* NauProjectContentViewTileDelegate::createEditor(QWidget* parent,
    const QStyleOptionViewItem&, const QModelIndex&) const
{
    return new NauLineEdit(parent);
}

void NauProjectContentViewTileDelegate::updateEditorGeometry(QWidget* editor,
    const QStyleOptionViewItem& option, const QModelIndex& /*index*/) const
{
    editor->setGeometry(calculateTextRect(option));
}

QRect NauProjectContentViewTileDelegate::calculateTextRect(const QStyleOptionViewItem& option) const
{
    const QRect boundary = option.rect - m_spacing - m_marging;
    QRect textRect{0, 0, boundary.width(), option.fontMetrics.height()};
    textRect.moveBottomLeft(boundary.bottomLeft());

    return textRect;
}

std::pair<QColor, QBrush> NauProjectContentViewTileDelegate::getBackgroundBrush(const QStyleOptionViewItem& option) const
{
    std::pair<QColor, QBrush> result{ Qt::transparent, Qt::transparent };

    if (option.state.testFlag(QStyle::State_Selected)) {
        result = std::make_pair(QColor(0x3143E5), Qt::transparent);

    } else if (option.state.testFlag(QStyle::State_MouseOver)) {
        result = std::make_pair(QColor(0x5566FF), Qt::transparent);

    } else if (option.state.testFlag(QStyle::State_ReadOnly)) {
        result = std::make_pair(Qt::red, Qt::transparent);
    }

    return result;
}


// ** NauProjectContentViewListDelegate

NauProjectContentViewTableDelegate::NauProjectContentViewTableDelegate(QObject* parent)
    : NauProjectBasicDelegate(parent)
{
    setPalette(Nau::Theme::current().paletteProjectBrowser());
    setColumnHighlighted(+NauProjectBrowserFileSystemModel::Column::Name);
    setRowHeight(32);
    setCellContentsMargins(0, 0, 0, 0);
    setRowContentsMargins(16, 8, 16, 8);
    setSpacing(8);
}

QString NauProjectContentViewTableDelegate::displayText(const QVariant& value, const QLocale& locale) const
{
    if (value.metaType() == QMetaType::fromType<QDateTime>()) {
        return value.value<QDateTime>().toString(QStringLiteral("MMM dd, yyyy, hh:mm:ss"));
    }

    return NauProjectBasicDelegate::displayText(value, locale);
}


// ** NauProjectTreeDelegate

NauProjectTreeDelegate::NauProjectTreeDelegate(QObject* parent)
    : NauTreeViewItemDelegate(parent)
{
    setPalette(Nau::Theme::current().paletteProjectBrowser());
    setRowHeight(40);
    setColumnHighlighted(0, true);
    setCellContentsMargins(4, 0, 4, 0);
    setRowContentsMargins(16, 11, 16, 11);
    setSpacing(8);
    setIndentation(16);
    setRootColumn(0);
}
