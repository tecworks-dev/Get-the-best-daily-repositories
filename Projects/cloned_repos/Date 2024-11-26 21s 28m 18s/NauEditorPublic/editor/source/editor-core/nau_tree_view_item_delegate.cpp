// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_tree_view_item_delegate.hpp"
#include "nau_plus_enum.hpp"
#include "themes/nau_theme.hpp"


// ** NauTreeViewItemDelegate

NauTreeViewItemDelegate::NauTreeViewItemDelegate(QObject* parent)
    : QStyledItemDelegate(parent)
    , m_arrowRight(Nau::Theme::current().iconArrowRight())
    , m_arrowDown(Nau::Theme::current().iconArrowDown())
    , m_rootAffected(false)
{
}

NauColor appropriateColor(const NauPalette& palette, NauPalette::Role role,
    const QStyleOptionViewItem& opt)
{
    const auto category = !opt.state.testFlag(QStyle::State_Enabled)
        ? NauPalette::Category::Disabled
        : NauPalette::Category::Active;

    NauColor result = palette.color(role, NauPalette::State::Normal, category);

    const auto maybeApplyColor = [&palette, &opt,  role, category, &result](QFlags<QStyle::StateFlag> states, int nauState)
    {
        if (opt.state.testFlags(states)) {
            const auto color = palette.color(role, nauState, category);
            if (color.isValid()) {
                result = color;
            }
        }
    };

    maybeApplyColor(QStyle::State_Selected, NauPalette::State::Selected);
    maybeApplyColor(QStyle::State_MouseOver, NauPalette::State::Hovered);
    maybeApplyColor(QStyle::State_Selected | QStyle::State_MouseOver, NauPalette::State::Hovered | NauPalette::State::Selected);

    return result;
}

void NauTreeViewItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option,
    const QModelIndex& index) const
{
    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);

    // First set the center, and only then calculate the space available for drawing
    QRect drawRect;
    drawRect.moveCenter(opt.rect.center());
    drawRect = calculateContentRect(opt);

    painter->save();
    painter->setRenderHints(QPainter::SmoothPixmapTransform);
    
    int paletteState = {};

    const bool selected = opt.state.testFlag(QStyle::State_Selected);
    const bool hovered = opt.state.testFlag(QStyle::State_MouseOver);
    const bool checked = opt.checkState == Qt::CheckState::Checked;

    if (selected) {
        paletteState |= NauPalette::State::Selected;
    }

    if (hovered) {
        paletteState |= NauPalette::State::Hovered;
    }

    NauPalette::Category paletteCategory = NauPalette::Category::Active;
    if (!opt.state.testFlag(QStyle::State_Enabled)) {
        paletteCategory = NauPalette::Category::Disabled;
    }

    if (!opt.state.testFlag(QStyle::State_Active)) {
        paletteCategory = NauPalette::Category::Inactive;
    }

    const auto originBackgroundData = index.data(Qt::BackgroundRole);
    if (originBackgroundData.canConvert<QBrush>()) {
        painter->fillRect(opt.rect, index.data(Qt::BackgroundRole).value<QBrush>());
    }

    const NauBrush backgroundBrush = opt.features.testFlag(QStyleOptionViewItem::Alternate) && 
        !selected && !hovered
        ? m_palette.brush(NauPalette::Role::AlternateBackground)
        : m_palette.brush(NauPalette::Role::Background, paletteState, paletteCategory);

    painter->fillRect(opt.rect, backgroundBrush);

    const NauColor borderColor = opt.state.testFlag(QStyle::State_MouseOver)
        ? m_palette.color(NauPalette::Role::Border, NauPalette::Hovered, paletteCategory)
        : m_palette.color(NauPalette::Role::Border, {}, paletteCategory);

    if (borderColor.isValid()) {
        painter->setPen(QPen(borderColor, m_penWidth, Qt::SolidLine, Qt::FlatCap));
        painter->setBrush(Qt::NoBrush);
        const QRectF borderRect = opt.rect - 0.5 * painter->pen().widthF() * QMarginsF(1.0f, 1.0f, 1.0f, 1.0f);

        if (opt.viewItemPosition == QStyleOptionViewItem::OnlyOne) {

            painter->drawRect(borderRect);
        } else {

            if (opt.viewItemPosition == QStyleOptionViewItem::Beginning) {
                painter->drawLine(borderRect.topLeft(), borderRect.bottomLeft());
            }
            
            if (opt.viewItemPosition == QStyleOptionViewItem::End) {
                painter->drawLine(borderRect.topRight(), borderRect.bottomRight());
            }

            const QRectF lineRect = opt.rect - painter->pen().widthF() * QMarginsF(0.0f, 1.0f, 0.0f, 1.0f);
            painter->drawLine(lineRect.topLeft(), lineRect.topRight());
            painter->drawLine(lineRect.bottomLeft(), lineRect.bottomRight());
        }
    }

    if (hasIndentation() && index.column() == m_rootColumn) {
        const QRect objectRect{ option.rect.left() + (option.decorationSize.width() / 2) + (calcIndentation(index) / 2),
    option.rect.top() + option.decorationSize.height() / 2, option.decorationSize.width(), option.decorationSize.height() };

        if (index.model()->hasChildren(index)) {
            if (opt.state.testFlag(QStyle::State_Open)) {
                m_arrowDown.paint(painter, objectRect);
            } else {
                m_arrowRight.paint(painter, objectRect);
            }
        }

        drawRect.setLeft(objectRect.right() + 1 + m_spacing);
    }

    if (opt.features.testFlag(QStyleOptionViewItem::HasDecoration)) {


        if (!m_interactableColumns.contains(index.column())) {
            const QRect objectRect{ drawRect.topLeft(), QPoint(drawRect.left() + opt.decorationSize.width(), drawRect.bottom()) };
            opt.icon.paint(painter, objectRect);
            drawRect.setLeft(objectRect.right() + m_spacing);

        } else if (hovered || checked) {
            QRect objectRect{ QPoint(0, 0), opt.decorationSize };
            objectRect.moveCenter(opt.rect.center());

            QStyleOptionButton button;
            button.iconSize = opt.decorationSize;
            button.rect = objectRect;
            button.icon = opt.icon;

            // One of the options to toggle the button state
            // Unfortunately it is impossible to get the state, because it is changed from the outside

            // TODO: Find another more feasible solution
            button.state = opt.checkState == Qt::CheckState::Checked ? QStyle::State_On : QStyle::State_Off;

            // To make the Button transparent.
            QApplication::style()->drawControl(QStyle::CE_PushButtonLabel, &button, painter);
        }
    }
    
    // TODO: 
    if (opt.features.testFlag(QStyleOptionViewItem::HasDisplay)) {
        const QRect objectRect{ drawRect.topLeft(), QPoint(drawRect.left() + opt.decorationSize.width(), drawRect.bottom()) };

        const QString displayText = opt.fontMetrics.elidedText(opt.text, opt.textElideMode, drawRect.width());//drawRect.width() + (hasIndentation() && index.column() == m_rootColumn ? 0 : 1000));

        const NauPalette::Role  foregroundRole = m_highlightedColumns.contains(index.column())
            ? NauPalette::Role::ForegroundBrightText
            : NauPalette::Role::Foreground;

        painter->setPen(appropriateColor(m_palette, foregroundRole, opt));
        painter->setFont(opt.font);
        painter->drawText(drawRect, displayText, QTextOption(Qt::AlignLeft | Qt::AlignVCenter));
    }

    painter->restore();
}

bool NauTreeViewItemDelegate::editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option, const QModelIndex& index)
{
    // TODO: Double click is not handled correctly in this event.
    // Need to fix it.
    if (event->type() == QEvent::MouseButtonRelease) {

        QMouseEvent* mouseEvent = (QMouseEvent*)event;
        const bool isLMB = mouseEvent->button() == Qt::MouseButton::LeftButton;
        // First processes the cell that controls the hierarchy
        if (index.column() == m_rootColumn) {
            if (event->type() == QEvent::MouseButtonRelease) {
                const QRect objectRect{ option.rect.left() + (option.decorationSize.width() / 2) + (calcIndentation(index) / 2),
                    option.rect.top() + option.decorationSize.height() / 2 + m_topArrowPadding, option.decorationSize.width(), option.decorationSize.height() };
               
                const int clickX = mouseEvent->position().x();
                const int clickY = mouseEvent->position().y();

                if ((clickX > objectRect.left() && clickX < objectRect.left() + objectRect.width()) 
                    && (clickY > objectRect.top() && clickY < objectRect.top() + objectRect.height())) {
                    emit buttonEventPressed(index, index.column());
                    return true;

                // If the center cell is still editable, send another event 
                } else if (isLMB && m_editableColumns.contains(index.column())) {
                    emit buttonRenamePressed(index, index.column());
                    return true;
                }
            }

        // Then there are the interactive cells 
        } else if (m_interactableColumns.contains(index.column())) {
            emit buttonEventPressed(index, index.column());
            return true;

        // And at the end what can be edited 
        } else if (isLMB && m_editableColumns.contains(index.column())) {
            emit buttonRenamePressed(index, index.column());
            return true;
        }
    }

    // In general, we call the parent method for all other cells
    return QStyledItemDelegate::editorEvent(event, model, option, index);
}

QSize NauTreeViewItemDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    QSize proposedSize = QStyledItemDelegate::sizeHint(option, index);
    proposedSize.setHeight(m_rowHeight);

    return proposedSize;
}

void NauTreeViewItemDelegate::updateEditorGeometry(QWidget* editor, 
    const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);

    QRect rect = calculateContentRect(opt);
    rect.setLeft(rect.left() + opt.decorationSize.width() + m_spacing);
    rect = (QRectF(rect) - 2 * QMarginsF(0, m_penWidth, 0, m_penWidth)).toRect();

    editor->setGeometry(rect);
    editor->setFont(opt.font);

    const QColor background = m_palette.brush(NauPalette::Role::Background,
        NauPalette::Selected | NauPalette::Hovered).color();

    // TO-DO: Use nau line editor when it is ready to avoid qss manipulating.
    editor->setStyleSheet(
        QStringLiteral("background-color : %1; border: none;").arg(background.name()));
}

void NauTreeViewItemDelegate::setRowHeight(int rowHeight)
{
    m_rowHeight = rowHeight;
}

void NauTreeViewItemDelegate::setSpacing(int spacing)
{
    m_spacing = spacing;
}

void NauTreeViewItemDelegate::setBorderThickness(float border)
{
    m_penWidth = border;
}

void NauTreeViewItemDelegate::setCellContentsMargins(int left, int top, int right, int bottom)
{
    m_cellContentMargin = QMargins{left, top, right, bottom};
}

void NauTreeViewItemDelegate::setRowContentsMargins(int left, int top, int right, int bottom)
{
    m_rowContentMargin = QMargins{left, top, right, bottom};
}

void NauTreeViewItemDelegate::setPalette(const NauPalette& palette)
{
    m_palette = palette;
}

void NauTreeViewItemDelegate::setColumnHighlighted(int logicalIndex, bool highlighted)
{
    if (highlighted) {
        m_highlightedColumns.insert(logicalIndex);
    } else {
        m_highlightedColumns.erase(logicalIndex);
    }
}

void NauTreeViewItemDelegate::setInteractiveColumnVisible(const QModelIndex& logicalIndex, bool highlighted)
{
    if (highlighted) {
        m_interactableColumnsVisible.insert(logicalIndex);
    } else {
        m_interactableColumnsVisible.erase(logicalIndex);
    }
}

void NauTreeViewItemDelegate::setIndentation(int indentation)
{
    m_indentation = std::max(0, indentation);
}

void NauTreeViewItemDelegate::setInteractableColumn(int logicalIndex)
{
    m_interactableColumns.insert(logicalIndex);
}

void NauTreeViewItemDelegate::setRootColumn(int logicalIndex)
{
    m_rootColumn = logicalIndex;
}

void NauTreeViewItemDelegate::setEditableColumn(int logicalIndex)
{
    m_editableColumns.insert(logicalIndex);
}

void NauTreeViewItemDelegate::setRootAffect(bool affected)
{
    m_rootAffected = affected;
}

QRect NauTreeViewItemDelegate::calculateContentRect(const QStyleOptionViewItem& opt) const
{
    QRect drawRect = opt.rect - m_cellContentMargin;
    if (opt.viewItemPosition == QStyleOptionViewItem::OnlyOne) {
        drawRect -= m_rowContentMargin;
    } else if (opt.viewItemPosition == QStyleOptionViewItem::Beginning) {
        drawRect -= QMargins(m_rowContentMargin.left(), m_rowContentMargin.top(), 0, m_rowContentMargin.bottom());
    } else if (opt.viewItemPosition == QStyleOptionViewItem::End) {
        drawRect -= QMargins(0, m_rowContentMargin.top(), m_rowContentMargin.right(), m_rowContentMargin.bottom());
    }

    if (hasIndentation()) {
        if (const int indentation = calcIndentation(opt.index); indentation > 0) {
            drawRect -= QMargins(indentation, 0, 0, 0);
        }
    }

    return drawRect;
}

int NauTreeViewItemDelegate::calcIndentation(const QModelIndex& index) const
{
    if (!index.isValid() || index.column() != m_rootColumn && hasIndentation()) {
        return 0;
    }

    int levelInTree = 0;
    QModelIndex iterIndex = index;

    while (iterIndex.parent().isValid()) {
        ++levelInTree;
        iterIndex = iterIndex.parent();
    }

    // We may say that all incoming indexes have a parent - root.
    // Although the root must not affect of level calculation.
    return std::max(0, levelInTree - (!m_rootAffected)) * m_indentation;
}

bool NauTreeViewItemDelegate::hasIndentation() const
{
    return m_indentation > 0;
}
