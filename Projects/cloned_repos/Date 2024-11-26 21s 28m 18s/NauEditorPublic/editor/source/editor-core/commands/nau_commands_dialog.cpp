// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_commands_dialog.hpp"
#include "nau_spoiler.hpp"
#include "themes/nau_theme.hpp"
#include "nau_log.hpp"

#include <QScrollBar>
#include <QTimer>


// ** NauCommandStackFooter

NauCommandStackFooter::NauCommandStackFooter(NauCommandStack& stack, NauWidget& parent)
    : NauWidget(&parent)
{
    setFixedHeight(Height);

    auto layoutMain = new NauLayoutVertical(this);

    auto separator = new NauLineWidget(ColorSeparator, 1, Qt::Horizontal, this);
    separator->setFixedWidth(NauCommandStackView::Width);
    separator->setFixedHeight(1);
    layoutMain->addWidget(separator);

    auto layoutLabel = new NauLayoutVertical;
    layoutLabel->setContentsMargins(MarginsLabel);
    layoutMain->addLayout(layoutLabel);
    
    auto labelStatus = new NauStaticTextLabel(QString(), this);
    labelStatus->setColor(ColorLabel);
    labelStatus->setFont(Nau::Theme::current().fontHistoryMain());
    layoutLabel->addStretch(1);
    layoutLabel->addWidget(labelStatus);
    layoutLabel->addStretch(1);

    auto modificationCallback = [&stack, labelStatus] {
        labelStatus->setText(tr("%1 Actions (%2 to Undo)").arg(stack.sizeTotal()).arg(stack.sizeRedo()));
        labelStatus->update();
    };

    stack.addModificationListener(modificationCallback);
    modificationCallback();
}


// ** NauCommandDescriptionView

NauCommandDescriptionView::NauCommandDescriptionView(const NauCommandDescriptor& descriptor, Role role, NauWidget* parent)
    : NauWidget(parent)
    , m_descriptor(descriptor)
    , m_role(role)
{
    setFixedHeight(Height);

    auto layout = new NauLayoutHorizontal(this);
    layout->setContentsMargins(MarginHorizontal, MarginVertical, MarginHorizontal, MarginVertical);

    // Text
    m_label = new NauStaticTextLabel(QString::fromUtf8(descriptor.name), this);
    if ((role == UndoCurrent) || (role == UndoLast)) {
        m_label->setColor(ColorTextActive);
    } else if (role == Undo) {
        m_label->setColor(ColorTextUndo);
    } else if ((role == Redo) || (role == RedoLast)) {
        m_label->setColor(ColorTextRedo);
    }

    m_label->setFont(Nau::Theme::current().fontHistoryMain());
    layout->addSpacing(OffsetLeft);
    layout->addWidget(m_label);
}

QPixmap& NauCommandDescriptionView::iconDotWhite()
{
    static QPixmap pixmap = Nau::Theme::current().iconFatDot().pixmap(6, 6);
    return pixmap;
}

QPixmap& NauCommandDescriptionView::iconDotRed()
{
    static QPixmap pixmap = Nau::paintPixmapCopy(Nau::Theme::current().iconFatDot().pixmap(6, 6), ColorIconRedDot);
    return pixmap;
}

QPixmap& NauCommandDescriptionView::iconUndo()
{
    static QPixmap pixmap = Nau::paintPixmapCopy(
        Nau::Theme::current().iconUndo().pixmap(UndoRedoButtonSize, UndoRedoButtonSize), ColorIconUndoRedo);
    return pixmap;
}

QPixmap& NauCommandDescriptionView::iconRedo()
{
    static QPixmap pixmap = Nau::paintPixmapCopy(
        Nau::Theme::current().iconRedo().pixmap(UndoRedoButtonSize, UndoRedoButtonSize), ColorIconUndoRedo);
    return pixmap;
}

QPixmap& NauCommandDescriptionView::iconLine()
{
    static QPixmap pixmap = Nau::Theme::current().iconDottedLine().pixmap(2, 40);
    return pixmap;
}

QPixmap& NauCommandDescriptionView::iconLineFade()
{
    static QPixmap pixmap = Nau::paintPixmapCopy(Nau::Theme::current().iconDottedLine().pixmap(2, 40), ColorLineFaded);
    return pixmap;
}

QPixmap& NauCommandDescriptionView::iconLineTail()
{
    static QPixmap pixmap = Nau::Theme::current().iconDottedLineTail().pixmap(2, 19);
    return pixmap;
}

void NauCommandDescriptionView::paintEvent(QPaintEvent* event) 
{
    QPainter painter(this);

    // Background
    if (m_hovered) {
        painter.fillRect(rect(), ColorSelection);
    } else {
        painter.fillRect(rect(), ColorBackground);
    }
    
    // Icon
    if ((m_hovered && ((m_role == Undo) || (m_role == UndoLast))) || (m_role == UndoCurrent)) {
        painter.drawPixmap(UndoRedoButtonSize, (height() - iconUndo().height()) * 0.5 + 1, iconUndo());
    } else if (m_hovered && ((m_role == Redo) || (m_role == RedoLast))) {
        painter.drawPixmap(UndoRedoButtonSize, (height() - iconUndo().height()) * 0.5 + 1, iconRedo());
    } else if (m_role == Redo) {
        if (hoveredView && (hoveredView->m_descriptor.id > m_descriptor.id)) {
            painter.drawPixmap(24, 0, iconLineFade());
        } else {
            painter.drawPixmap(24, 0, iconLine());
        }
    } else if (m_role == UndoLast) {
        painter.drawPixmap(22, (height() - iconDotWhite().height()) * 0.5 + 1, iconDotWhite());
    } else if (m_role == RedoLast) {
        painter.drawPixmap(24, 0, iconLineTail());
        painter.drawPixmap(22, (height() - iconDotRed().height()) * 0.5 + 1, iconDotRed());
    }

    // Text
    if ((m_role == Redo) || (m_role == RedoLast)) {
        if (m_hovered) {
            m_label->setColor(ColorTextActive);
        } else {
            m_label->setColor(ColorTextRedo);
        }
    }
}

void NauCommandDescriptionView::enterEvent(QEnterEvent* event)
{
    if ((m_role == Undo) || (m_role == UndoCurrent) || (m_role == UndoLast)) {
        m_label->setColor(ColorTextActive);
    }
    m_hovered = true;
    hoveredView = this;
    emit eventHover();
    update();
}

void NauCommandDescriptionView::leaveEvent(QEvent* event)
{
    if (m_role == Undo) {
        m_label->setColor(ColorTextUndo);
    }
    m_hovered = false;
    hoveredView = nullptr;
    emit eventHover();
    update();
}

void NauCommandDescriptionView::mousePressEvent(QMouseEvent* event)
{
    emit eventPressed(m_descriptor);
}


// ** NauCommandStackView

NauCommandStackView::NauCommandStackView(NauCommandStack& stack, NauWidget* parent)
    : NauWidget(parent)
{
    setFixedWidth(Width);
    setContentsMargins(0, 0, ScrollbarOffset, 0);

    auto layout = new NauLayoutVertical(this);

    // Header
    auto header = new NauHeader(tr("History"), this);
    [[maybe_unused]] auto buttonHelp = header->addStandardButton(NauHeader::Help);  // TODO
    [[maybe_unused]] auto buttonMenu = header->addStandardButton(NauHeader::Menu);  // TODO
    layout->addWidget(header);

    // Commands
    auto scrollWidget = new NauScrollWidgetVertical(this);
    scrollWidget->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    const auto addCommandView = [this, &stack, scrollWidget](const NauCommandDescriptor& descriptor, NauCommandDescriptionView::Role role)
    {
        auto view = new NauCommandDescriptionView(descriptor, role, this);
        connect(view, &NauCommandDescriptionView::eventPressed, [&stack, role](const NauCommandDescriptor& desc) {
            if ((role == NauCommandDescriptionView::Redo) || (role == NauCommandDescriptionView::RedoLast)) {
                stack.redo(desc.id);
            } else {
                stack.undo(desc.id);
            }
        });
        connect(view, &NauCommandDescriptionView::eventHover, [this] { update(); });
        scrollWidget->addWidget(view);
    };
    
    const auto fillCallback = [this, &stack, addCommandView, scrollWidget]
    {
        scrollWidget->layout()->clear();
        const NauCommandStackDescriptor description = stack.describe();

        // Redo commands
        for (auto i = int(description.redo.size() - 1); i >= 0 ; --i) {
            const auto& desc = description.redo[i];
            auto role = NauCommandDescriptionView::Redo;
            if (i == (description.redo.size() - 1)) {
                role = NauCommandDescriptionView::RedoLast;
            }
            addCommandView(desc, role);
        }

        // Separator
        if (!description.redo.empty() && !description.undo.empty()) {
            auto separator = new NauLineWidget(ColorSeparator, SeparatorSize, Qt::Horizontal, this);
            separator->setFixedWidth(width());
            separator->setFixedHeight(SeparatorSize);
            scrollWidget->addWidget(separator);
        }

        // Undo commands
        for (int i = 0; i < description.undo.size(); ++i) {
            const auto& desc = description.undo[i];
            auto role = NauCommandDescriptionView::Undo;
            if (i == 0) {
                if (description.redo.empty()) {
                    role = NauCommandDescriptionView::UndoLast;
                } else {
                    role = NauCommandDescriptionView::UndoCurrent;
                }
            } 
            addCommandView(desc, role);
        }
    };
    stack.addModificationListener(fillCallback);
    layout->addWidget(scrollWidget);

    // Footer
    layout->addWidget(new NauCommandStackFooter(stack, *this));

    fillCallback();

    // Scroll to the bottom
    QTimer::singleShot(1, [scrollWidget] {
        scrollWidget->verticalScrollBar()->setValue(scrollWidget->verticalScrollBar()->maximum());  // Scroll to bottom
    });
}


// ** NauCommandStackPopup

NauCommandStackPopup::NauCommandStackPopup(NauCommandStack& stack)
    : m_view(new NauCommandStackView(stack, this))
{
    setWindowFlags(Qt::Popup);
    setFixedWidth(NauCommandStackView::Width);
    setStyleSheet("background-color: #282828;");
 
    auto layout = new NauLayoutVertical(this);
    layout->addWidget(m_view);
}
