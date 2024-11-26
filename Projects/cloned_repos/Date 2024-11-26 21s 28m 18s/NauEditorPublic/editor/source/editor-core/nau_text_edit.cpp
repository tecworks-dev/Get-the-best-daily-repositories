// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_text_edit.hpp"
#include "themes/nau_theme.hpp"

#include <QApplication>
#include <QMimeData>
#include <QClipboard>
#include <QFocusEvent>
#include <QPainter>


NauTextEdit::NauTextEdit(QWidget* parent)
    : QTextEdit(parent)
    , m_currentState(NauWidgetState::Active)
    , m_maxLength(1000)
{
    const auto& theme = Nau::Theme::current();
    m_styleMap = theme.styleFeedbackTextEditWidget().styleByState;

    setFocusPolicy(Qt::StrongFocus);
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
}

void NauTextEdit::paintEvent(QPaintEvent* event)
{
    QPainter painter(viewport());
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);

    painter.fillRect(rect(), m_styleMap[m_currentState].background);

    painter.setPen(m_styleMap[m_currentState].outlinePen);
    QRect borderRect = rect();
    painter.drawRect(borderRect);

    QTextEdit::paintEvent(event);
}

void NauTextEdit::focusInEvent(QFocusEvent* event)
{
    if (m_currentState != NauWidgetState::Error) {
        setState(NauWidgetState::Pressed);
    }

    QTextEdit::focusInEvent(event);
}

void NauTextEdit::focusOutEvent(QFocusEvent* event)
{
    if (m_currentState != NauWidgetState::Error) {
        setState(NauWidgetState::Active);
    }

    QTextEdit::focusOutEvent(event);
}

void NauTextEdit::keyPressEvent(QKeyEvent* event)
{
    if (event->matches(QKeySequence::Paste)) {
        handlePasteEvent();
    } else if (toPlainText().length() >= m_maxLength && event->key() != Qt::Key_Backspace && event->key() != Qt::Key_Delete) {
        event->ignore();
    } else {
        QTextEdit::keyPressEvent(event);
    }
}

void NauTextEdit::handlePasteEvent()
{
    const QMimeData* clipboard = QApplication::clipboard()->mimeData();
    if (clipboard->hasText()) {
        QString text = clipboard->text();
        if (toPlainText().length() + text.length() > m_maxLength) {
            int availableSpace = m_maxLength - toPlainText().length();
            text = text.left(availableSpace);
        }
        QTextEdit::insertPlainText(text);
    }
}

void NauTextEdit::setState(NauWidgetState state)
{
    m_currentState = state;
    update();

    emit eventStateChanged(state);
}

void NauTextEdit::setMaxLength(int maxLength)
{
    m_maxLength = maxLength;
}
