// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_action.hpp"
#include "nau_log.hpp"

// ** NauAction

NauAction::NauAction(QObject* parent)
    : QAction(parent)
{
}

NauAction::NauAction(const QString& text, QObject* parent)
    : QAction(text, parent)
{
}

NauAction::NauAction(const QIcon& icon, const QString& text, QObject* parent)
    : QAction(icon, text, parent)
{
}

NauAction::NauAction(const QIcon& icon, const QString& text, const NauKeySequence& shortcut, QObject* parent)
    : QAction(icon, text, parent)
{
    setShortcut(shortcut);
}

void NauAction::setShortcut(const NauKeySequence&)
{
    // Do not install a shortcut for the editor has a special class to dispatch
    // keyboard short activations between handlers.
}

bool NauAction::event(QEvent* event)
{
    if (event->type() == QEvent::Shortcut || event->type() == QEvent::ShortcutOverride) {
        // Discard shortcut triggering event for we have a special dispatcher NauShortcutHub 
        // to distribute these triggers between receivers.
        return true;
    }

    return QAction::event(event);
}
