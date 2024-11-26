// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Custom implementation of action.

#pragma once

#include "nau/nau_editor_config.hpp"

#include "nau_icon.hpp"
#include "nau_widget_utility.hpp"

#include <QAction>
#include <QEvent>


// ** NauAction

class NAU_EDITOR_API NauAction : public QAction
{
public:
    explicit NauAction(QObject* parent = nullptr);
    explicit NauAction(const QString& text, QObject* parent = nullptr);
    explicit NauAction(const QIcon& icon, const QString& text, QObject* parent = nullptr);
    explicit NauAction(const QIcon& icon, const QString& text,
        const NauKeySequence& shortcut, QObject *parent = nullptr);

    void setShortcut(const NauKeySequence &shortcut);

protected:
    bool event(QEvent* event) override;
};