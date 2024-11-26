// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_menu.hpp"
#include "nau_widget.hpp"


// ** NauMenu

NauMenu::NauMenu(QWidget* parent)
    : QWidget(parent)
    , m_menu(new QMenu(parent))
{
    m_menu->setStyleSheet("background-color: #343434");
    m_menu->setContentsMargins(HorizontalMargin, VerticalMargin, HorizontalMargin, 0);
}

NauMenu::NauMenu(const QString& title, NauWidget* widget)
    : QWidget(widget)
    , m_menu(new QMenu(title, widget))
{
    m_menu->setStyleSheet("background-color: #343434");
    m_menu->setContentsMargins(HorizontalMargin, VerticalMargin, HorizontalMargin, 0);
}

void NauMenu::addAction(QAction* action)
{
    m_menu->addAction(action);
}

void NauMenu::addSeparator()
{
    m_menu->addSeparator();
}

void NauMenu::clear()
{
    m_menu->clear();
}

NauAction* NauMenu::addAction(const QString& text)
{
    auto result = new NauAction(text, m_menu);
    m_menu->addAction(result);

    return result;
}

NauAction* NauMenu::addAction(const NauIcon& icon, const QString& text)
{
    auto result = new NauAction(icon, text, m_menu);
    m_menu->addAction(result);

    return result;
}

NauAction* NauMenu::addAction(const QString& text, const QObject* receiver, 
    const char* member, Qt::ConnectionType type)
{
    auto result = addAction(text);
    QObject::connect(result, SIGNAL(triggered(bool)), receiver, member, type);

    return result;
}

NauAction* NauMenu::addAction(const NauIcon& icon, const QString& text,
    const QObject* receiver, const char* member, Qt::ConnectionType type)
{
    auto result = addAction(icon, text);
    QObject::connect(result, SIGNAL(triggered(bool)), receiver, member, type);

    return result;
}

NauAction* NauMenu::addAction(const QString& text, const NauKeySequence& shortcut)
{
    return addAction(NauIcon(), text, shortcut);
}

NauAction* NauMenu::addAction(const NauIcon& icon, const QString& text, const NauKeySequence& shortcut)
{
    auto result = new NauAction(icon, text, shortcut, m_menu);
    m_menu->addAction(result);

    return result;
}

NauAction* NauMenu::addAction(const QString& text, const NauKeySequence& shortcut,
    const QObject* receiver, const char* member, Qt::ConnectionType type)
{
    auto result = addAction(text, shortcut);
    QObject::connect(result, SIGNAL(triggered(bool)), receiver, member, type);

    return result;
}

NauAction* NauMenu::addAction(const NauIcon& icon, const QString& text,
    const NauKeySequence& shortcut, const QObject* receiver, const char* member, Qt::ConnectionType type)
{
    auto result = addAction(icon, text, shortcut);
    QObject::connect(result, SIGNAL(triggered(bool)), receiver, member, type);

    return result;
}
