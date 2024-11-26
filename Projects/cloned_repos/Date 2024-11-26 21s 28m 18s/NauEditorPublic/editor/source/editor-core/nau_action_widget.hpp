// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// This file will store everything related to ActionWidgets

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_label.hpp"
#include "baseWidgets/nau_static_text_label.hpp"
#include "filter/nau_filter_checkbox.hpp"


// ** NauActionWidgetAbstract
//
// The base abstract class that underlies all ActionWidgets

class NauActionWidgetAbstract : public NauWidget
{
public:
    NauActionWidgetAbstract(NauWidget* parent = nullptr);

protected:
    NauLayoutGrid* m_layout;

};


// ** NauActionWidgetBase
//
// The most basic of all widgets.
// It contains the same as the others: text and icon.

class NauActionWidgetBase : virtual public NauActionWidgetAbstract
{
public:

    // TODO: Add the ability to create a widget without an icon
    //NauActionWidgetBase(const QString& text, NauWidget* parent = nullptr);

    NauActionWidgetBase(const NauIcon& icon, const QString& text, NauWidget* parent = nullptr);

    NauStaticTextLabel& label();

protected:
    NauLabel* m_iconLabel;
    NauStaticTextLabel* m_label;

private:

    // TODO: It should be sized at 20px
    inline static constexpr int checkBoxSize = 24;

    inline static constexpr int iconSize = 16;
    inline static constexpr int ToolButtonSize = 16;
    inline static constexpr int MinTextBlockSize = 0;

    inline static constexpr int FirstBackspacing = 16;
    inline static constexpr int SecondBackspacing = 8;

    inline static constexpr int BottonMargin = 8;

};


// ** NauActionWidgetChecker
//
// Contains a checkBox that stores the state of this ActionWidget.

class NauActionWidgetChecker : virtual public NauActionWidgetBase
{
public:
    NauActionWidgetChecker(const NauIcon& icon, const QString& text, NauWidget* parent = nullptr);

    NauFilterCheckBox& checkBox();

protected:
    NauFilterCheckBox* m_checkBox;

private:
    inline static constexpr int checkBoxSize = 24;

};


// ** NauActionWidgetCatalog
//
// Has a button through which an additional instance of NauMenu is called.

class NauActionWidgetCatalog : virtual public NauActionWidgetBase
{
public:
    NauActionWidgetCatalog(const NauIcon& icon, const QString& text, NauWidget* parent = nullptr);

    // TODO: Add a method for creating and invoking a context menu with the NauActionWidgetBase class descendants

protected:
    NauToolButton* m_toolButton;

};


// ** NauActionWidgetCheckerCatalog
//
// Can display not only the additional menu, but also the internal status of all selected options in the sub menu

class NauActionWidgetCheckerCatalog : virtual public NauActionWidgetChecker, virtual public NauActionWidgetCatalog
{
public:
    NauActionWidgetCheckerCatalog(const NauIcon& icon, const QString& text, NauWidget* parent = nullptr);

    // TODO: Add a method for creating and invoking a context menu with the NauActionWidgetChecker class descendants
};
