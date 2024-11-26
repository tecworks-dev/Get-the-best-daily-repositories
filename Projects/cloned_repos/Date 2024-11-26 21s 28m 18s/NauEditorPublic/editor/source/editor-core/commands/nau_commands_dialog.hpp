// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Classes that provide view into a command stack

#pragma once

#include "nau_commands.hpp"
#include "nau_widget.hpp"
#include "nau_static_text_label.hpp"


// ** NauCommandStackFooter

class NauCommandStackFooter : public NauWidget
{
    Q_OBJECT
public:
    NauCommandStackFooter(NauCommandStack& stack, NauWidget& parent);

private:
    inline static constexpr int Height = 60;
    
    inline static const auto ColorSeparator  = NauColor( 27,  27,  27, 128);
    inline static const auto ColorLabel      = NauColor(153, 153, 153);

    inline static const auto MarginsLabel = QMargins(16, 12, 16, 16);
};


// ** NauCommandDescriptionView

class NauCommandDescriptionView : public NauWidget
{
    Q_OBJECT

    friend class NauCommandStackView;

public:
    enum Role {
        Undo,
        Redo,
        UndoCurrent,
        UndoLast,
        RedoLast,
    };

signals:
    void eventPressed(const NauCommandDescriptor& desc);
    void eventHover();

protected:
    void paintEvent(QPaintEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    NauCommandDescriptionView(const NauCommandDescriptor& descriptor, Role role, NauWidget* parent);

    static QPixmap& iconDotWhite();
    static QPixmap& iconDotRed();
    static QPixmap& iconUndo();
    static QPixmap& iconRedo();
    static QPixmap& iconLine();
    static QPixmap& iconLineFade();
    static QPixmap& iconLineTail();

private:
    const NauCommandDescriptor m_descriptor;
    const Role                 m_role;
    bool                       m_hovered  = false;
    NauStaticTextLabel*        m_label    = nullptr;

    inline static NauCommandDescriptionView* hoveredView = nullptr;

    inline static constexpr int Height             = 40;
    inline static constexpr int MarginVertical     = 10;
    inline static constexpr int MarginHorizontal   = 16;
    inline static constexpr int OffsetLeft         = 24;
    inline static constexpr int UndoRedoButtonSize = 16;

    inline static const auto ColorTextActive    = NauColor(255, 255, 255);
    inline static const auto ColorTextUndo      = NauColor(153, 153, 153);
    inline static const auto ColorTextRedo      = NauColor( 91,  91,  91);
    inline static const auto ColorIconRedDot    = NauColor(209,  35,  77);
    inline static const auto ColorIconUndoRedo  = NauColor(255, 255, 255);
    inline static const auto ColorBackground    = NauColor( 40,  40,  40);
    inline static const auto ColorSelection     = NauColor( 49,  67, 229);
    inline static const auto ColorLineFaded     = NauColor(100, 100, 100);
};


// ** NauCommandStackView

class NauCommandStackView : public NauWidget
{
    Q_DECLARE_TR_FUNCTIONS(NauCommandStackDialog)

public:
    NauCommandStackView(NauCommandStack& stack, NauWidget* parent = nullptr);

public:
    inline static constexpr int Width            = 360;
    inline static constexpr int ScrollbarOffset  = 7;
    inline static constexpr int SeparatorSize    = 2;

    inline static const auto ColorSeparator = NauColor(153, 153, 153);
};


// ** NauCommandStackPopup

class NauCommandStackPopup : public NauWidget
{
    Q_OBJECT

public:
    NauCommandStackPopup(NauCommandStack& stack);

private:
    NauCommandStackView* m_view;
};
