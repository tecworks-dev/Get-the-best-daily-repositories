// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Spoiler and its base that can be used as a separate header widget

#pragma once

#include <baseWidgets/nau_widget.hpp>
#include <baseWidgets/nau_buttons.hpp>
#include <baseWidgets/nau_static_text_label.hpp>


// ** NauHeader

class NAU_EDITOR_API NauHeader : public NauWidget
{
    Q_OBJECT

public:

    enum Button {
        Help,
        Menu
    };

    NauHeader(const QString& title = QString(), NauWidget* parent = nullptr);

    NauToolButton* addStandardButton(Button kind);

protected:
    QToolButton*          m_toggleButton;
    NauLayoutVertical*    m_mainLayout;
    NauLayoutHorizontal*  m_headerLayout;
    
 protected:
    inline static constexpr int HeaderHeight    = 24;
    inline static constexpr int OuterMargin     = 16;
    inline static constexpr int ButtonIconSize  = 16;
    inline static constexpr int ButtonMargin    = 4;
    inline static constexpr int Spacing         = 10;

    // TODO: Fix the color as it doesn't exactly match the mock-up, 
    // because paintPixmapCopy seems to be broken
    inline static const auto ColorDefaultButton = NauColor(100, 100, 100);
};


// ** NauSpoiler
//
// A spoiler is a widget that can hide and reveal the content that is stored in it

class NAU_EDITOR_API NauSpoiler : public NauHeader
{
    Q_OBJECT

public:
    NauSpoiler(const QString& title = QString(), int animationDuration = 300, NauWidget* parent = nullptr);

    void setExpanded();
    void addWidget(NauWidget* widget);
    void removeWidget(const NauWidget* widget);
    std::vector<NauWidget*> removeWidgets();
    void handleToggleRequested(bool expandFlag);
    [[nodiscard]]
    bool isToggled() const noexcept;
    [[nodiscard]]
    int userWidgetsCount() const noexcept;

signals:
    void eventStartedExpanding(bool expandFlag);

protected:
    void updateAnimation(int duration = -1, int areaHeight = -1);

    bool eventFilter(QObject* obj, QEvent* event) override;

protected:
    QFrame*                   m_headerLine;
    QParallelAnimationGroup*  m_toggleAnimation;
    QScrollArea*              m_contentArea;
    NauLayoutVertical*        m_contentLayout;

    int m_animationDuration;
    int m_userWidgetsCount;
    bool m_expanded;

private:
    inline static constexpr int SeparatorLineHeight  = 1;
    inline static constexpr int VerticalSpacer       = 16;
    inline static constexpr int VerticalItemSpacer   = 12;
};


// ** NauSimpleSpoiler
//
// A spoiler is a widget that can hide and reveal the content that is stored in it.
// Despite NauSpoiler no heavy animation attached to expanding/collapsing operations.
// Plus it allows to rename its title by double clicking. See setTitleEditable().

class NAU_EDITOR_API NauSimpleSpoiler : public NauFrame
{
    Q_OBJECT

public:
    NauSimpleSpoiler(const QString& title = QString(), QWidget* parent = nullptr);

    NauMiscButton* addHeaderButton();

    void setExpanded(bool expanded = true);

    void addWidget(QWidget* widget);

    void setHeaderFixedHeight(int fixedHeight);
    void setHeaderContentMargins(QMargins contentMargins);
    void setHeaderHorizontalSpace(int horizontalSpacing);
    void setHeaderPalette(NauPalette palette);
    void setToggleIconSize(QSize size);

    void setContentAreaMargins(QMargins contentMargins);
    void setContentVerticalSpacing(int vSpacing);
    void setContentPalette(NauPalette palette);

    void setTitleEditable(bool editable);
    void setTitle(const QString& title);
    void setTitleEditorOuterMargin(QMargins outerMargins);

signals:
    // Emitted when user double clicks on the title and enters a new title.
    // New title is not automatically applied to header.
    // Use setTitle() if entered @newName is acceptable.
    void eventRenameRequested(const QString& newName);

protected:
    NauFrame* m_headerArea = nullptr;
    NauLayoutHorizontal* m_headerLayout = nullptr;
    NauFrame* m_contentArea = nullptr;
    NauLayoutVertical* m_contentLayout = nullptr;
    NauStaticTextLabel* m_label = nullptr;

    NauMiscButton* m_toggleButton = nullptr;

    NauLineEdit* m_editor = nullptr;

    QMargins m_titleEditorOuterMargins = QMargins(40, 0, 40, 0);
    QSize m_toggleIconSize{24, 24};
};
