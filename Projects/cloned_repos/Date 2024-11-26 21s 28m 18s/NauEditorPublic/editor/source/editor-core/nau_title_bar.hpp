// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Main window title bar, that contains custom widgets.


#pragma once

#include "nau_font.hpp"
#include "nau_font_metrics.hpp"
#include "baseWidgets/nau_icon.hpp"
#include "nau_main_menu.hpp"
#include "nau_palette.hpp"
#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_label.hpp"
#include "nau/compiler/nau_source_state.hpp"

#include <QStaticText>


// ** NauTitleBarMenuItem

class NAU_EDITOR_API NauTitleBarMenuItem : public NauWidget
{
    Q_OBJECT

    friend class NauTitleBarMenu;

protected:
    NauTitleBarMenuItem(const QString& name,
        const NauMenu* menu, NauWidget* parent);

    void paintEvent(QPaintEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

    static constexpr int VerticalPadding = 16;

signals:
    void eventPressed();

private:
    const QString m_name;
    const NauMenu* m_menu;

    bool m_hover = false;
    bool m_menuIsShown = false;
    bool m_showNextMenu = true;
};


// ** NauTitleBarMenu

class NAU_EDITOR_API NauTitleBarMenu : public NauWidget
{
    Q_OBJECT

public:
    NauTitleBarMenu(NauShortcutHub* shortcutHub, NauWidget* parent);

    std::unique_ptr<NauMainMenu>& menu();

protected:
    void enterEvent(QEnterEvent* event) override;

private:
    std::unique_ptr<NauMainMenu> m_menu;
};


// ** NauTitleBarTooltip

class NAU_EDITOR_API NauTitleBarTooltip : public NauFrame
{
    Q_OBJECT

public:
    NauTitleBarTooltip(QWidget* parent);

    void setContent(const QString& title, const QString& content);

signals:
    void eventClosed();

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    NauLabel* m_title = nullptr;
    NauLabel* m_content = nullptr;
};


// ** NauTitleBarCompilationState

class NAU_EDITOR_API NauTitleBarCompilationState : public NauWidget
{
    Q_OBJECT

public:
    NauTitleBarCompilationState(NauWidget* parent);

    void setCompilationState(const NauSourceStateManifold& state);

protected:
    void paintEvent(QPaintEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    void updateTooltipUi();

    const NauPalette m_palette;
    NauSourceState m_compilationState = NauSourceState::NoBuildTools;
    bool m_hover = false;
    std::unordered_map<NauSourceState, NauIcon> m_stateIcons;
    NauIcon m_icon;
    QStaticText m_text;
    QMargins m_contentMargins;
    QSize m_iconSize;

    QPointer<NauTitleBarTooltip> m_tooltip;
};


// ** NauTitleBar

class NAU_EDITOR_API NauTitleBar : public NauWidget
{
    Q_OBJECT

public:
    NauTitleBar(NauShortcutHub* shortcutHub, QWidget* parent);

    std::unique_ptr<NauMainMenu>& menu();
    void updateTitle(const QString& projectName, const QString& sceneName);
    void setMaximized(bool maximized);
    bool maximized() const;

    bool moveRequestPending() const;
    void resetMoveAndResizeRequests();

    void setCompilationState(const NauSourceStateManifold& state);

    static constexpr int Height = 40;
    static constexpr int SourceStateSpace = 32;
    static constexpr int ResizeOffset = 6;

signals:
    void eventMinimize();
    void eventClose();
    void eventToggleWindowStateRequested(bool maximized);

    // Emitted when user tries to move main window by holding this title bar.
    void eventMoveWindowRequested();

    // Emitted when user tries to resize main window by the top edge.
    void eventResizeWindowRequested();

protected:
    void paintEvent(QPaintEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    bool resizingAcceptable(const QPoint& point) const;

private:
    class NauTitleBarSystemButton;
    enum class WindowManipulationState
    {
        None,
        Resizing,
        Moving,
    };

    NauTitleBarSystemButton* m_buttonMinimize;
    NauTitleBarSystemButton* m_buttonClose;
    NauTitleBarSystemButton* m_buttonWindowState;
    NauTitleBarCompilationState* m_compilationState;
    NauTitleBarMenu* m_menu;

    const NauIcon m_logoIcon;
    const NauPalette m_palette;

    QString m_projectName;
    QRect m_projectNameBoundingRect;

    QString m_sceneName;
    QRect m_sceneNameBoundingRect;

    bool m_sourceModifiedIconVisible = false;

    // Current operation on the owner window that is being requested by a user via this title bar.
    WindowManipulationState m_windowManipulationState = WindowManipulationState::None;
};
