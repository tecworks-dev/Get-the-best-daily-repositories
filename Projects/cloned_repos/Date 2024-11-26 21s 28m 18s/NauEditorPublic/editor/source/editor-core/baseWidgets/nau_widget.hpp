// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Set of Nau wrappers of base Qt classes.
// Ideally, no naked Qt classes should be used outside of this file.

#pragma once

#include "nau/nau_editor_config.hpp"

#include "nau_menu.hpp"
#include "nau_palette.hpp"
#include "themes/nau_widget_style.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMenuBar>
#include <QPainter>
#include <QParallelAnimationGroup>
#include <QProgressBar>
#include <QProxyStyle>
#include <QPushButton>
#include <QRadioButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedLayout>
#include <QStyledItemDelegate>
#include <QTableWidget>
#include <QTimeEdit>
#include <QToolButton>
#include <QTreeWidget>
#include <QWheelEvent>
#include <QWidget>
#include <QWidgetAction>


// ** NauMainWindow

class NAU_EDITOR_API NauMainWindow : public QMainWindow
{
public:
    NauMainWindow();
};


// ** NauWidget

class NAU_EDITOR_API NauWidget : public QWidget
{
    Q_OBJECT

public:
    NauWidget(QWidget* parent = nullptr);

signals:
    void eventWidgetClosed();

protected:
    void closeEvent(QCloseEvent* event) override;

    // TODO: Add std::unordered_map<NauWidgetState, NauWidgetStyle::NauStyle> m_styleMap
};


// ** NauFrame

class NAU_EDITOR_API NauFrame : public QFrame
{
    Q_OBJECT

public: 
    NauFrame(QWidget* parent = nullptr);

    void paintEvent(QPaintEvent*) override;

    void setPalette(NauPalette palette);

private:
    NauPalette m_palette;
};


// ** NauWidgetAction

class NAU_EDITOR_API NauWidgetAction : public QWidgetAction
{
    Q_OBJECT

public:
    NauWidgetAction(QWidget* parent = nullptr);

};


// ** Nau3DWidget

class QOffscreenSurface;
class QOpenGLContext;
class QOpenGLFramebufferObject;

class NAU_EDITOR_API Nau3DWidget : public QLabel
{
    Q_OBJECT
public:
    explicit Nau3DWidget(QWidget* parent);
    ~Nau3DWidget() noexcept override;

    void paintEvent(QPaintEvent* event) override;

protected:
    virtual void onRender() = 0;

    [[nodiscard]] bool isValid() const;
    [[nodiscard]] QOffscreenSurface* surface() const;
    [[nodiscard]] QOpenGLContext* context() const;

    void resetContext();
    void makeCurrent();
    void doneCurrent();
    void render();

private:
    QOffscreenSurface* m_surface = nullptr;
    QOpenGLContext* m_context = nullptr;
    QOpenGLFramebufferObject* m_framebuffer = nullptr;
};


// ** NauLayout

template<typename LayoutType>
class NauLayout : public LayoutType
{
    static_assert(std::is_base_of_v<QLayout, LayoutType>, "Has to be a subclass of QLayout");

public:
    using LayoutType::LayoutType;

    NauLayout(NauWidget* parent = nullptr) : LayoutType(parent)
    {
        LayoutType::setContentsMargins(0, 0, 0, 0);
        LayoutType::setSpacing(0);
    }

    void clear() 
    {
        std::function<void(QLayout* layout)> clearLevelRecursively{};

        clearLevelRecursively = [&clearLevelRecursively](QLayout* layout) {
            QLayoutItem* item = nullptr;
            while ((item = layout->takeAt(0)) != nullptr) {
                if (item->layout()) {
                    clearLevelRecursively(item->layout());
                } else if (item->widget()) {
                    item->widget()->deleteLater();
                }

                delete item;
            }
        };

        clearLevelRecursively(this);
    }
};

using NauLayoutHorizontal = NauLayout<QHBoxLayout>;
using NauLayoutVertical = NauLayout<QVBoxLayout>;
using NauLayoutStacked = NauLayout<QStackedLayout>;
using NauLayoutGrid = NauLayout<QGridLayout>;


// ** NauPainter

class NAU_EDITOR_API NauPainter : public QPainter
{
public:
    NauPainter(QPaintDevice* device);
};


// ** NauScrollWidget
// 
// Widget that has a built-in scroll bar.
// Supports vertical and horizontal scrolling.

template<typename LayoutType>
class NauScrollWidget : public QScrollArea
{
    static_assert(std::is_base_of_v<QLayout, LayoutType>, "Has to be a subclass of QLayout");

public:
    NauScrollWidget(NauWidget* parent = nullptr)
        : QScrollArea(parent)
        , m_layout(new LayoutType)
    {
        setFrameStyle(QFrame::NoFrame);
        auto scrollWidget = new NauWidget();
        scrollWidget->setLayout(m_layout);
        setWidget(scrollWidget);
        setWidgetResizable(true);
        m_layout->addStretch(1);
    }

    void addWidget(QWidget* widget)
    {
        const auto nWidgets = m_layout->children().size();
        m_layout->insertWidget(nWidgets, widget);   // Append to the end of the list, before the stretch
    }

    LayoutType* layout()
    {
        return m_layout;
    }

private:
    LayoutType* m_layout;
};

using NauScrollWidgetVertical = NauScrollWidget<NauLayoutVertical>;
using NauScrollWidgetHorizontal = NauScrollWidget<NauLayoutHorizontal>;


// ** NauMenuBar

class NAU_EDITOR_API NauMenuBar : public NauWidget
{
public:
    NauMenuBar(NauWidget* parent);
    auto base() const { return m_bar; }

    void addMenu(NauMenu* menu);

private:
    QMenuBar* m_bar;
};


// ** NauDialog

class NAU_EDITOR_API NauDialog : public QDialog
{
public:
    NauDialog(QWidget* parent);

    virtual int showModal();
    virtual void showModeless();
};


// ** NauDialogButtonBox

class NAU_EDITOR_API NauDialogButtonBox : public QDialogButtonBox
{
    Q_OBJECT

public:
    NauDialogButtonBox(NauDialog* parent = nullptr);
};


// ** NauSplitter

class NAU_EDITOR_API NauSplitter: public QSplitter
{
    Q_OBJECT

public:
    NauSplitter(NauWidget* parent = nullptr);
}; 


// ** NauListView

class NAU_EDITOR_API NauListView: public QListView
{
    Q_OBJECT

public:

    NauListView(QWidget* parent = nullptr);
};


// ** NauListWidget

class NAU_EDITOR_API NauListWidget: public QListWidget
{
    Q_OBJECT

public:
    NauListWidget(NauWidget* parent = nullptr);

};


// ** NauTreeView

class NAU_EDITOR_API NauTreeView: public QTreeView
{
    Q_OBJECT

public:
    NauTreeView(QWidget* parent = nullptr);
};


// ** NauTreeWidgetItem forward declaration

class NauTreeWidgetItem;


// ** NauTreeWidget

class NAU_EDITOR_API NauTreeWidget : public QTreeWidget
{
    Q_OBJECT

public:
    NauTreeWidget(NauWidget* parent = nullptr);

    void setupColumn(int columnIdx, const QString& text, const NauFont& font, bool visible = true,
        int width = 100, QHeaderView::ResizeMode resizeMode = QHeaderView::ResizeMode::Interactive, const QString& tooltip = {});

    void setupColumn(int columnIdx, const QString& text, const NauFont& font, const QIcon& icon, bool width = true,
        int size = 100, QHeaderView::ResizeMode resizeMode = QHeaderView::ResizeMode::Interactive, const QString& tooltip = {});
};


// ** NauTreeWidgetItem

class NAU_EDITOR_API NauTreeWidgetItem : public QTreeWidgetItem
{
public:
    NauTreeWidgetItem(NauTreeWidget* parent, const QStringList& strings, int type = QTreeWidgetItem::ItemType::UserType);
};


// ** NauToolButton

class NAU_EDITOR_API NauToolButton : public QToolButton
{
public:
    NauToolButton(QWidget* parent = nullptr);
    
    void setShortcut(const NauKeySequence& shortcut);

};


// ** NauPushButton

class NAU_EDITOR_API NauPushButton : public QPushButton
{
public:
    NauPushButton(NauWidget* parent = nullptr);
};


// ** NauBorderlessButton

class NAU_EDITOR_API NauBorderlessButton : public NauPushButton
{
public:
    NauBorderlessButton(const QIcon& icon, NauMenu* menu, NauWidget* parent = nullptr);
    NauBorderlessButton(NauMenu* menu, NauWidget* parent = nullptr);
};


// ** NauInspectorSubWindow

class NAU_EDITOR_API NauInspectorSubWindow : public NauWidget
{
    Q_OBJECT

public:
    explicit NauInspectorSubWindow(QWidget* parent);

    void setText(const QString& text);
    void setContentLayout(QLayout& contentLayout);

private:
    void expand(bool checked);

private:
    static constexpr QMargins COMMON_MARGIN{ 0, 0, 0, 0 };
    static constexpr QMargins CONTENT_MARGIN{ 16, 8, 16, 16 };
    static constexpr int TITLE_HEIGHT = 56;

    NauWidget* m_titleContainer = nullptr;
    NauToolButton* m_titleButton = nullptr;
    NauScrollWidgetVertical* m_content = nullptr;
};

class NauLineEdit;

// ** NauLineEditStateListener

class NAU_EDITOR_API NauLineEditStateListener : public QObject
{
public:
    enum class EditFlagState
    {
        True,
        False,
        Pass
    };
    explicit NauLineEditStateListener(const NauLineEdit& lineEdit);

    bool eventFilter(QObject* object, QEvent* event) override;

    void setState(NauWidgetState WidgetState, EditFlagState flagState) noexcept;
    [[nodiscard]]
    NauWidgetState state() const noexcept { return m_currentState; }
    [[nodiscard]]
    bool editing() const noexcept { return m_isEditing; }

private:
    const NauLineEdit* m_lineEdit;
    NauWidgetState m_currentState;
    bool m_isEditing;
};


// ** NauLineEdit

class NAU_EDITOR_API NauLineEdit : public QLineEdit
{
public:
    explicit NauLineEdit(QWidget* parent = nullptr);

    [[nodiscard]]
    const NauLineEditStateListener& stateListener() const noexcept { return m_listener; }
    [[nodiscard]]
    NauWidgetState state() const noexcept { return m_listener.state(); }
    [[nodiscard]]
    bool editing() const noexcept { return m_listener.editing(); }

private:
    inline static constexpr int Height = 24;

    NauLineEditStateListener m_listener;
};


// ** NauSlider

class NAU_EDITOR_API NauSlider: public QSlider
{
public:
    NauSlider(QWidget* parent = nullptr);

    void setPalette(const NauPalette& palette);
    const NauPalette& nauPalette() const;
protected:
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;

    bool hovered() const;

private:
    NauPalette m_palette;
    bool m_hovered = false;
};


// ** NauCheckBox

class NAU_EDITOR_API NauCheckBox : public QCheckBox
{
    Q_OBJECT

public:
    NauCheckBox(QWidget* parent = nullptr);
    NauCheckBox(const QString& text, QWidget* parent = nullptr);
};


// ** NauComboBox

class NAU_EDITOR_API NauComboBox : public QComboBox
{
    Q_OBJECT

public:
    NauComboBox(QWidget* parent = nullptr);

protected:
    void wheelEvent(QWheelEvent* event) override;
};


// ** NauSpinBox

class NAU_EDITOR_API NauSpinBox : public QSpinBox
{
    Q_OBJECT

public:
    explicit NauSpinBox(QWidget* parent = nullptr);

    NauLineEdit* lineEdit() const noexcept { return m_lineEdit; }

protected:
    void wheelEvent(QWheelEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    inline static constexpr int Height = 32;

    NauLineEdit* m_lineEdit;
};


// ** NauDoubleSpinBox

class NAU_EDITOR_API NauDoubleSpinBox : public QDoubleSpinBox
{
    Q_OBJECT

public:
    explicit NauDoubleSpinBox(QWidget* parent = nullptr);

    NauLineEdit* lineEdit() const noexcept { return m_lineEdit; }

protected:
    void wheelEvent(QWheelEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    inline static constexpr int Height = 32;
    NauLineEdit* m_lineEdit;
};


// ** NauTimeEdit

class NAU_EDITOR_API NauTimeEdit : public QTimeEdit
{
    Q_OBJECT

public:
    explicit NauTimeEdit(QWidget* parent = nullptr);

    NauLineEdit* lineEdit() const noexcept { return m_lineEdit; }

protected:
    void wheelEvent(QWheelEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    inline static constexpr int Height = 32;
    NauLineEdit* m_lineEdit;
};


// ** NauMultiValueSpinBox
//
// Serves for displaying values in the form of multiple spin box fields

class NAU_EDITOR_API NauMultiValueSpinBox : public NauWidget
{
    Q_OBJECT

public:
    NauMultiValueSpinBox(NauWidget* parent, int valuesCount);
    NauMultiValueSpinBox(NauWidget* parent = nullptr, QStringList valuesNames = QStringList());

    NauSpinBox* operator[] (int index) const;

    void setMinimum(int min);
    void setMaximum(int max);

signals:
    void eventValueChanged();

private:
    NauWidget* m_valueWidget;
    NauLayoutGrid* m_valueLayout;

    std::vector<NauSpinBox*> m_spinBoxes;

    inline static constexpr int VerticalSpacer = 8;

    inline static constexpr int BoxFrameWidth = 13;
    inline static constexpr int LineFrameWidth = 2;
    inline static constexpr int LineFrameHeight = 24;
    inline static constexpr int LineWidgetWidth = 10;

    inline static constexpr int FirstColumnWidth = 13;
    inline static constexpr int ThirdColumnWidth = 16;
};


// ** NauMultiValueDoubleSpinBox
//
// Serves for displaying values in the form of multiple floating spin box fields

class NAU_EDITOR_API NauMultiValueDoubleSpinBox : public NauWidget
{
    Q_OBJECT

public:
    NauMultiValueDoubleSpinBox(NauWidget* parent, int valuesCount);
    NauMultiValueDoubleSpinBox(NauWidget* parent = nullptr, QStringList valuesNames = QStringList());

    NauDoubleSpinBox* operator[] (int index) const;

    void setMinimum(double min);
    void setMaximum(double max);

    void setDecimals(int decimalPrecision);

signals:
    void eventValueChanged();

private:
    NauWidget* m_valueWidget;
    NauLayoutGrid* m_valueLayout;

    std::vector<NauDoubleSpinBox*> m_spinBoxes;

    inline static constexpr int VerticalSpacer = 8;

    inline static constexpr int BoxFrameWidth = 13;
    inline static constexpr int LineFrameWidth = 2;
    inline static constexpr int LineFrameHeight = 24;
    inline static constexpr int LineWidgetWidth = 10;

    inline static constexpr int FirstColumnWidth = 13;
    inline static constexpr int ThirdColumnWidth = 16;
};


// ** NauColorDialog
//
// The NauColorDialog class provides the ability to select colors based on RGB, HSV, or CMYK values.
// In addition, the widget remembers the last selected color so that the user can get it without accessing the dialog box.

class NAU_EDITOR_API NauColorDialog : public NauWidget
{
    Q_OBJECT

public:
    NauColorDialog(NauWidget* parent = nullptr);

    void colorDialogRequested();
    QColor color() const;

    // Allows you to set the initial color from outside the widget, without calling the dialog box
    void setColor(const QColor& currentColor);

signals:
    void eventColorChanged(QColor color);

private:
    QColor m_color;
};


// ** NauTableHeaderIconDelegate
//
// The NauTableHeaderIconDelegate class provides display facilities from the model for the NauTableWidget.

class NAU_EDITOR_API NauTableHeaderIconDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    NauTableHeaderIconDelegate(QObject* parent = nullptr);
    void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const override;
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;

private:
    const int m_pixmapLeftPadding = -6;
};


// ** NauProxyStyle
//
// Base class for working with proxy styles

class NAU_EDITOR_API NauProxyStyle : public QProxyStyle
{
public:
    NauProxyStyle(QStyle* style = nullptr);
};

// TODO: Move all of the table related stuff into separate files

// ** NauTableWidgetItem
//
// Table widgetItem for for storing data from a data model.

class NAU_EDITOR_API NauTableWidgetItem : public QTableWidgetItem
{
public:
    NauTableWidgetItem();
    NauTableWidgetItem(const QIcon& icon);
    NauTableWidgetItem(const QString& text);
    NauTableWidgetItem(const QIcon& icon, const QString& text);
};


// ** NauTableProxySyle
//
// Proxy style for table widget

class NAU_EDITOR_API NauTableProxySyle : public NauProxyStyle
{
public:
    NauTableProxySyle(QStyle* style = nullptr);
};


// ** NoFocusNauTableProxyStyle
//
// Proxy style for a table widget that has the default visual primitives disabled when focusing

class NAU_EDITOR_API NauNoFocusNauTableProxyStyle : public NauTableProxySyle
{
public:
    NauNoFocusNauTableProxyStyle(QStyle* style = nullptr);

    void drawPrimitive(PrimitiveElement element, const QStyleOption* option, QPainter* painter, const QWidget* widget) const;
};


// ** NauTableHeaderContextMenu
//
// Context menu that is used to hide the columns of the World Outline table.

class NAU_EDITOR_API NauTableHeaderContextMenu : public NauMenu
{
public:
    NauTableHeaderContextMenu(const QString& title, NauWidget* widget = nullptr);

    NauCheckBox* addAction(const QString& title, bool isChecked);
};


// ** NauTableWidget
//
// Table widget for displaying data from a data model.

class NAU_EDITOR_API NauTableWidget : public QTableWidget
{
public:
    NauTableWidget(QWidget* parent = nullptr);

    virtual void addColumn(const QString& titleName, bool visibilityStatus = true,
        int size = 100, QHeaderView::ResizeMode mode = QHeaderView::ResizeMode::Interactive, const QString& tooltip = {});
    virtual void addColumn(const QString& titleName, const QIcon& columnIcon,
        bool visibilityStatus = true, int size = 100, QHeaderView::ResizeMode mode = QHeaderView::ResizeMode::Interactive, const QString& tooltip = {});
    
    void changeColumnVisibility(int column, bool visibilityStatus);

private:
    void addColumnHeaderItem(QTableWidgetItem* header, bool visibilityStatus, int size, QHeaderView::ResizeMode mode);

protected:
    const int m_rowHeight = 40;
    const int m_horizontalHeaderheight = 34;
};


// ** NauLineWidget
//
// Widget with line vertical/horizontal line inside

class NAU_EDITOR_API NauLineWidget : public NauWidget
{
    Q_OBJECT

public:

    NauLineWidget(const QColor& lineColor, int lineWidth, Qt::Orientation orientation, QWidget* parent = nullptr);

    void setOffset(int offset);
protected:

    void paintEvent(QPaintEvent*) override;

    QColor m_lineColor;
    int m_lineWidth;
    Qt::Orientation m_orientation;
    int m_offset;
};


// ** NauSpacer
// An empty widget for fixed spacing between widgets.
// Use it instead of QSpaceItem if it requires to have a custom background color for it.

class NAU_EDITOR_API NauSpacer : public NauFrame
{
public:
    NauSpacer(Qt::Orientation orientation = Qt::Horizontal, int size = 4, QWidget* parent = nullptr);
};

// ** NauRadioButton
//
// Widget provides a radio button with a text label

class NAU_EDITOR_API NauRadioButton : public QRadioButton
{
    Q_OBJECT

public:
    explicit NauRadioButton(NauWidget* parent = nullptr);
    explicit NauRadioButton(const QString& text, NauWidget* parent = nullptr);
};


// ** NauProgressBar

class NAU_EDITOR_API NauProgressBar : public QProgressBar
{
    Q_OBJECT

public:
    explicit NauProgressBar(QWidget* parent = nullptr);
};


// ** NauToogleButton
//
// Widget provides a toogle button based on QCheckBox

class NAU_EDITOR_API NauToogleButton : public QCheckBox
{
    Q_OBJECT

public:
    // TODO Add animations
    explicit NauToogleButton(NauWidget* parent = nullptr);

    void setState(NauWidgetState state);
    void setStateStyle(NauWidgetState state, NauWidgetStyle::NauStyle style);

    void setChecked(bool isChecked);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    void toggle();
    void updateWidgetState();

    QPainterPath getOutlinePath(NauWidgetState state);

private:
    inline static constexpr int Radius = 2;
    inline static constexpr int Offset = 2;

private:
    NauWidgetState m_currentState;
    std::unordered_map<NauWidgetState, NauWidgetStyle::NauStyle> m_styleMap;
};


// TODO: Make Nau wrappers for the relevant components:
//    QScrollbar
//    
//    QTimer
