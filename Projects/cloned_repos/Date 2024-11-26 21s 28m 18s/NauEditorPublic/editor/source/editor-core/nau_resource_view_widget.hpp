// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// This file will contain everything related to the ComboBox, which is able to display various resources of the engine.

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_static_text_label.hpp"
#include "baseWidgets/nau_buttons.hpp"
#include "filter/nau_search_widget.hpp"
#include "nau_assert.hpp"
#include "nau/assets/asset_db.h"
#include "nau/assets/nau_file_types.hpp"


// At the abstract level are the classes that define the basic behavior
// of all NauComboBox components.
#pragma region ABSTRACT LEVEL

// ** NauPopupAbstractWidgetHeader
// 
// The header class necessarily has a search string for the elements of the popup,
// as well as provides an interface for adding auxiliary buttons.

class NAU_EDITOR_API NauPopupAbstractWidgetHeader : public NauWidget
{
    Q_OBJECT

public:
    NauPopupAbstractWidgetHeader(NauWidget* parent = nullptr);

    void setPlaceholderText(const QString& placeholderText);
    QString placeholderText() const;

signals:
    void eventSearchFilterChanged(const QString&);

protected:
    NauLayoutHorizontal* m_layout;
    NauLayoutHorizontal* m_additinalButtonsLayout;
    NauSearchWidget* m_search;

    inline static constexpr int HeaderWidht = 264;
    inline static constexpr int HeaderHeight = 48;
    inline static constexpr int OuterMargin = 8;
    inline static constexpr int InnerMargin = 4;
    inline static constexpr int Spacing = 8;

protected:
    // This is necessary so that you can't create an instance of this class.
    // In the descendants we paint the background of the widget and call NauWidget::paintEvent.
    virtual void paintEvent(QPaintEvent* event) = 0;

};


// ** NauPopupAbstractTreeWidget
// 
// In the middle part of the popup widget there is a tree structure of data in a tabular view.

class NAU_EDITOR_API NauPopupAbstractTreeWidget : public NauTreeWidget
{
    Q_OBJECT

public:
    NauPopupAbstractTreeWidget(NauWidget* parent = nullptr);

    virtual QTreeWidgetItem* addItem(const QString& item, NauEditorFileType type) = 0;
    virtual void addItems(const std::vector<QString>& items, NauEditorFileType type) = 0;

    virtual void insertItem(int index, const QString& item, NauEditorFileType type) = 0;
    virtual void insertItems(int index, const std::vector<QString>& items, NauEditorFileType type) = 0;

    void clearData();
    size_t count() const;
    QString currentData(QTreeWidgetItem* index) const;

protected:
    std::unordered_map<QTreeWidgetItem*, QString> m_items;

};


// ** NauPopupAbstractWidgetFooter
// 
// The bottom part of the pop-up widget contains information
// about the total number of objects and how many of them are selected.

class NAU_EDITOR_API NauPopupAbstractWidgetFooter : public NauWidget
{
    Q_OBJECT

public:
    NauPopupAbstractWidgetFooter(NauWidget* parent = nullptr);

    void setText(const QString& text);

protected:
    NauLayoutHorizontal* m_layout;
    NauStaticTextLabel* m_text;

    inline static constexpr int HeaderWidht = 264;
    inline static constexpr int HeaderHeight = 32;
    inline static constexpr int OuterMarginHeight = 8;
    inline static constexpr int OuterMarginWidth = 16;

protected:
    // This is necessary so that you can't create an instance of this class.
    // In the descendants we paint the background of the widget and call NauWidget::paintEvent.
    virtual void paintEvent(QPaintEvent* event) = 0;

};


// ** NauAbstractPopupWidget
// 
// The popup widget class combines all the components and ensures that they work with each other.

class NAU_EDITOR_API NauAbstractPopupWidget : public NauWidget
{
public:
    NauAbstractPopupWidget(NauPopupAbstractWidgetHeader* header, NauPopupAbstractTreeWidget* container, NauPopupAbstractWidgetFooter* footer, NauWidget* parent = nullptr);

    inline NauPopupAbstractWidgetHeader& header() { return *m_header; }
    inline NauPopupAbstractTreeWidget& container() { return *m_container; }
    inline NauPopupAbstractWidgetFooter& footer() { return *m_footer; }

    virtual void updateFilterData(const QString& filter) = 0;

protected:
    NauPopupAbstractWidgetHeader* m_header;
    NauPopupAbstractTreeWidget* m_container;
    NauPopupAbstractWidgetFooter* m_footer;
    NauLayoutVertical* m_layout;

    inline static constexpr int HeaderWidht = 264;
    inline static constexpr int HeaderHeight = 472;

};

// ** NauAbstractComboBox
// 
// This widget is able to correctly display a popup window depending on the screen
// on which the application is located,
// as well as provides a convenient interface for working with data from the engine.

class NAU_EDITOR_API NauAbstractComboBox : public NauPrimaryButton
{
    Q_OBJECT
public:
    NauAbstractComboBox(NauWidget* parent = nullptr);

    size_t count() const;

    void setPlaceholderText(const QString& placeholderText);
    QString placeholderText() const;

    void addItem(const QString& item, nau::Uid uid, const std::string& primPath, NauEditorFileType type);
    void addItems(const std::vector<QString>& items, NauEditorFileType type);
    void addClearSelectionButton();

    virtual void clear();
    void clearSelection();

signals:
    void eventSelectionChanged();
    void eventSelectionChanged(const QString& itemText);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent *event) override;

    virtual void showPopup();
    virtual void onPressed() override;
    
    QString currentData(QTreeWidgetItem* item) const;

protected:
    NauAbstractPopupWidget* m_container;

    std::unordered_map<QTreeWidgetItem*, nau::Uid> m_dataUid;
    std::unordered_map<QTreeWidgetItem*, std::string> m_dataPrimPath;

    inline static constexpr int OuterMarginHeight = 8;
    inline static constexpr int OuterMarginWidth = 16;
    inline static constexpr int Gap = 8;
    QSize ButtonSize{16, 16};

private:
    NauMiscButton* m_cleanButton = nullptr;
};

#pragma endregion


// At the component level there are parts of the ComboBox that are responsible
// for displaying specific data from the engine.
#pragma region COMPONENT LEVEL

// ** NauPopupResourceHeader
// 
// Header widget for work with engine resources.

class NAU_EDITOR_API NauPopupResourceHeader final : public NauPopupAbstractWidgetHeader
{
public:
    NauPopupResourceHeader(NauWidget* parent = nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;

};


// ** NauResourceTreeWidget
// 
// Table widget for displaying engine resources: models, materials, textures, etc.

class NAU_EDITOR_API NauResourceTreeWidget final : public NauPopupAbstractTreeWidget
{
    Q_OBJECT

public:
    enum class Columns
    {
        Preview = 0,
        Name = 1,
    };

    NauResourceTreeWidget(NauWidget* parent = nullptr);

    QTreeWidgetItem* addItem(const QString& item, NauEditorFileType type) override;
    void addItems(const std::vector<QString>& items, NauEditorFileType type) override;

    void insertItem(int index, const QString& item, NauEditorFileType type) override;
    void insertItems(int index, const std::vector<QString>& items, NauEditorFileType type) override;

protected:

    inline static constexpr int HeaderIconSize = 16;
    inline static constexpr int ContentIconSize = 48;
    inline static constexpr int HeaderHeight = 32;
    inline static constexpr int RowHeight = 72;
    inline static constexpr int PreviewColumnWight = 72;
    inline static constexpr int NameColumnWight = 100;
    inline static constexpr int RowContentsMarginsHight = 12;
    inline static constexpr int RowContentsMarginsWight = 16;
    inline static constexpr int CellContentsMargins = 4;
    inline static constexpr int Spacing = 8;

private:
    QMap<NauEditorFileType, QIcon> m_iconsRepository;
};


// ** NauPopupResorceFooter
// 
// Footer widget for work with engine resources.

class NAU_EDITOR_API NauPopupResorceFooter final : public NauPopupAbstractWidgetFooter
{
public:
    NauPopupResorceFooter(NauWidget* parent = nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;

};


// ** NauResourcePopupWidget
// 
// Popup widget customized to display engine resources.

class NAU_EDITOR_API NauResourcePopupWidget final : public NauAbstractPopupWidget
{
public:
    NauResourcePopupWidget(NauWidget* parent = nullptr);
    void updateFilterData(const QString& filter) override;

protected:
    void paintEvent(QPaintEvent* event) override;

};
#pragma endregion


// At the widget level there are widgets already fully ready for integration into other systems.
#pragma region WIDGET LEVEL

// ** NauResourceComboBox
// 
// ComboBox widget for working with engine resources.

class NAU_EDITOR_API NauResourceComboBox: public NauAbstractComboBox
{
public:
    NauResourceComboBox(NauWidget* parent = nullptr);

    nau::Uid getCurrentUid() { return m_currentContainerUid; };
    std::string getCurrentPrimPath() { return m_currentPrimPath; };

    void setCurrentData(nau::Uid containerUid, const std::string& primPath)
    {
        m_currentContainerUid = containerUid;
        m_currentPrimPath = primPath;
    };

    void clear() override;

private:
    nau::Uid m_currentContainerUid;
    std::string m_currentPrimPath;
};

#pragma endregion
