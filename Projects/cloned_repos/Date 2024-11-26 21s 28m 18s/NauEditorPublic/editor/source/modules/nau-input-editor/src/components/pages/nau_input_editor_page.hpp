// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Input editor classes

#pragma once

#include "nau/rtti/rtti_impl.h"

#include "nau_dock_widget.hpp"
#include "baseWidgets/nau_spoiler.hpp"

#include <QFileSystemWatcher>
#include <QGroupBox>
#include <QRadioButton>

#include <string>

#include "baseWidgets/nau_buttons.hpp"

#include "nau/input_system.h"
#include "nau/service/service_provider.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/relationship.h>


class NauComboBox;
class NauDockManager;
class NauEntityInspectorPage;
class NauLabel;
class NauStaticTextLabel;
class NauRadioButton;
class NauWidget;
class NauPrimaryButton;
class NauPropertyAbstract;


// ** NauInputEditorLineEdit

class NauInputEditorLineEdit : public NauLineEdit
{
    Q_OBJECT

public:
    explicit NauInputEditorLineEdit(const QString& text, NauLayoutHorizontal& layout, NauSpoiler& parent);

    NauMiscButton* createButtonAt(int index, const NauIcon& icon, bool needNotifySpoiler = false);
    void setTextEdit(bool editFlag) noexcept;
    void switchToggle();
    bool isOutlineDrawn() const;

    void mousePressEvent(QMouseEvent* event) override;

    [[nodiscard]]
    NauMiscButton* arrowButton() const noexcept;
    [[nodiscard]]
    NauMiscButton* addButton() const noexcept;
    [[nodiscard]]
    NauMiscButton* closeButton() const noexcept;

signals:
    void eventBindNameChanged(const QString& newName);
    void eventToggle(bool toggleFlag);
    void eventHoveredChanged(bool isHovered);

protected:
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    void notifySpoiler();

    void paintEvent(QPaintEvent* event) override;

private:
    NauSpoiler* m_spoiler;
    NauLayoutHorizontal* m_container;
    NauMiscButton* m_arrowButton;
    NauMiscButton* m_closeButton;
    NauMiscButton* m_addButton;
    bool m_expanded;
    bool m_canTextEdit;
};


// ** NauInputEditorHeaderContainer

class NauInputEditorHeaderContainer : public NauWidget
{
    Q_OBJECT
public:
    NauInputEditorHeaderContainer(const QString& text, NauSpoiler& parent);

    [[nodiscard]]
    NauInputEditorLineEdit* lineEdit() const noexcept;

private:
    NauLayoutHorizontal* m_layout;
    NauInputEditorLineEdit* m_lineEdit;
};


// ** NauInputEditorSpoiler

class NauInputEditorSpoiler : public NauSpoiler
{
public:
    explicit NauInputEditorSpoiler(const QString& text, int duration, bool hasHeaderLine, NauWidget* parent);

    void setText(const QString& text);

protected:
    NauInputEditorHeaderContainer* m_headerContainerWidget;
};


// ** NauInputEditorSpoilerContentLayout

class NauInputEditorSpoilerContentLayout : public NauLayoutVertical
{
protected:
    [[nodiscard]]
    int calculateHeight() const noexcept;
    [[nodiscard]]
    QSize sizeHint() const override;
};


// ** NauInputEditorTabBar

class NauInputEditorTabs : public QTabBar
{
    Q_OBJECT
public:
    explicit NauInputEditorTabs(QWidget* parent);

    [[nodiscard]]
    QSize minimumTabSizeHint(int index) const override;
    [[nodiscard]]
    QSize tabSizeHint(int index) const override;

    void wheelEvent(QWheelEvent* event) override {}
};


// ** NauInputEditorTabContent

class NauInputEditorTabContent : public NauInputEditorSpoiler
{
    Q_OBJECT
public:
    explicit NauInputEditorTabContent();
};


// ** NauInputModifier

enum class NauInputModifier : uint8_t
{
    Scale,
    DeadZone,
    Clamp
};


// ** NauInputEditorTabWidget

class NauInputEditorBaseView;
class NauInputEditorSignalView;
class NauInputEditorModifierView;


// ** NauInputEditorTabWidget

class NauInputEditorTabWidget : public QTabWidget
{
    Q_OBJECT
public:
    explicit NauInputEditorTabWidget(QWidget* parent);

    void addSignalView(NauInputEditorSignalView* view);
    void deleteSignalView(const NauInputEditorSignalView* view);

    void addModifierView(NauInputEditorModifierView* view);
    void deleteModifierView(const NauInputEditorModifierView* view);

private:
    enum class Reason : uint8_t
    {
        Add,
        Delete
    };

    void setupTab(NauInputEditorTabContent* tab);

    void updateNoLabel(Reason reason, NauWidget*& labelContainer, NauInputEditorTabContent* tab, const QString& text);
    void hideAndDeleteLabel(NauWidget*& labelContainer, NauInputEditorTabContent* tab);
    void showNoItemsLabel(NauWidget*& labelContainer, NauInputEditorTabContent* tab, const QString& text);

    void updateNoSignalsLabel(Reason reason);
    void updateNoModifiersLabel(Reason reason);

    void addView(NauInputEditorBaseView* view, NauInputEditorTabContent* tabContent, NauWidget*& labelContainer, const QString& noItemsText);
    void deleteView(const NauInputEditorBaseView* view, NauInputEditorTabContent* tabContent, NauWidget*& labelContainer, const QString& noItemsText);

private:
    std::unique_ptr<NauInputEditorTabContent> m_signalsTab;
    std::unique_ptr<NauInputEditorTabContent> m_modifiersTab;

    NauWidget* m_noSignalsLabelContainer;
    NauWidget* m_noModifiersLabelContainer;
};


// ** NauInputEditorTabWidgetContainer

class NauInputEditorTabWidgetContainer : public NauWidget
{
public:
    explicit NauInputEditorTabWidgetContainer(NauWidget* parent);

    [[nodiscard]]
    NauInputEditorTabWidget* tabWidget() const noexcept;

private:
    void updateWidgetSize(bool needResize);

    void resizeEvent(QResizeEvent* event);
    bool eventFilter(QObject* tab, QEvent* event) override;

private:
    NauInputEditorTabWidget* m_tabWidget;
};


// ** NauInputEditorPageHeader

class NauInputEditorPageHeader : public NauWidget
{
    Q_OBJECT

public:
    NauInputEditorPageHeader(const QString& title, const QString& subtitle, NauWidget* parent);

    void setTitle(const std::string& title);

private:
    NauStaticTextLabel* m_title;
};


// ** NauInputBindType

enum class NauInputBindType : uint8_t
{
    Digital,
    Analog,
    Axis
};


// ** NauInputDevice

enum class NauInputDevice : uint8_t
{
    Unknown,
    Mouse,
    Keyboard
};


// ** NauInputTrigger

enum class NauInputTrigger : uint8_t
{
    Pressed,
    Released
};


// ** NauInputAxis

enum class NauInputAxis : uint8_t
{
    AxisX,
    AxisY
};


// ** NauInputSignalType

enum class NauInputSignalType : uint8_t
{
    Positive,
    Negative
};


// ** NauInputLinkingType

enum class NauInputLinkingType : uint8_t
{
    Or,
    And,
    Nor,
};


// ** NauInputTriggerCondition

enum class NauInputTriggerCondition : uint8_t
{
    Single,
    Delay,
    Multiple
};


// ** NauInputEditorSignalView

class NauInputEditorBaseView : public NauInputEditorSpoiler
{
    Q_OBJECT

public:
    NauInputEditorBaseView(const pxr::UsdPrim& prim, NauWidget* parent = nullptr);

    [[nodiscard]]
    pxr::UsdPrim prim() const noexcept;

signals:
    void eventViewDeleteRequire(NauInputEditorBaseView* view);
    void eventBindUpdateRequired();

protected:
    struct FieldLayoutInfo
    {
        constexpr FieldLayoutInfo() noexcept;

        const int HEIGHT;
        const int SPACING;
        const int LABEL_PART;
        const int VALUE_PART;
    };

    void updateHover(bool flag);

    void addParamWidget(const QString& paramName, QWidget* widget);

    bool event(QEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

protected:
    // TODO Static factory
    using WidgetCreator = std::function<void(const pxr::UsdPrim&)>;
    std::unordered_map<QString, WidgetCreator> m_widgetFactory;

    void initializeWidgetFactory();

protected:
    pxr::UsdPrim m_prim;
    std::vector<NauWidget*> m_paramsWidgets;

    QLayout* m_containerLayout;
    NauInputEditorLineEdit* m_bindNameWidget;

    bool m_hovered;
};


// ** NauInputEditorSignalView

class NauInputEditorSignalView final : public NauInputEditorBaseView
{
    Q_OBJECT

public:
    explicit NauInputEditorSignalView(const pxr::UsdPrim& signalPrim, NauInputBindType bindType, NauWidget* parent = nullptr);

    static NauIcon iconByDeviceName(const QString& name) noexcept;

private:
    void updateParamsWidgets(const QString& conditionName);

    NauComboBox* createSignalInfo(const QString& name, const auto& values);

    void createButtonsInfo(const QString& deviceName, NauComboBox* keyComboBox);
    void setupComboBoxFromPrim(NauComboBox* comboBox, const pxr::TfToken& attrToken);
    void setupSourceKey(const pxr::UsdPrim& signalPrim, NauComboBox* keyBox);

    void setupDigitalBindings();
    void setupAnotherBindings();

private:
    nau::IInputSystem* m_inputSystem;
};


// ** NauInputEditorModifierView

class NauInputEditorModifierView : public NauInputEditorBaseView
{
    Q_OBJECT

public:
    explicit NauInputEditorModifierView(const pxr::UsdPrim& modifierPrim, NauWidget* parent = nullptr);
};


// ** NauInputEditorBindView

class NauInputEditorBindView final : public NauInputEditorSpoiler
{
    Q_OBJECT

public:
    explicit NauInputEditorBindView(const pxr::UsdPrim& bindPrim, NauWidget* parent = nullptr);

    [[nodiscard]]
    pxr::UsdPrim bindPrim() const noexcept;

signals:
    void eventDeleteBindRequested();
    void eventBindUpdateRequested();

private:
    void addSignal(const QString& deviceName);
    void addModifier(const QString& modifierName);

    int signalsCount(const std::string& bindType) const noexcept;
    bool isSignalLimitReached() const;

private:
    void setupSignalActions(NauMenu* menu);
    void setupModifierActions(NauMenu* menu);

private:
    void addView(NauInputEditorBaseView* view, const std::string& arrayToken, std::function<void(NauInputEditorTabWidget*, NauInputEditorBaseView*)> addFunc, std::function<void(NauInputEditorTabWidget*, const NauInputEditorBaseView*)> deleteFunc);

    void addSignalWidget(const pxr::UsdPrim& signalPrim);
    void addModifierWidget(const pxr::UsdPrim& modifierPrim);

    void processRelationship(const pxr::TfToken& relationshipToken, const std::function<void(const pxr::UsdPrim&)>& addWidgetCallback);

    void setupSignalAttributes(pxr::UsdPrim& prim, int signalId, const QString& deviceName);
    void setupModifierAttributes(pxr::UsdPrim& prim, int modifierId, const QString& modifierName);
    void addPrimToRelationship(const pxr::UsdPrim& prim, const std::string& relationshipName);

private:
    pxr::UsdPrim m_bindPrim;

    int m_signalsCount;
    NauInputBindType m_bindType;

    NauInputEditorTabWidgetContainer* m_tabWidget;
};


// ** NauInputEditorBindListView

class NauInputEditorBindListView : public NauInputEditorSpoiler
{
    Q_OBJECT

public:
    explicit NauInputEditorBindListView(NauWidget* parent);

    [[nodiscard]]
    NauInputEditorLineEdit* lineEdit() const noexcept;

signals:
    void eventAddBindRequested(const QString& bindType);

private:
    NauComboBox* m_comboBox;
};


// ** NauInputEditorPage

class NauInputEditorPage : public NauWidget
{
    Q_OBJECT

public:
    explicit NauInputEditorPage(NauWidget* parent);

    void setName(const std::string& actionName);
    void setStage(const pxr::UsdStagePtr& stage);
    // Workaround to process assets
    // TODO Fix it when watcher appears
    void setIsAdded(bool isAdded);

signals:
    void eventProcessAsset(bool isAdded);
    void eventInputUpdateRequired();

private:
    void loadBindsFromStage();

    void removeBindFromStage(const pxr::SdfPath& bindPath);
    void removeRelatedPrims(const pxr::UsdPrim& prim, const pxr::TfToken& relationshipToken);

    void addBindWidget(const pxr::UsdPrim& bindPrim);
    void updateNoBindingsLabel();

    void addBind(const QString& bindType);
    pxr::UsdPrim createBindPrim(const pxr::SdfPath& bindPath, int bindId, const QString& bindType);
    void addBindToRootRelationship(const pxr::UsdPrim& bindPrim);

private:
    pxr::UsdStagePtr m_stage;

    // Workaround to process assets
    bool m_isAdded;

    NauLayoutVertical* m_layout;

    NauInputEditorPageHeader* m_editorHeader;
    NauInputEditorBindListView* m_bindsEditorHeader;

    NauPrimaryButton* m_processAssetButton;

    NauWidget* m_noBindingsLabelContainer;
};
