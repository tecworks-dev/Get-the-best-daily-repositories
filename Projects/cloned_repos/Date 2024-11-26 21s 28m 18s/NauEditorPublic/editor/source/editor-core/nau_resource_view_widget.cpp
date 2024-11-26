// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_resource_view_widget.hpp"
#include "nau/outliner/nau_world_outline_panel.hpp"
#include "nau_plus_enum.hpp"
#include "nau_log.hpp"
#include "nau_tree_view_item_delegate.hpp"
#include "themes/nau_theme.hpp"
#include "nau/assets/nau_asset_preview.hpp"

#include <QPainter>
#include <QPainterPath>

#include <ctime>
#include <format>


#pragma region ABSTRACT LEVEL

// ** NauPopupAbstractWidgetHeader

NauPopupAbstractWidgetHeader::NauPopupAbstractWidgetHeader(NauWidget* parent)
    : NauWidget(parent)
    , m_layout(new NauLayoutHorizontal(this))
    , m_additinalButtonsLayout(new NauLayoutHorizontal())
    , m_search(new NauSearchWidget(this))
{
    // TODO: In the future, make the length of the widget depend on the length of its ComboBox widget.
    setFixedSize(HeaderWidht, HeaderHeight);
    setContentsMargins(OuterMargin, OuterMargin, OuterMargin, OuterMargin);

    m_additinalButtonsLayout->setContentsMargins(OuterMargin, InnerMargin, 0, InnerMargin);
    m_additinalButtonsLayout->setSpacing(Spacing);

    m_layout->addWidget(m_search);
    m_layout->addLayout(m_additinalButtonsLayout);

    connect(m_search, &QLineEdit::textChanged, this, &NauPopupAbstractWidgetHeader::eventSearchFilterChanged);
}

void NauPopupAbstractWidgetHeader::setPlaceholderText(const QString& placeholderText)
{
    m_search->setPlaceholderText(placeholderText);
}

QString NauPopupAbstractWidgetHeader::placeholderText() const
{
    return m_search->placeholderText();
}


// ** NauPopupAbstractTreeWidget

NauPopupAbstractTreeWidget::NauPopupAbstractTreeWidget(NauWidget* parent)
    : NauTreeWidget(parent)
{
    // TODO: In the future, make the length of the widget depend on the length of its ComboBox widget.
}

void NauPopupAbstractTreeWidget::clearData()
{
    m_items.clear();
}

size_t NauPopupAbstractTreeWidget::count() const
{
    return m_items.size();
}

QString NauPopupAbstractTreeWidget::currentData(QTreeWidgetItem* index) const
{
    return m_items.find(index)->second;
}


// ** NauPopupAbstractWidgetFooter

NauPopupAbstractWidgetFooter::NauPopupAbstractWidgetFooter(NauWidget* parent)
    : NauWidget(parent)
    , m_layout(new NauLayoutHorizontal(this))
    , m_text(new NauStaticTextLabel(""))
{
    // TODO: In the future, make the length of the widget depend on the length of its ComboBox widget.
    setFixedSize(HeaderWidht, HeaderHeight);
    setContentsMargins(OuterMarginWidth, OuterMarginHeight, OuterMarginWidth, OuterMarginHeight);

    m_layout->addWidget(m_text);
}

void NauPopupAbstractWidgetFooter::setText(const QString& text)
{
    m_text->setText(text);
    m_text->update();
}


// ** NauAbstractPopupWidget

NauAbstractPopupWidget::NauAbstractPopupWidget(NauPopupAbstractWidgetHeader* header, NauPopupAbstractTreeWidget* container, NauPopupAbstractWidgetFooter* footer, NauWidget* parent)
    :NauWidget(parent)
    , m_layout(new NauLayoutVertical(this))
    , m_header(header)
    , m_container(container)
    , m_footer(footer)
{
    // TODO: In the future, make the length of the widget depend on the length of its ComboBox widget.
    setFixedSize(HeaderWidht, HeaderHeight);
    setWindowFlags(Qt::Popup);

    m_layout->addWidget(m_header);
    m_layout->addWidget(m_container);
    m_layout->addWidget(m_footer);

    connect(m_header, &NauPopupAbstractWidgetHeader::eventSearchFilterChanged, this, &NauAbstractPopupWidget::updateFilterData);
}


// ** NauAbstractComboBox

NauAbstractComboBox::NauAbstractComboBox(NauWidget* parent)
    : NauPrimaryButton(parent)
    , m_container(nullptr)
{
    const auto& theme = Nau::Theme::current();
    m_styleMap = theme.styleResourceWidget().styleByState;

    setContentsMargins(OuterMarginWidth, OuterMarginHeight,
        OuterMarginWidth + ButtonSize.width() + Gap, OuterMarginHeight);

    textAlign = Qt::AlignLeft;
    iconAlign = Qt::AlignLeft;
}

size_t NauAbstractComboBox::count() const
{
    return m_container->container().count();
}

void NauAbstractComboBox::setPlaceholderText(const QString& placeholderText)
{
    m_container->header().setPlaceholderText(placeholderText);
}

QString NauAbstractComboBox::placeholderText() const
{
    return m_container->header().placeholderText();
}

QString NauAbstractComboBox::currentData(QTreeWidgetItem* item) const
{
    return m_container->container().currentData(item);
}

void NauAbstractComboBox::addItem(const QString& item, nau::Uid uid, const std::string& primPath, NauEditorFileType type)
{
   QTreeWidgetItem* treeItem = m_container->container().addItem(item, type);
   m_dataUid[treeItem] = uid;
   m_dataPrimPath[treeItem] = primPath;
}

void NauAbstractComboBox::addItems(const std::vector<QString>& items, NauEditorFileType type)
{
    m_container->container().addItems(items, type);
}

void NauAbstractComboBox::addClearSelectionButton()
{
    if (m_cleanButton) {
        NED_WARNING("Clear button already added to combobox {}", objectName());
        return;
    }

    m_cleanButton = new NauMiscButton(this);
    m_cleanButton->setIcon(Nau::Theme::current().iconClean());
    m_cleanButton->setFixedSize(ButtonSize);
    setContentsMargins(contentsMargins() + QMargins(0, 0, ButtonSize.width() + Gap, 0));

    connect(m_cleanButton, &NauAbstractButton::clicked, this, &NauAbstractComboBox::clearSelection);
}

void NauAbstractComboBox::clear()
{
    m_container->container().clear();
    m_container->container().clearData();

    m_dataUid.clear();
    m_dataPrimPath.clear();
}

void NauAbstractComboBox::clearSelection()
{
    if (count() == 0 && !text().isEmpty()) {
        setText({});
        emit eventSelectionChanged(QString());
    } else {        
        setText({});
        m_container->container().selectionModel()->clear();
    }
}

void NauAbstractComboBox::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    NauPrimaryButton::paintEvent(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);

    const auto& theme = Nau::Theme::current();
    auto icon = theme.iconThreeDotsHorisont();

    QRect iconRect{QPoint(0, 0), ButtonSize};
    iconRect.moveCenter(event->rect().center());
    iconRect.moveRight(event->rect().right() - OuterMarginWidth);

    if (m_cleanButton) {
        iconRect.moveRight(iconRect.right() - ButtonSize.width() - Gap);
    }

    icon.paint(&painter, iconRect, Qt::AlignVCenter | Qt::AlignLeft,
        m_styleMap[m_currentState].iconState, isChecked() ? NauIcon::On : NauIcon::Off);
}

void NauAbstractComboBox::resizeEvent(QResizeEvent* event)
{
    if (m_cleanButton) {
        QRect buttonRect = m_cleanButton->geometry();
        buttonRect.moveCenter(rect().center());
        buttonRect.moveRight(rect().right() - OuterMarginWidth);

        m_cleanButton->setGeometry(buttonRect);
    }
}

void NauAbstractComboBox::showPopup()
{
    // Initially the code was taken from Qt sources and transferred for our needs,
    // because the standard ComboBox had limited possibilities for customization.

    // The original source code can be found here:
    // https://github.com/qt/qtbase/blob/5b151ea2d23dc3834180d3ec6495ac5d99cae550/src/widgets/widgets/qcombobox.cpp#L2621

    if (count() <= 0) {
        return;
    }

    QRect popupRect = rect();

    // Since the methods of getting the available geometry via QApplication are already obsolete,
    // the most reliable way to do it is via the ComboBox widget, since it is always on the right screen.
    QRect availableScreenGeometry = screen()->availableGeometry();

    QPoint belowPoint = mapToGlobal(popupRect.bottomLeft());
    int belowHeight = availableScreenGeometry.bottom() - belowPoint.y();

    QPoint abovePoint = mapToGlobal(popupRect.topLeft());
    int aboveHeight = abovePoint.y() - availableScreenGeometry.y();

    bool boundToScreen = !window()->testAttribute(Qt::WA_DontShowOnScreen);

    // We need to activate the layout to make sure the min/maximum size are set when the widget was not yet show
    m_container->layout()->activate();

    // Takes account of the minimum/maximum size of the container
    popupRect.setSize(popupRect.size().expandedTo(m_container->minimumSize()).boundedTo(m_container->maximumSize()));

    // Make sure the widget fits on screen
    if (boundToScreen) {

        if (popupRect.width() > availableScreenGeometry.width()) {
            popupRect.setWidth(availableScreenGeometry.width());
        }

        if (mapToGlobal(popupRect.bottomRight()).x() > availableScreenGeometry.right()) {
            belowPoint.setX(availableScreenGeometry.x() + availableScreenGeometry.width() - popupRect.width());
            abovePoint.setX(availableScreenGeometry.x() + availableScreenGeometry.width() - popupRect.width());
        }

        if (mapToGlobal(popupRect.topLeft()).x() < availableScreenGeometry.x()) {
            belowPoint.setX(availableScreenGeometry.x());
            abovePoint.setX(availableScreenGeometry.x());
        }
    }

    if (!boundToScreen || popupRect.height() <= belowHeight) {
        popupRect.moveTopLeft(belowPoint);

    } else if (popupRect.height() <= aboveHeight) {
        popupRect.moveBottomLeft(abovePoint);

    } else if (belowHeight >= aboveHeight) {
        popupRect.setHeight(belowHeight);
        popupRect.moveTopLeft(belowPoint);

    } else {
        popupRect.setHeight(aboveHeight);
        popupRect.moveBottomLeft(abovePoint);
    }

    if (qApp) {
        QGuiApplication::inputMethod()->reset();
    }

    m_container->setGeometry(popupRect);

    // Don't disable updates on OS X. Windows are displayed immediately on this platform,
    // which means that the window will be visible before the call to container->show() returns.
    // If updates are disabled at this point we'll miss our chance at painting the popup
    // menu before it's shown, causing flicker since the window then displays the standard gray
    // background.

#ifndef Q_OS_MAC
    const bool updatesEnabled = m_container->updatesEnabled();
    m_container->setUpdatesEnabled(false);
#endif

    m_container->raise();
    m_container->show();

    // Slide the widget down a bit to open an overview of the widget's header
    const QPoint buttonAbsoluteCoords = m_container->mapToGlobal(QPoint(0, 0));
    const int newXCoord = buttonAbsoluteCoords.x();
    const int newYCoord = buttonAbsoluteCoords.y() + this->size().height();

    m_container->move(newXCoord, newYCoord);

    m_container->footer().setText(std::format("{} items", m_container->container().count()).c_str());

    m_container->setFocus();

#ifndef Q_OS_MAC
    m_container->setUpdatesEnabled(updatesEnabled);
#endif

    m_container->update();
}

void NauAbstractComboBox::onPressed()
{
    showPopup();
    NauPrimaryButton::onPressed();
}
#pragma endregion

#pragma region COMPONENT LEVEL

// ** NauPopupResourceHeader

NauPopupResourceHeader::NauPopupResourceHeader(NauWidget* parent)
    : NauPopupAbstractWidgetHeader(parent)
{
    const auto& theme = Nau::Theme::current();
    const auto& palette = theme.paletteWorldOutline();

    auto* folderButton = new NauToolButton();
    auto* settingsButton = new NauToolButton();

    folderButton->setIcon(theme.iconButtonFolder());
    settingsButton->setIcon(theme.iconButtonSettings());

    m_additinalButtonsLayout->addWidget(folderButton);
    m_additinalButtonsLayout->addWidget(settingsButton);
}

void NauPopupResourceHeader::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    const auto& theme = Nau::Theme::current();
    const auto& palette = theme.paletteResourcePopupWidget();

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.setRenderHint(QPainter::TextAntialiasing);

    painter.setPen(NauPen());

    QPainterPath path;
    path.addRect(rect());
    painter.fillPath(path, palette.color(NauPalette::Role::BackgroundHeader));

    NauWidget::paintEvent(event);
}

// ** NauResourceTreeWidget

NauResourceTreeWidget::NauResourceTreeWidget(NauWidget* parent)
    : NauPopupAbstractTreeWidget(parent)
{
    setObjectName("NauResourceTreeWidget");

    const auto& theme = Nau::Theme::current();
    const auto& palette = theme.paletteWorldOutline();

    auto m_header = new NauWorldOutlineHeaderView(Qt::Orientation::Horizontal);
    m_header->setObjectName("NauResourceTreeWidgetHeader");
    m_header->setIconSize(QSize(HeaderIconSize, HeaderIconSize));
    m_header->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    m_header->setFixedHeight(HeaderHeight);

    setHeader(m_header);
    setColumnCount(magic_enum::enum_count<Columns>());
    
    setSortingEnabled(true);
    setSelectionMode(SingleSelection);
    sortByColumn(+Columns::Name, Qt::AscendingOrder);

    // We pass palette colors from our theme to Qt to make it draw widgets according the current theme.
    // But not all features from theme are supported by QPalette. We had to implement them via qss and coding.
    // This translation looks like redundant and hack and should be removed as soon as outline has own paint routines.
    QPalette qtPalette;
    qtPalette.setBrush(QPalette::Base, palette.brush(NauPalette::Role::Background));
    qtPalette.setBrush(QPalette::AlternateBase, palette.brush(NauPalette::Role::AlternateBackground));
    qtPalette.setColor(QPalette::Text, palette.color(NauPalette::Role::Foreground));
    setPalette(qtPalette);

    // Setup columns
    setupColumn(+Columns::Preview, tr("Preview"), theme.fontWorldOutlineHeaderRegular(), true, PreviewColumnWight, QHeaderView::ResizeMode::Fixed);
    setupColumn(+Columns::Name, tr("Name"), theme.fontWorldOutlineHeaderSemibold(), true, NameColumnWight, QHeaderView::ResizeMode::Stretch);

    setIconSize(QSize(ContentIconSize, ContentIconSize));

    auto delegate = new NauTreeViewItemDelegate;
    delegate->setPalette(palette);
    delegate->setColumnHighlighted(+Columns::Name);
    delegate->setRowHeight(RowHeight);
    delegate->setRowContentsMargins(RowContentsMarginsWight, RowContentsMarginsHight, RowContentsMarginsWight, RowContentsMarginsHight);
    delegate->setCellContentsMargins(CellContentsMargins, 0, CellContentsMargins, 0);
    delegate->setSpacing(Spacing);

    setItemDelegate(delegate);

    // TODO: Temporary solution, cope when the generated resource icons appear
    srand(time(0));

    m_iconsRepository = {
    {NauEditorFileType::Unrecognized, QIcon(":/UI/icons/browser/unknown.png")},
    {NauEditorFileType::EngineCore, QIcon(":/UI/icons/browser/engine.png")},
    {NauEditorFileType::Project, QIcon(":/UI/icons/browser/editor.png")},
    {NauEditorFileType::Config, QIcon(":/UI/icons/browser/config.png")},
    {NauEditorFileType::Texture, QIcon(":/UI/icons/browser/texture.png")},
    {NauEditorFileType::Model, QIcon(":/UI/icons/browser/model.png")},
    {NauEditorFileType::Shader, QIcon(":/UI/icons/browser/shader.png")},
    {NauEditorFileType::Script, QIcon(":/UI/icons/browser/script.png")},
    {NauEditorFileType::VirtualRomFS, QIcon(":/UI/icons/browser/vromfs.png")},
    {NauEditorFileType::Scene, QIcon(":/UI/icons/browser/scene.png")},
    {NauEditorFileType::Material, QIcon(":/UI/icons/browser/material.png")},
    {NauEditorFileType::Action, QIcon(":/UI/icons/browser/action.png")},
    {NauEditorFileType::AudioContainer, QIcon(":/UI/icons/browser/temp-audio-cointainer.png")},
    {NauEditorFileType::RawAudio, QIcon(":/UI/icons/browser/temp-wave.png")},
    {NauEditorFileType::UI, QIcon(":/UI/icons/iAllTemplateTab.png")},
    {NauEditorFileType::Font, QIcon(":/UI/icons/iAllTemplateTab.png")},
    {NauEditorFileType::VFX, QIcon(":/UI/icons/browser/vfx.png")},
    {NauEditorFileType::PhysicsMaterial, QIcon(":/UI/icons/browser/material.png")},
    };
}

QTreeWidgetItem* NauResourceTreeWidget::addItem(const QString& item, NauEditorFileType type)
{
    auto treeItem = new QTreeWidgetItem;

    int start = 0;
    int end = 2;
    int randomInRange = rand() % (end - start + 1) + start;
    
    QIcon assetIcon = m_iconsRepository[type];

    // TODO: Temporary solution, cope when the generated resource icons appear
    treeItem->setIcon(+Columns::Preview, assetIcon);
    treeItem->setText(+Columns::Name, QFileInfo(item).completeBaseName());
    treeItem->setData(+Columns::Name, Qt::UserRole, QString(item));

    NauPopupAbstractTreeWidget::addTopLevelItem(treeItem);

    m_items[treeItem] = item;

    return treeItem;
}

void NauResourceTreeWidget::addItems(const std::vector<QString>& items, NauEditorFileType type)
{
    for (auto treeItem : items) {
        addItem(treeItem, type);
    }
}

void NauResourceTreeWidget::insertItem(int index, const QString& item, NauEditorFileType type)
{
    auto treeItem = new QTreeWidgetItem;

    srand(time(0));
    int start = 0;
    int end = 2;
    int randomInRange = rand() % (end - start + 1) + start;

    // TODO: Temporary solution, cope when the generated resource icons appear
    treeItem->setIcon(+Columns::Preview, m_iconsRepository[type]);
    treeItem->setText(+Columns::Name, item);

    NauPopupAbstractTreeWidget::insertTopLevelItem(index, treeItem);

    m_items[treeItem] = item;
}

void NauResourceTreeWidget::insertItems(int index, const std::vector<QString>& items, NauEditorFileType type)
{
    for (auto treeItem : items) {
        insertItem(index, treeItem, type);
    }
}


// ** NauPopupResorceFooter

NauPopupResorceFooter::NauPopupResorceFooter(NauWidget* parent)
    : NauPopupAbstractWidgetFooter(parent)
{
}

void NauPopupResorceFooter::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    const auto& theme = Nau::Theme::current();
    const auto& palette = theme.paletteResourcePopupWidget();

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.setRenderHint(QPainter::TextAntialiasing);

    painter.setPen(NauPen());

    QPainterPath path;
    path.addRect(rect());

    painter.fillPath(path, palette.color(NauPalette::Role::BackgroundFooter));

    NauWidget::paintEvent(event);
}


// **NauResourcePopupWidget

NauResourcePopupWidget::NauResourcePopupWidget(NauWidget* parent)
    : NauAbstractPopupWidget(new NauPopupResourceHeader(), new NauResourceTreeWidget(), new NauPopupResorceFooter(), parent)
{
    m_header->setPlaceholderText(tr("Search in resources..."));
}

void NauResourcePopupWidget::updateFilterData(const QString& filter)
{
    // TODO: In the future to do it through: QSortFilterProxyModel
    auto itemsList = m_container->findItems("", Qt::MatchContains, +NauResourceTreeWidget::Columns::Name);
    auto count = itemsList.count();

    for (QTreeWidgetItem* item : itemsList) {
        if (!item->text(+NauResourceTreeWidget::Columns::Name).contains(filter)) {
            item->setHidden(true);
            count--;
        } else {
            item->setHidden(false);
        }
    }

    footer().setText(std::format("{} items", count).c_str());
}

void NauResourcePopupWidget::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    const auto& theme = Nau::Theme::current();
    const auto& palette = theme.paletteResourcePopupWidget();

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.setRenderHint(QPainter::TextAntialiasing);

    painter.setPen(NauPen());

    QPainterPath path;
    path.addRect(rect());

    painter.fillPath(path, palette.color(NauPalette::Role::Background));

    NauWidget::paintEvent(event);
}

#pragma endregion

#pragma region WIDGET LEVEL

// ** NauResourceComboBox

NauResourceComboBox::NauResourceComboBox(NauWidget* parent)
    : NauAbstractComboBox(parent)
{
    m_container = new NauResourcePopupWidget();

    connect(m_container->container().selectionModel(), &QItemSelectionModel::currentChanged,
    [this, widget = &m_container->container()](const QModelIndex& current) {
        if (current.isValid()) {
            const QString itemText = current.data( Qt::UserRole).toString();
            setText(itemText);

            if (auto item = widget->itemFromIndex(current)) {
                m_currentContainerUid = m_dataUid[item];
                m_currentPrimPath = m_dataPrimPath[item];

                emit eventSelectionChanged();
                emit eventSelectionChanged(itemText);
            }
        }
    });
}

void NauResourceComboBox::clear()
{
    NauAbstractComboBox::clear();
    m_currentContainerUid = nau::Uid();
    m_currentPrimPath = "";
}

#pragma endregion
