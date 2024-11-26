// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_assert.hpp"
#include "nau_logger.hpp"

#include "log/nau_log_constants.hpp"
#include "themes/nau_theme.hpp"

#include <QClipboard>
#include <QFileDialog>
#include <QLabel>
#include <QMessageBox>
#include <QPointer>
#include <QPushButton>
#include <QRandomGenerator>
#include <QRegularExpression>
#include <QScrollArea>
#include <QScrollBar>
#include <QStandardPaths>
#include <QString>
#include <QTimer>

#include <DockAreaWidget.h>


// ** NauLoggerOutputPanel

NauLoggerOutputPanel::NauLoggerOutputPanel(NauShortcutHub* shortcutHub, NauWidget* widget)
    : m_logger(new NauLogWidget(shortcutHub, this))
    , m_statusBar(new NauLogStatusPanel(this))
{
    m_logger->setObjectName("loggerBackend");
    auto layout = new NauLayoutVertical(this);
    layout->addWidget(m_logger);

    m_statusBar->setObjectName("logStatusBar");
    m_statusBar->handleMessageCountChanged(m_logger->itemsCount());
    m_statusBar->handleSelectionChanged({});
    layout->addWidget(m_statusBar);

    // Context menu
    setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, &NauLoggerOutputPanel::customContextMenuRequested, this, &NauLoggerOutputPanel::showContextMenu);
    connect(m_logger, &NauLogWidget::eventCurrentMessageChanged, this, &NauLoggerOutputPanel::eventCurrentMessageChanged);
    connect(m_logger, &NauLogWidget::eventOutputCleared, [this]{
        m_statusBar->handleMessageCountChanged({});
        emit eventOutputCleared();
    });

    connect(m_logger, &NauLogWidget::eventSelectedMessagesCopyRequested, this, &NauLoggerOutputPanel::copySelectedMessageToClipboard);
    connect(m_logger, &NauLogWidget::eventMessagesSelectionChanged, this, [this](const QModelIndexList& selection){
        m_statusBar->handleSelectionChanged(selection);
        emit eventMessageSelectionChanged(selection);
    });
    connect(m_logger, &NauLogWidget::eventMessageCountChanged, m_statusBar, &NauLogStatusPanel::handleMessageCountChanged);

    connect(m_statusBar, &NauLogStatusPanel::eventToggleDetailVisibilityRequested, this, &NauLoggerOutputPanel::eventToggleDetailVisibilityRequested);
}

void NauLoggerOutputPanel::registerControlPanel(NauLogToolBar* toolBar)
{
    m_statusBar->setDetailsPanelVisibilityAction(toolBar->detailsPanelVisibilityAction());
    m_logger->setAutoScrollPolicy(toolBar->autoScrollEnabled());

    connect(toolBar, &NauLogToolBar::eventFilterChanged, m_logger, &NauLogWidget::filterData);
    connect(toolBar, &NauLogToolBar::eventAutoScrollPolicyChanged, m_logger, &NauLogWidget::setAutoScrollPolicy);
    connect(toolBar, &NauLogToolBar::eventClearOutputRequested, this, &NauLoggerOutputPanel::clear);
    connect(toolBar, &NauLogToolBar::eventSaveLogAsRequested, this, &NauLoggerOutputPanel::handleSaveLogAsRequest);
}

void NauLoggerOutputPanel::setLogSourceFilter(const QStringList& sourceNames)
{
    m_logger->setLogSourceFilter(sourceNames);
}

void NauLoggerOutputPanel::setLogLevelFilter(const std::vector<NauLogLevel>& preferredLevels)
{
    m_logger->setLogLevelFilter(preferredLevels);
}

void NauLoggerOutputPanel::showContextMenu(const QPoint& position)
{
    QMenu menu(this);
    menu.addAction(tr("Clear output entries"), this, &NauLoggerOutputPanel::clear);
    menu.addSeparator();
    menu.addAction(tr("Copy output entries"), this, &NauLoggerOutputPanel::copyMessagesToClipboard);
    menu.addAction(tr("Copy selected message(s)", nullptr, static_cast<int>(m_logger->selectedItemsCount())), 
        this, &NauLoggerOutputPanel::copySelectedMessageToClipboard);

    menu.exec(this->mapToGlobal(position));
}

void NauLoggerOutputPanel::handleSaveLogAsRequest()
{
    const QString fileName = QFileDialog::getSaveFileName(this, tr("Save log output as..."),
        QStandardPaths::writableLocation(QStandardPaths::DesktopLocation), tr("Text files (*.txt)"));

    if (fileName.isEmpty()) {
        NED_TRACE("Saving log output declined");
        return;
    }

    QFile outputFile{fileName};
    if (!outputFile.open(QFile::WriteOnly | QFile::Truncate)) {
        NED_ERROR("Failed to open file to save the log output '{}'", fileName.toUtf8().constData());
        return;
    }

    QTextStream outStream(&outputFile);
    m_logger->writeData(outStream);
    QMessageBox::information(nullptr, qApp->applicationDisplayName(), tr("Log output successfully saved!"));
    NED_DEBUG("Log output successfully saved to '{}'", fileName.toUtf8().constData());
}

void NauLoggerOutputPanel::clear()
{
    NED_DEBUG("Clear output entries");
    m_logger->clear();
}

void NauLoggerOutputPanel::copyMessagesToClipboard() const
{
    NED_DEBUG("Copy output entries to clipboard");
    QString result;
    QTextStream logs(&result);
    for (int i = 0; i < m_logger->itemsCount(); ++i) {
        logs << m_logger->messageAt(i) << Qt::endl;
    }
    auto clipboard = QGuiApplication::clipboard();
    clipboard->setText(result);
}

void NauLoggerOutputPanel::copySelectedMessageToClipboard() const
{
    NED_DEBUG("Copy message to clipboard");
    auto clipboard = QGuiApplication::clipboard();
    clipboard->setText(m_logger->selectedMessages());
}

bool NauLoggerOutputPanel::detailsPanelVisible() const
{
    return m_statusBar->detailsPanelVisible();
}

std::size_t NauLoggerOutputPanel::selectedItemsCount() const
{
    return m_logger->selectedItemsCount();
}


// ** NauLoggerSourceTreeContainer

NauLoggerSourceTreeContainer::NauLoggerSourceTreeContainer(QWidget* parent)
    : NauFrame(parent)
{
    sourceModel = new NauLogSourceModel;
    m_sourceTree = new NauTreeView;
    m_sourceTree->setObjectName("loggerSourceTree");
    m_sourceTree->setModel(sourceModel);
    m_sourceTree->setRootIsDecorated(false);
    m_sourceTree->setHeaderHidden(true);
    m_sourceTree->setSelectionMode(QAbstractItemView::SelectionMode::SingleSelection);

    auto delegate = new NauLogTreeViewItemDelegate();
    delegate->setPalette(Nau::Theme::current().paletteLogger());
    delegate->setColumnHighlighted(+NauLogSourceModel::Column::SourceName);
    delegate->setRowHeight(NauLogConstants::sourceItemHeight());

    m_sourceTree->setItemDelegate(delegate);

    auto bottomPanel = new NauFrame(this);
    bottomPanel->setFixedHeight(NauLogConstants::bottomInfoPanelHeight());

    auto layout = new NauLayoutVertical(this);
    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_sourceTree);
    layout->addWidget(bottomPanel);

    connect(m_sourceTree->selectionModel(), &QItemSelectionModel::currentChanged, [this] {
        const QModelIndex index = m_sourceTree->currentIndex();
        const QStringList sourceNames = index.data(NauLogSourceModel::SourceNamesRole).toStringList();

        emit eventSourceToggleRequested(sourceNames);
    });
    connect(m_sourceTree->selectionModel(), &QItemSelectionModel::selectionChanged, [this](const QItemSelection& selected) {
        if (selected.isEmpty()) {
            emit eventSourceToggleRequested({});
        }
    });
}


// ** NauLoggerWidget

NauLoggerWidget::NauLoggerWidget(NauShortcutHub* shortcutHub, NauWidget* parent)
    : NauWidget(parent)
    , m_sourceTreeContainer(new NauLoggerSourceTreeContainer(this))
    , m_controlPanel(new NauLogToolBar(this))
    , m_loggerPanel(new NauLoggerOutputPanel(shortcutHub, this))
    , m_detailsPanel(new NauLoggerDetailsPanel(shortcutHub, this))
{
    setFont(Nau::Theme::current().fontLogger());

    auto layout = new NauLayoutVertical(this);
    auto splitter = new NauSplitter();
    splitter->setOrientation(Qt::Horizontal);

    m_loggerPanel->registerControlPanel(m_controlPanel);
    
    splitter->addWidget(m_sourceTreeContainer);
    splitter->addWidget(m_loggerPanel);
    splitter->addWidget(m_detailsPanel);
    splitter->setSizes({ 200, 640, 280 });
    splitter->setCollapsible(0, false);
    splitter->setCollapsible(1, false);
    splitter->setCollapsible(2, false);

    // Put together
    layout->addWidget(m_controlPanel);
    layout->addWidget(new NauSpacer);
    layout->addWidget(splitter);
    layout->setStretch(2, 1);
    layout->setSpacing(0);

    connect(m_loggerPanel, &NauLoggerOutputPanel::eventCurrentMessageChanged, [this](const QModelIndex& current)
    {
        m_detailsPanel->loadMessageInfo(current);
        m_detailsPanel->setVisible(m_loggerPanel->detailsPanelVisible() && m_loggerPanel->selectedItemsCount() == 1);
    });
    connect(m_loggerPanel, &NauLoggerOutputPanel::eventMessageSelectionChanged, [this](const QModelIndexList& selection)
    {
        // Showing the details panel, if user selects several items may lead to user's confusion.
        m_detailsPanel->setVisible(m_loggerPanel->detailsPanelVisible() && selection.count() == 1);
    });
    connect(m_controlPanel, &NauLogToolBar::eventFilterByLevelChangeRequested, 
        m_loggerPanel, &NauLoggerOutputPanel::setLogLevelFilter);

    connect(m_detailsPanel, &NauLoggerDetailsPanel::eventCloseRequested, m_detailsPanel, &QWidget::hide);
    connect(m_sourceTreeContainer, &NauLoggerSourceTreeContainer::eventSourceToggleRequested,
        m_loggerPanel, &NauLoggerOutputPanel::setLogSourceFilter);

    connect(m_loggerPanel, &NauLoggerOutputPanel::eventToggleDetailVisibilityRequested, [this](bool visible) {
        m_detailsPanel->setVisible(visible && m_loggerPanel->selectedItemsCount() == 1);
    });
    connect(m_loggerPanel, &NauLoggerOutputPanel::eventOutputCleared, m_detailsPanel, &NauLoggerDetailsPanel::hide);

    m_detailsPanel->setVisible(false);
}


// ** NauTabbedLoggerWidget

NauTabbedLoggerWidget::NauTabbedLoggerWidget(NauShortcutHub* shortcutHub, NauDockManager* manager)
    : NauDockWidget(tr("Console", "The Console tab of log panel"), nullptr)
    , m_shortcutHub(shortcutHub)
    , m_manager(manager)
{
    // Default tab
    auto loggerDefault = new NauLoggerWidget(m_shortcutHub);
    setWidget(loggerDefault, NauDockWidget::ForceNoScrollArea);
    setObjectName("DockConsole");
    setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromContent);
    m_tabs[NauLogConstants::consoleTabName()] = this;

    auto nauDockTab = dynamic_cast<NauDockWidgetTab*>(tabWidget());
    if (!nauDockTab) {
        NED_CRITICAL("Logger header expected to be an instance of NauDockWidgetTab");
        return;
    }

    connect(&NauLog::editorLogModel(), &NauLogModel::eventCriticalEventOccurred, [this, nauDockTab] {
        nauDockTab->setFlashingEnabled(!nauDockTab->isActiveTab());
        nauDockTab->setBadgeTextVisible(!nauDockTab->isActiveTab());
    });
}

NauLoggerWidget* NauTabbedLoggerWidget::addTab(const QString& name)
{
    auto loggerPanel = new NauLoggerWidget(m_shortcutHub);
    auto dockWidget = new NauDockWidget(QString(name.data()), nullptr);
    dockWidget->setWidget(loggerPanel);
    dockWidget->setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromContent);
    m_manager->addDockWidgetTabToArea(dockWidget, dockAreaWidget());
    m_tabs[name] = dockWidget;
    return loggerPanel;
}

NauDockWidget* NauTabbedLoggerWidget::getTab(const QString& name)
{
    NED_ASSERT(m_tabs.find(name) != m_tabs.end());
    return m_tabs[name];
}

void NauTabbedLoggerWidget::switchTab(const QString& name)
{
    // TODO: error checking
    NED_ASSERT(m_tabs.find(name) != m_tabs.end());
    dockAreaWidget()->setCurrentDockWidget(m_tabs[name]);
}
