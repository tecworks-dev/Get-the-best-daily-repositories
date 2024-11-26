// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_about.hpp"
#include "nau_feedback.hpp"
#include "nau_assert.hpp"
#include "nau_dock_widget.hpp"
#include "nau_editor_window.hpp"
#include "nau_log.hpp"
#include "nau_main_menu.hpp"
#include "nau/scene/nau_scene_manager.hpp"
#include "nau_settings.hpp"
#include "log/nau_log_constants.hpp"

#ifdef NAU_BUILD_WATERMARK
#include "viewport/nau_viewport_utils.hpp"
#endif // NAU_BUILD_WATERMARK

#include <QDesktopServices>
#include <QMessageBox>
#include <QtWidgets/QApplication>
#include <QSplitter>
#include <QSignalBlocker>
#include <QWindow>
#include <QFileDialog>

#include <QJSonArray>
#include <QJSonDocument>

#ifdef Q_OS_WIN
#include <Windows.h>
#include <windowsx.h>
#endif


// ** NauEditorWindow

NauEditorWindow::NauEditorWindow()
    : NauEditorWindowAbstract()
    , m_shortcutHub(new NauShortcutHub(*this))
    , m_projectBrowser(new NauProjectBrowser(m_shortcutHub, nullptr))
    , m_dockManager(nullptr)
    , m_viewport(new NauViewportContainerWidget(nullptr))
    , m_titleBar(new NauTitleBar(m_shortcutHub, this))
    , m_toolbar(new NauMainToolbar(this))
    , m_entityInspector(new NauEntityInspectorPage(nullptr))
    , m_inspector(new NauInspectorPage(nullptr))
    , m_sceneSettingsPanel(new NauSceneSettingsPanel(nullptr))
    , m_worldOutline(new NauWorldOutlinerWidget(m_shortcutHub, nullptr))
{
    const QSize minSize{ 1280, 720 };
    setMinimumSize(minSize);

    setWindowState(windowState() | Qt::WindowMaximized);
    setObjectName("NauEditor");

    // Setup title bar
    setAttribute(Qt::WA_NativeWindow);   // Need this to hide default title bar, bar keep its features like aero-snapping
    setMenuWidget(m_titleBar);

    // Title bar events
    connect(m_titleBar, &NauTitleBar::eventToggleWindowStateRequested, [this](bool maximized) {
        if (!maximized) {
            showNormal();
        } else {
            showMaximized();
        }

        m_titleBar->setMaximized(maximized);
    });

    connect(m_titleBar, &NauTitleBar::eventMinimize, [this] {
        showMinimized();
    });

    connect(m_titleBar, &NauTitleBar::eventClose, [this] {
        close();
    });
    connect(m_titleBar, &NauTitleBar::eventMoveWindowRequested, [this] {
        windowHandle()->startSystemMove();
    });
    connect(m_titleBar, &NauTitleBar::eventResizeWindowRequested, [this] {
        windowHandle()->startSystemResize(Qt::TopEdge);
    });

    connect(windowHandle(), &QWindow::windowStateChanged, [this](Qt::WindowState windowState) {
        m_titleBar->setMaximized(windowState & Qt::WindowMaximized);
    });

    // Note that these flags must be set before NauDockManager instance is created.
    // Explicitly disable opaque mode to spare the viewport from heavy redrawing.
    NauDockManager::setConfigFlag(NauDockManager::OpaqueSplitterResize, false);
    NauDockManager::setConfigFlag(NauDockManager::DockAreaHasTabsMenuButton, false);
    NauDockManager::setConfigFlag(NauDockManager::DockAreaHasUndockButton, false);
    NauDockManager::setConfigFlag(NauDockManager::DockAreaHasCloseButton, false);
    NauDockManager::setConfigFlag(NauDockManager::ActiveTabHasCloseButton, true);
    NauDockManager::setConfigFlag(NauDockManager::TabCloseButtonIsToolButton, true);
    NauDockManager::setConfigFlag(NauDockManager::AllTabsHaveCloseButton, true);
    NauDockManager::setAutoHideConfigFlags(NauDockManager::DefaultAutoHideConfig);

    ads::CDockComponentsFactory::setFactory(new NauAdsComponentsFactory);

    // Docking widget and the main toolbar
    auto widgetMain = new QWidget(this);
    auto layoutMain = new NauLayoutVertical(widgetMain);
    layoutMain->setContentsMargins(0, 0, 0, 0);
    layoutMain->setSpacing(0);
    setCentralWidget(widgetMain);
    m_dockManager = new NauDockManager(widgetMain);

    auto separatorTop = new NauLineWidget(QColor("#141414"), 4, Qt::Orientation::Horizontal, this);
    separatorTop->setFixedHeight(4);
    layoutMain->addWidget(separatorTop);

    layoutMain->addWidget(m_toolbar);

    auto separatorBottom = new NauLineWidget(QColor("#141414"), 4, Qt::Orientation::Horizontal, this);
    separatorBottom->setFixedHeight(4);
    layoutMain->addWidget(separatorBottom);

    layoutMain->addWidget(m_dockManager);

    // Logger
    m_logger = new NauTabbedLoggerWidget(m_shortcutHub, m_dockManager);

#ifdef NAU_BUILD_WATERMARK
    NauViewportWatermark::setDevicePixelRatio(QWidget::devicePixelRatioF());
#endif // NAU_BUILD_WATERMARK

    // Window docking setup
    auto dwSceneViewPort = new NauDockWidget(tr("Viewport"), nullptr);
    dwSceneViewPort->setWidget(m_viewport);
    dwSceneViewPort->setObjectName("DockViewport");
    dwSceneViewPort->setFeature(ads::CDockWidget::DockWidgetAloneHasNoTitleBar, true);
    auto centralArea = m_dockManager->addDockWidget(ads::CenterDockWidgetArea, dwSceneViewPort);
   
    auto dwInspector = new NauDockWidget(tr("Inspector"), nullptr);
    dwInspector->setWidget(m_inspector);
    dwInspector->setObjectName("DockInspector");
    dwInspector->setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromContentMinimumSize);
    m_dockManager->addDockWidget(ads::RightDockWidgetArea, dwInspector);
    m_dockManager->setInspector(dwInspector);

    auto dwSceneHierarchy = new NauDockWidget(tr("World Outliner"), nullptr);
    dwSceneHierarchy->setWidget(m_worldOutline);
    dwSceneHierarchy->setObjectName("DockOutliner");
    dwSceneHierarchy->setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromContent);
    m_dockManager->addDockWidget(ads::LeftDockWidgetArea, dwSceneHierarchy);

    auto dwSceneSettingsPanel = new NauDockWidget(tr("Scene Settings"), nullptr);
    dwSceneSettingsPanel->setWidget(m_sceneSettingsPanel);
    dwSceneSettingsPanel->setObjectName("DockSettings");
    dwSceneSettingsPanel->setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromContent);
    m_dockManager->addDockWidgetTabToArea(dwSceneSettingsPanel, dwInspector->dockAreaWidget())->setCurrentDockWidget(dwInspector);
 
    auto dwProjectBrowser = new NauDockWidget(tr("Project Browser"), nullptr);
    dwProjectBrowser->setWidget(m_projectBrowser, NauDockWidget::ForceNoScrollArea);
    dwProjectBrowser->setObjectName("DockProjectBrowser");
    dwProjectBrowser->setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromContent);
    m_dockManager->setProjectBrowser(dwProjectBrowser);

    auto bottomArea = m_dockManager->addDockWidget(ads::BottomDockWidgetArea, dwProjectBrowser, centralArea);

    // Logger
    m_logger->setStyleSheet("background-color: transparent");
    auto centralDownArea = m_dockManager->addDockWidgetTabToArea(m_logger, dwProjectBrowser->dockAreaWidget());
    centralDownArea->setTabsPosition(ads::CDockAreaWidget::South);
    centralDownArea->setCurrentDockWidget(dwProjectBrowser);

    NauMainMenu* mainMenu = m_titleBar->menu().get();
    NauMenu* menu = mainMenu->getMenu(NauMainMenu::MenuItem::View);
    NED_ASSERT(menu);

    menu->addAction(dwSceneViewPort->toggleViewAction());
    menu->addAction(dwInspector->toggleViewAction());
    menu->addAction(dwSceneSettingsPanel->toggleViewAction());
    menu->addAction(dwProjectBrowser->toggleViewAction());
    menu->addAction(dwSceneHierarchy->toggleViewAction());
    menu->addAction(m_logger->getTab(NauLogConstants::consoleTabName())->toggleViewAction());

    // About dialog
    connect(mainMenu, &NauMainMenu::aboutDialogRequested, [this] {
        NauAboutDialog aboutDialog(this);
        aboutDialog.showModal();

        // TODO: Use analytics singleton
//#ifdef USE_NAU_ANALYTICS
//        sendUIInteractionEvent("main_menu_about_dialog");
//#endif
    });

    // Feedback dialog
    connect(mainMenu, &NauMainMenu::feedbackDialogRequested, [this] {
        if (QDesktopServices::openUrl(QUrl("https://nauengine.org/#b9235e9f-2bcf-4e9d-8af6-b5558ed929eb"))) {
            NED_INFO("URL successfully opened.");
        } else {
            NED_INFO("Failed to open URL.");
        }

        /*
        NauFeedbackDialog feedbackDialog(this);

        // Request from the auto test team
        [[maybe_unused]] const bool isSuccessFeedback = feedbackDialog.setProperty("MyPropertyFeedback", QVariant("Hello, World! Feedback"));

        feedbackDialog.showModal();
        */
    });

    NED_DEBUG("The editor window is initialized");
}

NauEditorWindow::~NauEditorWindow()
{
}

void NauEditorWindow::closeEvent(QCloseEvent* event)
{
    // window client need to handle this event independently
    emit eventCloseWindow(event);
}

void NauEditorWindow::showEvent(QShowEvent* event)
{
    NauMainWindow::showEvent(event);

    // Load default perspective when the editor window shows up for the first time.
    // Note We have to set docking proportion here for ads::CDockAreaWidget::splitterSizes
    // doesn't return a valid values before.
    if (!event->spontaneous() && !m_mainWindowAppearance) {
        loadDefaultUiPerspective();
    }
}

void NauEditorWindow::closeWindow(QCloseEvent* event)
{
    m_dockManager->deleteLater();
    NauMainWindow::closeEvent(event);
}

#ifdef Q_OS_WIN
bool NauEditorWindow::nativeEvent([[maybe_unused]] const QByteArray& eventType, void* _message, qintptr* result)
{
    auto* message = static_cast<MSG*>(_message);
    UINT messageType = message->message;
    static constexpr int borderSize = NauTitleBar::ResizeOffset;

    if (messageType == WM_NCCALCSIZE && message->wParam && message->lParam) {  // Allow some space for native resize, top edge is handled separately in mousePressEvent
         NCCALCSIZE_PARAMS& params = *reinterpret_cast<NCCALCSIZE_PARAMS*>(message->lParam);

        params.rgrc[0].left += borderSize;
        params.rgrc[0].right -= borderSize;
        params.rgrc[0].bottom -= borderSize;

        if (isMaximized() && !m_titleBar->moveRequestPending()) {
            params.rgrc[0].top += borderSize;
        }

        *result = 0;
        return true;
    }

    return QWidget::nativeEvent(eventType, message, result);
}
#endif

void NauEditorWindow::mouseMoveEvent(QMouseEvent* event)
{
    m_titleBar->resetMoveAndResizeRequests();
    NauMainWindow::mouseMoveEvent(event);
}

void NauEditorWindow::resizeEvent(QResizeEvent* event)
{
    m_titleBar->resetMoveAndResizeRequests();
    NauMainWindow::resizeEvent(event);
}

void NauEditorWindow::loadDefaultUiPerspective()
{
    // TODO: clean up
    // Splitters are 4 pixels wide by design.
    constexpr int HandleWidth = 4;

    // 1. Align this window at center of 1st screen.
    const QRect screenRect = QGuiApplication::screens().at(0)->geometry();

    QRect myGeometry = geometry();
    myGeometry.moveCenter(screenRect.center());
    setGeometry(myGeometry);

    auto const applyProportion = [this](const QList<int>&sizes, std::vector<float> const& proportion) {
        static const float eps = 0.001f;
        NED_ASSERT(std::abs(std::accumulate(proportion.begin(), proportion.end(), 0.0f) - 1.0f) < eps);

        if (sizes.size() != proportion.size()) return sizes;

        QList<int> result = sizes;

        const int sum = std::accumulate(std::begin(sizes), std::end(sizes), 0);
        for (size_t idx = 0; idx < proportion.size(); ++idx) {
            result[idx] = proportion[idx] * sum;
        }

        return result;
    };

    // 2. Set the width of the windows when the editor is first opened to match the application's resolution.
    
    // TODO: This code is the beginning of a potential system that will be responsible for scaling the editor
    // interface to the right proportions.

    // In the future we will build a scaling curve, 
    // where the x - axis will be the resolution of the side,
    // and the y - axis will be the scaling of this or that element.

    // Thus we will get a universal system that will
    // not simply scale 1:1 windows and will not accumulate
    // a math error on the most common screen resolutions.

    std::unordered_map<int, QList<int>> typicalResolutionSplitterSizes;
    typicalResolutionSplitterSizes[1280] = { 340, 600  - 2 * HandleWidth, 340 };
    typicalResolutionSplitterSizes[1366] = { 340, 686  - 2 * HandleWidth, 340 };
    typicalResolutionSplitterSizes[1920] = { 424, 1072 - 2 * HandleWidth, 424 };
    typicalResolutionSplitterSizes[2560] = { 532, 1496 - 2 * HandleWidth, 532 };
    typicalResolutionSplitterSizes[3840] = { 532, 2776 - 2 * HandleWidth, 532 };

    // 3. Set proportion of top level widgets(Scenes, Viewport, Inspector).
    auto topLevelArea = m_dockManager->dockArea(1);
    if (!topLevelArea) return;

    if (typicalResolutionSplitterSizes.contains(screenRect.width())) {
        m_dockManager->setSplitterSizes(topLevelArea, typicalResolutionSplitterSizes[screenRect.width()]);
    } else {
        static const std::vector<float> topLevelWidgetsProportion{0.222f, 0.556f, 0.222f};
        m_dockManager->setSplitterSizes(topLevelArea, applyProportion(m_dockManager->splitterSizes(topLevelArea), topLevelWidgetsProportion));
    }

    auto titleBarHeight = m_titleBar->geometry().height();
    auto toolbarHeight = m_toolbar->geometry().height();

    // 4. Set the height of the windows when the editor is first opened to match the application's resolution.
    // TODO: And this code will be removed when we can hide tabs normally on first launch.
    // But it will still be taken into account in the scaling system,
    // because the initial window size will still have to be set depending on the target screen resolution.

    std::unordered_map<int, QList<int>> typicalResolutionSplitterSizesDown;
    typicalResolutionSplitterSizesDown[720] =  {720  - 2 * HandleWidth - titleBarHeight - toolbarHeight - 328, 328};
    typicalResolutionSplitterSizesDown[768] =  {768  - 2 * HandleWidth - titleBarHeight - toolbarHeight - 328, 328};
    typicalResolutionSplitterSizesDown[1080] = {1080 - 2 * HandleWidth - titleBarHeight - toolbarHeight - 424, 424};
    typicalResolutionSplitterSizesDown[1440] = {1440 - 2 * HandleWidth - titleBarHeight - toolbarHeight - 532, 532};
    typicalResolutionSplitterSizesDown[2560] = {2560 - 2 * HandleWidth - titleBarHeight - toolbarHeight - 532, 532};

    // 5. Set proportion of bottom level widgets(Project Browser, Output).
    auto bottomLevelArea = m_dockManager->dockArea(0);
    if (!bottomLevelArea) return;

    if (typicalResolutionSplitterSizesDown.contains(screenRect.height())) {
        m_dockManager->setSplitterSizes(bottomLevelArea, typicalResolutionSplitterSizesDown[screenRect.height()]);
    } else {
        static const std::vector<float> bottomLevelWidgetsProportion{0.61f, 0.39f};
        m_dockManager->setSplitterSizes(bottomLevelArea, applyProportion(m_dockManager->splitterSizes(bottomLevelArea), bottomLevelWidgetsProportion));
    }
}

void NauEditorWindow::maybeLoadMainWindowAppearance()
{
    // TODO: Perhaps this code causes errors with window scaling calculation in the editor, when minimizing/expanding a window multiple times.
    // We need to study how it works in the future.
    if (!m_mainWindowAppearance) {
        return;
    }
    NED_TRACE("Trying to load user defined main window settings");

    const auto& appearance = *m_mainWindowAppearance;

    const bool geometrySuccess = restoreGeometry(appearance.geometryState);
    const bool dockingSuccess = m_dockManager->restoreState(appearance.dockingState);
    m_titleBar->setMaximized(appearance.isMaximized);

    NED_DEBUG("Main window appearance restored with: geometry {}, docking {}, maximized {}", 
        geometrySuccess, dockingSuccess, appearance.isMaximized);
}

NauWorldOutlinerWidget* NauEditorWindow::outliner() const noexcept
{
    return m_worldOutline;
}

NauInspectorPage* NauEditorWindow::inspector() const noexcept
{
    return m_inspector;
}

NauViewportContainerWidget* NauEditorWindow::viewport() const noexcept
{
    return m_viewport;
}

NauDockManager* NauEditorWindow::dockManager() const noexcept
{
    return m_dockManager;
}

NauShortcutHub* NauEditorWindow::shortcutHub() const noexcept
{
    return m_shortcutHub;
}

NauTitleBar* NauEditorWindow::titleBar() const noexcept
{
    return m_titleBar;
}

NauMainToolbar* NauEditorWindow::toolbar() const noexcept
{
    return m_toolbar;
}

NauTabbedLoggerWidget* NauEditorWindow::logger() const noexcept
{
    return m_logger;
}

NauProjectBrowser* NauEditorWindow::projectBrowser() const noexcept
{
    return m_projectBrowser;
}

void NauEditorWindow::applyUserSettings(const NauProject& project)
{
    // Outline settings
    if (const auto columns = project.userSettings().worldOutlineVisibleColumns(); !columns.empty()) {
        m_worldOutline->outlinerTab().setColumnVisibility(columns);
    }

    // Outline settings
    connect(&m_worldOutline->outlinerTab(), &NauWorldOutlineTableWidget::eventColumnsVisibilityChanged, &(project.userSettings()), &NauProjectUserSettings::setWorldOutlineVisibleColumns);

    // Camera settings
    /*NauCameraSettingsWidget* cameraSettings = m_viewport->cameraSettings();
    if (project && project->userSettings().cameraSettings().has_value()) {
        cameraSettings->load(project->userSettings().cameraSettings().value());
    }*/

    m_mainWindowAppearance = project.userSettings().mainWindowAppearance();

    maybeLoadMainWindowAppearance();
}

void NauEditorWindow::saveUserSettings(NauProjectPtr project)
{
    if (!project) {
        NED_ERROR("Failed to save user project settings! Project does not exist");
        return;
    }
    project->userSettings().setMainWindowAppearance({
        isMaximized(),
        saveGeometry(),
        m_dockManager->saveState()
    });

    // Save camera settings
    //NauCameraSettingsWidget* cameraSettings = m_viewport->cameraSettings();
    //cameraSettings->updateSettings(true);  // Force update to refresh all values
    //project->userSettings().setCameraSettings(cameraSettings->save());
}

void NauEditorWindow::setUpdateWindowTitleCallback(const std::function<void()>& cb)
{
    // Temporary title bar update system
    auto windowTitleUpdateTimer = new QTimer(this);
    windowTitleUpdateTimer->callOnTimeout(cb);
    windowTitleUpdateTimer->setInterval(1000 / 60);
    windowTitleUpdateTimer->start();
}