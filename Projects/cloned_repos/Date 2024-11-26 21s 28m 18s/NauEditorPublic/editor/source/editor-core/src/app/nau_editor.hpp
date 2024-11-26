// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Main editor implementation

#pragma once

#include "nau/app/nau_editor_interface.hpp"
#include "nau/assets/nau_asset_editor.hpp"
#include "nau/assets/nau_asset_manager.hpp"
#include "nau/scene/nau_editor_scene_manager_interface.hpp"
#include "nau/outliner/nau_outliner_client_interface.hpp"

#include "commands/nau_commands.hpp"
#include "project/nau_project.hpp"

#ifdef USE_NAU_ANALYTICS
#include "analytics/source/nau_analytics.hpp"
#endif


// ** NauEditor

class NauEditor final : public NauEditorInterface, public NauUndoable
{
public:
    NauEditor(NauProjectPtr project);
    ~NauEditor();

    const NauEditorWindowAbstract& mainWindow() const noexcept override;
    NauAssetManagerInterfacePtr assetManager() const noexcept override;
    NauThumbnailManagerInterfacePtr thumbnailManager() const noexcept override;
    NauUndoable* undoRedoSystem() override;

    void showMainWindow() override;
    void switchScene(const NauSceneManager::SceneInfo& scene) override;
    void postInit() override;

private:
    // UI initialzation
    void initMainWindow();
    void initMainMenu();
    void initToolbar();
    void initProjectBrowser();
    void initSceneManager();

    // Main editor events handlers
    void onEditorClosed(QCloseEvent* event);
    void onUpdateTitlebar();

    // Toolbar handlers
    void handleUndo();
    void handleRedo();
    void handleLanguageRequest(const QString& lang);
    void showCommandHistory(const NauToolButton& button);
    void startPlayModeRequested();
    void stopPlayModeRequested();
    void pausePlayModeRequested(const bool isPaused) const;
    void openBuildWindow();

    // Scene manager handlers
    void showProjectDialog(); //unused
    bool showUnsavedProjectDialog(); //unused
    bool showUnsavedSceneDialog(bool showCancel = true);
    void loadScene(const std::string& path);
    void unloadCurrentScene();  

    // Project managment
    void openProject(NauProjectPtr project);

    // Modules loading
    void initModules();
    void terminateModules();

    void initSceneEditorModule();
    void initAssetManagerModule();

    void registerAssetEditor(NauAssetEditorInterface* editor);
    void configureProjectBrowser();
    void configureThumbnailManager();

private:
#ifdef USE_NAU_ANALYTICS
    void initAnalytics();
    void initCountlyAnalyticsProvider(const QJsonObject& countlyCredentials, const QString& userId, bool isUserExist);

    // Nau Editor Event Analytics block
    void sendEditorInfoEvent();
    void sendSystemInfoEvent();

    void sendUIInteractionEvent(const std::string& uiName);
#endif

private:
    inline static const char* const m_launchLoggerName = "Build";

    std::unique_ptr<NauEditorWindowAbstract> m_mainWindow;
    NauAssetManagerInterfacePtr m_assetManager;
    NauThumbnailManagerInterfacePtr m_thumbnailManager;
    std::vector<NauAssetEditorInterface*> m_registeredEditors;

    // Scene managment
    std::shared_ptr<NauEditorSceneManagerInterface> m_sceneManager;

#ifdef USE_NAU_ANALYTICS
    // TODO: Make analytics a singleton
    // Analytics
    std::unique_ptr<NauAnalytics> m_analytics;
#endif
};