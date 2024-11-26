// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Main NauGui editor class

#pragma once

#include "outlainer/nau_gui_outliner_client.hpp"
#include "nau/inspector/nau_usd_inspector_client.hpp"

#include "nau/scene-manager/nau_usd_scene_manager.hpp"
#include "nau/undo-redo/nau_usd_scene_undo_redo.hpp"
#include "nau/selection/nau_usd_selection_container.hpp"
#include "nau/ui-translator/nau_usd_scene_ui_translator.hpp"

#include "nau/assets/nau_asset_watcher.hpp"
#include "nau/assets/nau_asset_editor.hpp"
#include "nau/app/nau_editor_interface.hpp"
#include "nau/app/nau_editor_window_interface.hpp"

#include "nau/rtti/rtti_impl.h"

#include "nau/nau_ui_asset_editor_synchronizer.hpp"

#include <QObject>

#include <memory>

#include <pxr/usd/usd/notice.h>
#include <pxr/base/tf/notice.h>

#include <pxr/base/tf/weakBase.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/sdf/notice.h>


// ** NauGuiEditorShapshot

class NauGuiEditorSnapshot
{
    struct Info
    {
        std::string scenePath;
    };

public:
    NauGuiEditorSnapshot() = default;
    ~NauGuiEditorSnapshot() = default;

    void takeShapshot(class NauGuiEditor* editor);
    void restoreShapshot();

private:
    std::unique_ptr<Info> m_snapshotInfo;
    class NauGuiEditor* m_editor;
};
using NauGuiEditorSnapshotPtr = std::unique_ptr<NauGuiEditorSnapshot>;


// ** NauGuiEditor

class AttributeChangeListener;

class NauGuiEditor final : public NauAssetEditorInterface, public NauAssetManagerClientInterface
{
    NAU_CLASS_(NauGuiEditor, NauAssetEditorInterface)

public:
    NauGuiEditor();
    ~NauGuiEditor();

    void initialize(NauEditorInterface* mainEditor) override;
    void terminate() override;
    void postInitialize() override;
    void preTerminate() override;

    void createAsset(const std::string& assetPath) override;
    bool openAsset(const std::string& assetPath) override;
    bool saveAsset(const std::string& assetPath = "") override;

    std::string editorName() const override;
    NauEditorFileType assetType() const override;

    std::shared_ptr<NauEditorSceneManagerInterface> sceneManager();
    std::shared_ptr<NauOutlinerClientInterface> outlinerClient();

    void handleSourceAdded(const std::string& sourcePath) override;
    void handleSourceRemoved(const std::string& sourcePath) override;

    void startPlay() override;
    void stopPlay() override;

    void openEditorPanel();
    void createEditorPanel();

private:

    NauGuiEditor(const NauGuiEditor&) = default;
    NauGuiEditor(NauGuiEditor&&) = default;
    NauGuiEditor& operator=(const NauGuiEditor&) = default;
    NauGuiEditor& operator=(NauGuiEditor&&) = default;

    void initPrimFactory();
    void initSceneManager();
    void initSelectionContainer();
    void initSceneUndoRedo();
    void initInspectorClient();
    void initOutlinerClient();

    void onSceneLoaded(pxr::UsdStageRefPtr scene);
    void onSceneUnloaded();

    PXR_NS::UsdPrim createPrim(const PXR_NS::SdfPath& parentPath, const PXR_NS::TfToken& name, const PXR_NS::TfToken& typeName, bool isComponent);
    void addPrimsFromOther(const std::vector<PXR_NS::UsdPrim>& prims);
    void addPrimsFromOther(const std::vector<PXR_NS::UsdPrim>& prims, const PXR_NS::SdfPath& pathToAdd);
    void addPrimFromOther(const pxr::SdfPathSet& primsPaths, PXR_NS::UsdPrim other, const PXR_NS::SdfPath& parentPath);

    void removePrims(const std::vector<PXR_NS::UsdPrim>& normalizedPrimList);
    void removePrim(const PXR_NS::UsdPrim& prim);

    // Loads default layout of docking widgets.
    // Used as alternative of user defined perspectives in case of there are no such one.
    void loadDefaultUiPerspective();

private:
    std::shared_ptr<NauUsdSceneManager> m_sceneManager;
    std::shared_ptr<NauUsdOutlinerClient> m_outlinerClient;
    std::shared_ptr<NauUsdInspectorClient> m_inspectorClient;
    std::unique_ptr<AttributeChangeListener> m_primChangeListener;

    NauUsdSceneUndoRedoSystemPtr m_sceneUndoRedoSystem;
    NauUsdSelectionContainerPtr m_selectionContainer;

    std::unique_ptr<NauUsdSceneUITranslator> m_uiTranslator;
    std::unique_ptr<NauUsdUIAssetEditorSynchronizer> m_sceneEditorSynchronizer;

    NauDockManager* m_editorDockManger;
    NauEditorInterface* m_mainEditor;

    std::string m_assetPath;
    std::unique_ptr<NauAssetWatcher> m_uiWatcher;

    // TODO: Combine in one editor scene class
    pxr::UsdStageRefPtr m_scene;
    nau::scene::IWorld::WeakRef m_coreWorld;

    // TODO: Combine in one widget class
    NauDockManager* m_guiEditorDockManger;
    NauDockWidget* m_dwEditorPanel;
    NauWidget* m_editorPanel;
    NauViewportContainerWidget* m_viewportContainer;
    NauWorldOutlinerWidget* m_worldOutline;
    NauInspectorPage* m_inspector;

    // Snapshot for playmode
    NauGuiEditorSnapshotPtr m_snapshot;
};


class AttributeChangeListener : public pxr::TfWeakBase
{
public:
    explicit AttributeChangeListener(const pxr::UsdStageRefPtr& stage, std::shared_ptr<NauUsdInspectorClient> inspectorClient) 
    : m_stage(stage), m_isSubscribed(false)
    {        
        subscribeToChanges();
        m_inspectorClient = inspectorClient;
    }

    ~AttributeChangeListener() 
    {
        unsubscribeFromChanges();
    }

    void OnObjectsChanged(const pxr::UsdNotice::ObjectsChanged& notice , const pxr::UsdStageWeakPtr& _) 
    {
        for (const auto& path : notice.GetChangedInfoOnlyPaths()) 
        {
            pxr::SdfPath primPath = path.IsPrimPath() 
                ? path 
                : path.IsPropertyPath() ? path.GetPrimPath() : path.EmptyPath();

            if (primPath.IsEmpty()) 
            {
                NAU_LOG_ERROR("Changed prim path is empty");

                continue;       
            }

            auto prim = m_stage->GetPrimAtPath(primPath);
            if (prim) 
            {
                m_inspectorClient->updateFromPrim(prim);
            }
        }
    }

private:
    pxr::UsdStageRefPtr m_stage;
    pxr::TfNotice::Key m_noticeKey;
    bool m_isSubscribed;
    std::shared_ptr<NauUsdInspectorClient> m_inspectorClient;

    void subscribeToChanges() 
    {
        if (m_isSubscribed) 
        {
            return;
        }

        m_noticeKey = pxr::TfNotice::Register(
            pxr::TfWeakPtr<AttributeChangeListener>(this),
            &AttributeChangeListener::OnObjectsChanged,
            m_stage);

        m_isSubscribed = true;
    }

    void unsubscribeFromChanges() 
    {
        if (m_isSubscribed) 
        {
            pxr::TfNotice::Revoke(m_noticeKey);
            m_isSubscribed = false;
        }
    }
};