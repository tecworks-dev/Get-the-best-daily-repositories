// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Animation editing classes


#pragma once

#include "nau/rtti/rtti_impl.h"

#include "widgets/nau_timeline_window.hpp"
#include "baseWidgets/nau_widget.hpp"
#include "nau_dock_manager.hpp"
#include "inspector/nau_object_inspector.hpp"
#include "nau/assets/nau_file_types.hpp"
#include "nau/assets/nau_asset_editor.hpp"
#include "nau/app/nau_editor_interface.hpp"

#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/reference.h>
#include <pxr/usd/usd/prim.h>

#include <map>


namespace nau::animation
{
    class AnimationComponent;
}

namespace nau
{
    class SkeletonComponent;
}

class NauUsdSceneEditorInterface;

using NauAnimationPathList = std::vector<pxr::SdfReference>;

// ** NauAnimationClipEditor

class NauAnimationClipEditor final : public QObject, public NauAssetEditorInterface, public NauAssetManagerClientInterface
{
    Q_OBJECT

    NAU_CLASS_(NauAnimationClipEditor, NauAssetEditorInterface)

public:
    NauAnimationClipEditor();

    void initialize(NauEditorInterface* mainEditor) override;
    void terminate() override;
    void postInitialize() override;
    void preTerminate() override;

    void createAsset(const std::string& assetPath) override;
    bool openAsset(const std::string& assetPath) override;
    bool saveAsset(const std::string& assetPath) override;

    [[nodiscard]] std::string editorName() const override;
    [[nodiscard]] NauEditorFileType assetType() const override;
    [[nodiscard]] NauDockWidget& dockWidget() noexcept;

private:
    enum class AnimationState
    {
        Play,
        Pause,
        Stop,
    };

    enum class AnimationType
    {
        None,
        Clip,
        Skeleton
    };

private slots:
    void updateControllerComponent();

    void handleSourceAdded(const std::string& sourcePath) override;
    void handleSourceRemoved(const std::string& sourcePath) override;

    [[nodiscard]]
    NauAnimationNameList createNameList();
    
    void updateControllerPrim();
    void tryGetControllerComponent();
    void switchClip(int clipIndex);
    void addTrack(int propertyIndex);
    void deleteTrack(int propertyIndex);
    void addKeyframe(int propertyIndex, const NauAnimationPropertyData& data, float time);
    void deleteKeyframe(int propertyIndex, float time);
    void changeKeyframeTime(int propertyIndex, float timeOld, float timeNew);

    void loadPathList();
    void loadSkelProperties();
    void loadClipProperties();
    void loadPropertyList();
    void resetComponents();
    void updateAnimationDuration();
    void setControllerComponentTime(float time);
    void setAnimationState(AnimationState state);
    void createClipAsset();
    void addAnimationAsset(const std::string& assetPath, const pxr::SdfPath& primPath);
    void setSelectedObject(const pxr::UsdPrim& prim, bool needReset);

private:
    using SkeletonComponent = nau::SkeletonComponent;
    using AnimationComponent = nau::animation::AnimationComponent;

    NauEditorInterface* m_mainEditor;
    NauDockManager* m_editorDockManger;
    NauUsdSceneEditorInterface* m_usdSceneEditor;
    AnimationComponent* m_controllerPtr;

    std::unique_ptr<NauDockWidget> m_editorDockWidget;
    std::unique_ptr<NauTimelineWindow> m_timelineWindow;
    std::unique_ptr<QTimer> m_creationComponentTimer;
    std::unique_ptr<QTimer> m_updateComponentTimer;

    pxr::UsdStageRefPtr m_assetStage;
    pxr::UsdPrim m_selectedPrim;
    pxr::UsdPrim m_controllerPrim;
    pxr::UsdPrim m_animationPrim;
    pxr::UsdPrim m_skeletonPrim;

    NauAnimationPropertyListPtr m_propertyListPtr;

    AnimationState m_animationState;
    int m_keyIndex = 0;

    const pxr::TfToken m_keyframeToken;
    const pxr::TfToken m_hackToken;
};
