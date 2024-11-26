// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_vfx_editor.hpp"

#include "usd_proxy/usd_proxy.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/base/tf/token.h>
#include "components/pages/nau_vfx_editor_page.hpp"
#include "nau/assets/asset_descriptor.h"
#include "nau/assets/asset_manager.h"
#include "nau/service/service_provider.h"
#include "nau/service/service_provider.h"

#include "nau/nau_usd_scene_editor.hpp"

// ** NauVFXEditor

NauVFXEditor::NauVFXEditor()
    : m_inspectorWithVFX(nullptr)
{

}

NauVFXEditor::~NauVFXEditor()
{
    terminate();
}

void NauVFXEditor::initialize(NauEditorInterface* mainEditor)
{
    m_mainEditor = mainEditor;
    m_editorDockManger = mainEditor->mainWindow().dockManager();
    //m_mainInspector = mainEditor->mainWindow().inspector();
}


void NauVFXEditor::terminate()
{
    // TODO: Reset resources
}

void NauVFXEditor::postInitialize()
{
    auto assetManager = m_mainEditor->assetManager();
    assetManager->addClient({NauEditorFileType::VFX}, this);
}

void NauVFXEditor::preTerminate()
{
    // TODO: unbind ui from NauEditor window
}

void NauVFXEditor::createAsset(const std::string& assetPath)
{
    if (!assetPath.empty()) {
        auto stage = PXR_NS::UsdStage::CreateInMemory(assetPath);
        auto prim = stage->DefinePrim(PXR_NS::SdfPath("/VFX"), PXR_NS::TfToken("VFXInstance"));
        stage->SetDefaultPrim(prim);
        stage->GetRootLayer()->Export(assetPath);
    }
}

bool NauVFXEditor::openAsset(const std::string& assetPath)
{
    if (m_mainInspector == nullptr) {
        return openAssetInNewWindow(QString(assetPath.c_str()));
    }

    loadVFXData(QString(assetPath.c_str()), *m_mainInspector);

    NED_DEBUG("VFX asset {} opened in the inspector.", assetPath);
    return true;
}

bool NauVFXEditor::saveAsset(const std::string& assetPath)
{
    if (!m_vfxAsset) {
        NED_ERROR("Trying to save non-existent material asset");
        return false;
    }

    NED_TRACE("VFX asset saved to {}.", assetPath);
    m_vfxAsset->Save();
    
    // Reimport asset process
    // TODO: Temp solutiuon. Will be deleted
    auto root = m_vfxAsset->GetPseudoRoot();
    auto children = root.GetAllChildren();
    if (children.empty()) {
        return false;
    }

    const auto assetMetaPath = std::format("{}.{}", m_vfxAsset->GetRootLayer()->GetRealPath(), "nausd");
    auto assetMetaStage = pxr::UsdStage::Open(assetMetaPath);

    auto vfxPrim = assetMetaStage->GetPrimAtPath(pxr::SdfPath("/Root/VFX"));

    UsdProxy::UsdProxyPrim proxyPrim(vfxPrim);
    auto proxyProp = proxyPrim.getProperty(pxr::TfToken("uid"));
    if (!proxyProp) {
        return false;
    }
    pxr::VtValue uidVal;
    proxyProp->getValue(&uidVal);
    auto coreMatPath = "uid:" + uidVal.Get<std::string>();

    // Unload asset from engine
    nau::IAssetDescriptor::Ptr asset = nau::getServiceProvider().get<nau::IAssetManager>().openAsset(nau::strings::toStringView(coreMatPath));
    nau::IAssetDescriptor::LoadState state = asset->getLoadState();

    nau::IAssetDescriptor::UnloadResult unloadResult = nau::IAssetDescriptor::UnloadResult::Unloaded;
    if (state == nau::IAssetDescriptor::LoadState::Ready) {
        unloadResult = asset->unload();
    }

    // Reimport material
    m_mainEditor->assetManager()->importAsset(assetPath);

    // Load material again if needed
    state = asset->getLoadState();
    if ((unloadResult == nau::IAssetDescriptor::UnloadResult::UnloadedHasReferences) && (state == nau::IAssetDescriptor::LoadState::None)) {
        asset->load();
    }

    // TODO: Temporary. Reload materials on prims with current material
    // Need to change engine asset directly 
    auto& mainSceneEditor = Nau::EditorServiceProvider().get<NauUsdSceneEditorInterface>();
    auto translator = mainSceneEditor.sceneSynchronizer().translator();

    std::function<void(UsdTranslator::IPrimAdapter::Ptr)> updatePrimVFX;
    std::filesystem::path currentAssetPath = m_vfxAssetPath;
    updatePrimVFX = [currentAssetPath, &updatePrimVFX, translator, &uidVal](UsdTranslator::IPrimAdapter::Ptr adapter) {
        if (adapter->getType() == "NauAssetVFX") {
            if (pxr::VtValue vfxPath; UsdProxy::UsdProxyPrim(adapter->getPrim()).getProperty("uid"_tftoken)->getValue(&vfxPath))
            {
                if (vfxPath.CanCast<std::string>())
                {
                    auto assetUID = vfxPath.Get<std::string>();
                    if (assetUID == uidVal.Get<std::string>()) {
                        translator->forceUpdate(adapter->getPrim());
                    }
                }
            }

        }

        for (auto child : adapter->getChildren()) {
            updatePrimVFX(child.second);
        }
    };
    updatePrimVFX(translator->getRootAdapter());

    NED_TRACE("VFX asset saved to {}.", assetPath);
    return true;
}

std::string NauVFXEditor::editorName() const
{
    return "VFX Editor";
}

NauEditorFileType NauVFXEditor::assetType() const
{
    return NauEditorFileType::VFX;
}

void NauVFXEditor::handleSourceAdded(const std::string& sourcePath)
{
}

void NauVFXEditor::handleSourceRemoved(const std::string& sourcePath)
{
}

bool NauVFXEditor::openAssetInNewWindow(const QString& assetPath)
{
    ads::CDockWidget* inspector = m_editorDockManger->findDockWidget("DockInspector");
    if (inspector == nullptr) {
        NED_ERROR("Failed to open VFX editor in tab.");
        return false;
    }

    const QFileInfo assetFileInfo(assetPath);
    const QString assetName = assetFileInfo.fileName();

    if (!m_dwMVFXPropertyPanel) {
        m_dwMVFXPropertyPanel = new NauDockWidget(QObject::tr(assetName.toUtf8().constData()), nullptr);
        m_dwMVFXPropertyPanel->setStyleSheet("background-color: #282828");
        m_dwMVFXPropertyPanel->setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromContent);

        m_inspectorWithVFX = new NauInspectorPage(nullptr);
        m_dwMVFXPropertyPanel->setWidget(m_inspectorWithVFX);
    }

    m_editorDockManger->addDockWidgetTabToArea(m_dwMVFXPropertyPanel, inspector->dockAreaWidget());
    m_dwMVFXPropertyPanel->toggleView(true);

    loadVFXData(assetPath, *m_inspectorWithVFX);

    NED_DEBUG("VFX asset {} opened in new window.", assetPath.toUtf8().constData());

    return true;
}

void NauVFXEditor::loadVFXData(const QString& assetPath, NauInspectorPage& inspector)
{
    m_inspectorWithVFX = &inspector;
    m_vfxAssetPath = assetPath.toUtf8().constData();

    m_vfxAsset = pxr::UsdStage::Open(m_vfxAssetPath);

    if (!m_vfxAsset) {
        NED_ERROR("Error while opening vfx asset at path {}", m_vfxAssetPath);
        return;
    }

    m_sceneUndoRedoSystem = std::make_shared<NauUsdSceneUndoRedoSystem>(m_mainEditor->undoRedoSystem());
    m_sceneUndoRedoSystem->bindCurrentScene(m_vfxAsset);

    m_inspectorClient = std::make_shared<NauUsdInspectorClient>(m_inspectorWithVFX);
    m_inspectorClient->connect(m_inspectorClient.get(), &NauUsdInspectorClient::eventPropertyChanged, [this](const PXR_NS::SdfPath& path, const PXR_NS::TfToken propName, const PXR_NS::VtValue& value) {
        m_sceneUndoRedoSystem->addCommand<NauCommandChangeUsdPrimProperty>(path, propName, value);

        saveAsset(m_vfxAssetPath);
    });

    m_inspectorClient->connect(m_inspectorClient.get(), &NauUsdInspectorClient::eventAssetReferenceChanged, [this](const PXR_NS::SdfPath& path, const PXR_NS::VtValue& value) {
        m_sceneUndoRedoSystem->addCommand<NauCommandChangeUsdPrimAssetReference>(path, value);
        saveAsset(m_vfxAssetPath);
    });

    auto rootPrim = m_vfxAsset->GetPseudoRoot();
    auto children = rootPrim.GetAllChildren();
    auto vfxPipelinePrim = children.front();
    m_inspectorClient->buildFromPrim(vfxPipelinePrim);
}