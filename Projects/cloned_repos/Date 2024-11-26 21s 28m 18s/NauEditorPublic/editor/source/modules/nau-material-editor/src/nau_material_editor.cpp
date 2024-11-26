// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_material_editor.hpp"

#include "nau_material_preview.hpp"
#include "nau_log.hpp"
#include "nau_assert.hpp"

#include "nau/undo-redo/nau_usd_scene_commands.hpp"

#include "nau/usd_meta_tools/usd_meta_generator.h"

#include <QFileInfo>

// TODO: Temporary. Needed for update materials in all prims with this material
#include "nau/nau_usd_scene_editor.hpp"

#include "nau/assets/asset_descriptor.h"
#include "nau/assets/asset_manager.h"


// ** NauMaterialEditor

NauMaterialEditor::NauMaterialEditor()
    : m_inspectorWithMaterial(nullptr)
{
}

void NauMaterialEditor::initialize(NauEditorInterface* mainEditor)
{
    m_mainEditor = mainEditor;
    m_editorDockManger = mainEditor->mainWindow().dockManager();
}

void NauMaterialEditor::terminate()
{
    // TODO: Reset resources
}

void NauMaterialEditor::postInitialize()
{
    // TODOL ui initialize
}

void NauMaterialEditor::preTerminate()
{
    // TODOL unbind ui from NauEditor window
}

void NauMaterialEditor::createAsset(const std::string& assetPath)
{
    nau::UsdMetaGenerator::instance().generateAssetTemplate(assetPath, "Material", nau::MetaArgs());
    if (!std::filesystem::exists(assetPath)) {
        NED_ERROR("Failed to create material asset.");
    }
}

bool NauMaterialEditor::openAsset(const std::string& assetPath)
{
    if (m_mainInspector == nullptr) {
        return openAssetInNewWindow(QString(assetPath.c_str()));
    }

    loadMaterialData(QString(assetPath.c_str()), *m_mainInspector);

    NED_DEBUG("Material asset {} opened in the inspector.", assetPath);
    return true;
}

bool NauMaterialEditor::saveAsset(const std::string& assetPath)
{
    if (!m_materialAsset) {
        NED_ERROR("Trying to save non-existent material asset");
        return false;
    }

    m_materialAsset->Save();   

    // Reimport asset process
    // TODO: Temp solutiuon. Will be deleted
    auto root = m_materialAsset->GetPseudoRoot();
    auto children = root.GetAllChildren();
    if (children.empty()) {
        return false;
    }
    auto materialPrim = children.front();

    UsdProxy::UsdProxyPrim proxyPrim(materialPrim);
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

    std::function<void(UsdTranslator::IPrimAdapter::Ptr)> updatePrimMaterial;
    std::filesystem::path currentAssetPath = m_materialAssetPath;
    updatePrimMaterial = [currentAssetPath, &updatePrimMaterial, translator](UsdTranslator::IPrimAdapter::Ptr adapter) {
        if (adapter->getType() == "AssetMesh") {
            if (pxr::VtValue materialPath; UsdProxy::UsdProxyPrim(adapter->getPrim()).getProperty("Material:assign"_tftoken)->getValue(&materialPath))
            {
                if(materialPath.CanCast<pxr::SdfAssetPath>())
                {
                    std::filesystem::path assetPath = materialPath.Get<pxr::SdfAssetPath>().GetResolvedPath();
                    if (assetPath == currentAssetPath) {
                        translator->forceUpdate(adapter->getPrim());
                    }
                }
            }
            
        }

        for (auto child : adapter->getChildren()) {
            updatePrimMaterial(child.second);
        }
    };
    updatePrimMaterial(translator->getRootAdapter());

    NED_TRACE("Material asset saved to {}.", assetPath);
    return true;
}

std::string NauMaterialEditor::editorName() const
{
    return "Material editor";
}

NauEditorFileType NauMaterialEditor::assetType() const
{
    return NauEditorFileType::Material;
}

void NauMaterialEditor::handleSourceAdded(const std::string& path)
{
    // Unused
}

void NauMaterialEditor::handleSourceRemoved(const std::string& assetPath)
{
    // TODO: implement
}

bool NauMaterialEditor::openAssetInNewWindow(const QString& assetPath)
{
    const QFileInfo assetFileInfo(assetPath);
    const QString assetName = assetFileInfo.fileName();

    if (!m_dwMaterialPropertyPanel) {  
        m_dwMaterialPropertyPanel = new NauDockWidget(QObject::tr(assetName.toUtf8().constData()), nullptr);
        m_dwMaterialPropertyPanel->setStyleSheet("background-color: #282828");
        m_dwMaterialPropertyPanel->setObjectName("DockingMaterialProp");
        m_dwMaterialPropertyPanel->setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromContent);

        m_inspectorWithMaterial = new NauInspectorPage(nullptr);
        m_dwMaterialPropertyPanel->setWidget(m_inspectorWithMaterial);
    }

    ads::CDockWidget* inspector = m_editorDockManger->inspector();
    if (inspector == nullptr ) {
        NED_ERROR("Failed to open Material editor in tab.");
        return false;
    }

    m_editorDockManger->addDockWidgetTabToArea(m_dwMaterialPropertyPanel, inspector->dockAreaWidget());
    m_dwMaterialPropertyPanel->toggleView(true);

    loadMaterialData(assetPath, *m_inspectorWithMaterial);

    NED_DEBUG("Material asset {} opened in new window.", assetPath.toUtf8().constData());

    return true;
}

void NauMaterialEditor::loadMaterialData(const QString& assetPath, NauInspectorPage& inspector)
{
    m_inspectorWithMaterial = &inspector;
    m_materialAssetPath = assetPath.toUtf8().constData();

    m_materialAsset = pxr::UsdStage::Open(m_materialAssetPath);

    if (!m_materialAsset) {
        NED_ERROR("Error while opening material asset at path {}", m_materialAssetPath);
        return;
    }

    m_sceneUndoRedoSystem = std::make_shared<NauUsdSceneUndoRedoSystem>(m_mainEditor->undoRedoSystem());
    m_sceneUndoRedoSystem->bindCurrentScene(m_materialAsset);

    m_inspectorClient = std::make_shared<NauUsdInspectorClient>(m_inspectorWithMaterial);
    m_inspectorClient->connect(m_inspectorClient.get(), &NauUsdInspectorClient::eventPropertyChanged, [this](const PXR_NS::SdfPath& path, const PXR_NS::TfToken propName, const PXR_NS::VtValue& value) {
        m_sceneUndoRedoSystem->addCommand<NauCommandChangeUsdPrimProperty>(path, propName, value);

        saveAsset(m_materialAssetPath);
    });

    m_inspectorClient->connect(m_inspectorClient.get(), &NauUsdInspectorClient::eventAssetReferenceChanged, [this](const PXR_NS::SdfPath& path, const PXR_NS::VtValue& value) {
        m_sceneUndoRedoSystem->addCommand<NauCommandChangeUsdPrimAssetReference>(path, value);

        saveAsset(m_materialAssetPath);
    });

    auto rootPrim = m_materialAsset->GetPseudoRoot();
    auto children = rootPrim.GetAllChildren();
    if (children.empty()) {
        NED_ERROR("Invalid material");
        return;
    }

    // TODO: Now we can build only from one NauMaterialPipline
    auto materialPipelinePrim = children.front();
    m_inspectorClient->buildFromMaterial(materialPipelinePrim);
}