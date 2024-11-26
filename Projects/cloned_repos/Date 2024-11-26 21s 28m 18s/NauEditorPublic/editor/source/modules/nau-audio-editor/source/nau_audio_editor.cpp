#include "nau_audio_editor.hpp"
#include "nau/app/nau_editor_window_interface.hpp"
#include "nau/nau_editor_plugin_manager.hpp"
#include "nau/audio/audio_service.hpp"

#include "nau_audio_container_view.hpp"
#include "usd_proxy/usd_proxy.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/base/tf/token.h>
#include "nau/assets/asset_db.h"
#include "nau/shared/file_system.h"



// ** NauAudioEditor

NauAudioEditor::NauAudioEditor()
    : m_engine(nau::getServiceProvider().get<nau::audio::AudioService>().engine())
    , m_editorDockManger(nullptr)
    , m_mainEditor(nullptr)
    , m_audioInspector(nullptr)
{
}

NauAudioEditor::~NauAudioEditor()
{
    terminate();
}

void NauAudioEditor::initialize(NauEditorInterface* mainEditor)
{
    m_mainEditor = mainEditor;
    m_editorDockManger = m_mainEditor->mainWindow().dockManager();
}

void NauAudioEditor::terminate()
{
    // TODO: Reset resources
}

void NauAudioEditor::postInitialize()
{
    auto assetManager = m_mainEditor->assetManager();
    assetManager->addClient({NauEditorFileType::RawAudio}, this);


    auto& assetDb = nau::getServiceProvider().get<nau::IAssetDB>();

    // Load all sounds in project
    const eastl::vector<nau::AssetMetaInfoBase>& audioAssetsMetaInfo = assetDb.findAssetMetaInfoByKind("RawAudio");
    for (const auto& assetMetaInfo : audioAssetsMetaInfo) {
        handleSourceAdded(std::format("{}.{}", assetMetaInfo.sourcePath.c_str(), assetMetaInfo.sourceType.c_str()));
    }

    // Load all containers
    const auto& containerAssetsMetaInfo = assetDb.findAssetMetaInfoByKind("AudioContainer");
    for (const auto& assetMetaInfo : containerAssetsMetaInfo) {
        loadContainer(assetMetaInfo.sourcePath.c_str());
    }

    m_audioDockWidget = new NauDockWidget(QObject::tr(editorName().data()), nullptr);
}

void NauAudioEditor::preTerminate()
{
    // TODO: unbind ui from NauEditor window
}

void NauAudioEditor::createAsset(const std::string& assetPath)
{
    // Create an audio container
    if (!assetPath.empty()) {
        auto stage = PXR_NS::UsdStage::CreateInMemory(assetPath);
        auto prim = stage->DefinePrim(PXR_NS::SdfPath("/AudioContainer"), PXR_NS::TfToken("AudioContainer"));
        stage->SetDefaultPrim(prim);
        stage->GetRootLayer()->Export(assetPath);
        loadContainer(assetPath);
    }
}

bool NauAudioEditor::openAsset(const std::string& assetPath)
{
    ads::CDockWidget* inspector = m_editorDockManger->inspector();
    if (inspector == nullptr ) {
        NED_ERROR("Failed to open Audio editor in tab.");
        return false;
    }
    
    if (m_audioInspector == nullptr) {
        m_audioInspector = new NauInspectorPage(nullptr);
        m_audioInspector->setParent(m_audioDockWidget);
        m_audioDockWidget->setWidget(m_audioInspector);
    }

    m_editorDockManger->addDockWidgetTabToArea(m_audioDockWidget, inspector->dockAreaWidget());

    const auto containers = m_engine.containerAssets();
    const auto itContainer = std::find_if(containers.begin(), containers.end(), [assetPath](nau::audio::AudioAssetPtr asset) {
        return std::filesystem::path(asset->name().c_str()) == std::filesystem::path(assetPath);
    });

    if (itContainer == containers.end()) {
        NED_ERROR("Couldn't find container at: {}", assetPath);
        return false;
    }

    auto container = *itContainer;
    auto spoiler = new NauComponentSpoiler("Audio Container");
    auto containerView = new NauAudioContainerView(container, spoiler);
    spoiler->addWidget(containerView);
    containerView->connect(containerView, &NauAudioContainerView::eventContainerChanged, [this, container] {
        saveContainer(container);
    });

    m_audioInspector->clear();
    m_audioInspector->addSpoiler(spoiler);
    m_audioInspector->mainLayout()->addStretch(1);

    return true;
}

bool NauAudioEditor::saveAsset(const std::string& assetPath)
{
    // TODO: implement saving
    return true;
}

std::string NauAudioEditor::editorName() const
{
    return "Audio Editor";
}

NauEditorFileType NauAudioEditor::assetType() const
{
    return NauEditorFileType::AudioContainer;
}

void NauAudioEditor::handleSourceAdded(const std::string& path)
{
    if (!findAsset(path)) {
        // TODO: if sound didn't load don't add it to the asset manager!
        std::filesystem::path sourcePath = nau::FileSystemExtensions::resolveToNativePathContentFolder(path);
        if (nau::audio::AudioAssetPtr asset = m_engine.loadSound(sourcePath.string().c_str()); !asset) {
            NED_WARNING("Failed to load {} from the asset db into the engine", path);
        } 
    }
}

void NauAudioEditor::handleSourceRemoved(const std::string& assetPath)
{
    // TODO: implement
}

nau::audio::AudioAssetPtr NauAudioEditor::findAsset(const std::string& path)
{
    auto assets = m_engine.assets();
    auto itAsset = eastl::find_if(assets.begin(), assets.end(), [path](nau::audio::AudioAssetPtr asset) -> bool {
        return std::filesystem::path(asset->name().c_str()) == std::filesystem::path(path);
    });
    return itAsset != assets.end() ? *itAsset : nullptr;
}

// TODO: use handleSourceAdded
bool NauAudioEditor::loadContainer(const std::string& name)
{
    const std::string path = nau::FileSystemExtensions::resolveToNativePathContentFolder(name);
    
    if (findAsset(path)) {
        return true;  // Already loaded
    }

    const auto stage = PXR_NS::UsdStage::Open(path);
    if (!stage) {
        NED_DEBUG("Invalid container prim path: {}", path);
        return false;
    }

    const auto primRaw = stage->GetPrimAtPath(pxr::SdfPath("/AudioContainer"));
    if (!primRaw.IsValid()) {
        NED_DEBUG("Invalid audio container prim at: {}", path);
        return false;
    }

    auto prim = UsdProxy::UsdProxyPrim(primRaw); 
    auto container = m_engine.createContainer(path.c_str());

    // Load container kind
    const auto containerKindProp = prim.getProperty(PXR_NS::TfToken("containerKind"));
    if (!containerKindProp) {
        NED_DEBUG("Invalid audio container prim kind property");
        return false;
    }

    PXR_NS::VtValue kindValue;
    if (!containerKindProp->getValue(&kindValue)) {
        NED_DEBUG("Failed to retrieve audio container kind");
        return false;
    }

    const auto kindToken = kindValue.Get<PXR_NS::TfToken>();
    if (const auto kind = nau::EnumTraits<nau::audio::AudioContainerKind>().parse(kindToken.data()); kind) {
        container->setKind(*kind);
    }

    // Load sources
    const auto sourcesProp = prim.getProperty(PXR_NS::TfToken("sources"));
    if (!sourcesProp) {
        NED_DEBUG("Invalid audio container prim sources property");
        return false;
    }

    PXR_NS::VtValue sourcesValue;
    if (!sourcesProp->getValue(&sourcesValue)) {
        NED_DEBUG("Failed to retrieve audio container sources");
        return false;
    }

    PXR_NS::VtArray<PXR_NS::SdfAssetPath> sourcePaths;
    sourcePaths = sourcesValue.Get<PXR_NS::VtArray<PXR_NS::SdfAssetPath>>();

    const auto registeredAssets = m_engine.assets();
    for (const auto sdfPath : sourcePaths) {
        const std::string path = sdfPath.GetAssetPath().c_str();
        auto asset = findAsset(path);             // The asset might no have been loaded yet
        const bool loaded = loadContainer(path);  // Try loading
        if (asset = findAsset(path); !loaded || !asset) {
            NED_WARNING("Failed to find {} among registered audio assets", path);
        } else {
            container->add(asset);
        }
    }

    return true;
}

void NauAudioEditor::saveContainer(nau::audio::AudioAssetContainerPtr container)
{
    const auto& path = container->name();
    const auto stage = PXR_NS::UsdStage::Open(path.c_str());
    if (!stage) {
        NED_DEBUG("Invalid container prim path: {}", path);
        return;
    }

    const auto primRaw = stage->GetPrimAtPath(pxr::SdfPath("/AudioContainer"));
    if (!primRaw.IsValid()) {
        NED_DEBUG("Invalid audio container prim at: {}", path);
        return;
    }

    auto prim = UsdProxy::UsdProxyPrim(primRaw);

    // Save container kind
    const auto containerKindProp = prim.getProperty(PXR_NS::TfToken("containerKind"));
    if (!containerKindProp) {
        NED_DEBUG("Invalid audio container prim kind property");
        return;
    }

    const auto kind = nau::EnumTraits<nau::audio::AudioContainerKind>().toString(container->kind());
    containerKindProp->setValue(PXR_NS::VtValue(PXR_NS::TfToken(std::string(kind).c_str())));

    // Save sources
    const auto sourcesProp = prim.getProperty(PXR_NS::TfToken("sources"));
    if (!sourcesProp) {
        NED_DEBUG("Invalid audio container prim sources property");
        return;
    }

    PXR_NS::VtArray<PXR_NS::SdfAssetPath> sourcePaths;
    for (auto& source : *container) {
        sourcePaths.emplace_back(source->name().c_str());
    }

    sourcesProp->setValue(PXR_NS::VtValue(sourcePaths));

    // Save
    stage->Save();
}
