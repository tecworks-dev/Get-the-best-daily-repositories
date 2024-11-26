// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_project_browser_icon_provider.hpp"
#include "nau/assets/nau_asset_preview.hpp"

// ** NauProjectBrowserIconProvider

NauProjectBrowserIconProvider::NauProjectBrowserIconProvider(
        std::vector<std::shared_ptr<NauProjectBrowserItemTypeResolverInterface>> itemTypeResolvers)
    : QAbstractFileIconProvider()
    , m_itemTypeResolvers(std::move(itemTypeResolvers))
    , m_defaultDirIcon(QIcon(":/UI/icons/browser/directory.png"))
    , m_defaultFileIcon(QIcon(":/UI/icons/browser/unknown.png"))
{
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

QIcon NauProjectBrowserIconProvider::icon(QAbstractFileIconProvider::IconType type) const
{
    if (type == QAbstractFileIconProvider::IconType::File) {
        return m_defaultFileIcon;
    }

    return m_defaultDirIcon;
}

QIcon NauProjectBrowserIconProvider::icon(const QFileInfo& info) const
{
    if (info.isDir()) {
        return m_defaultDirIcon;
    }

    if (NauAssetPreview::hasPreview(info.filePath())) {
        return NauAssetPreview::assetPreview(info.filePath());
    }

    NauEditorFileType type = NauEditorFileType::Unrecognized;
    for (const auto& typeResolver : m_itemTypeResolvers) {
        const NauEditorFileType candidateType = typeResolver->resolve(info.filePath()).type;
        if (candidateType != NauEditorFileType::Unrecognized) {
            type = candidateType;
            break;
        }
    }

    auto it = m_iconsRepository.find(type);
    if (it != m_iconsRepository.end()) {
        return it.value();
    }

    return info.isDir() ? m_defaultDirIcon : m_defaultFileIcon;

}
