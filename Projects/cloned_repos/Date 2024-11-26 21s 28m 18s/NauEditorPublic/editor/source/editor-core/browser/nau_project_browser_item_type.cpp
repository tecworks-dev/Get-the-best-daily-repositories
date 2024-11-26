// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_project_browser_item_type.hpp"
#include "nau_assert.hpp"
#include "nau_log.hpp"

#include <QApplication>


QString Nau::itemTypeToString(NauEditorFileType type)
{
     switch (type)
    {
        case NauEditorFileType::Unrecognized: return QApplication::translate("AssetItemType", "Unrecognized files");
        case NauEditorFileType::EngineCore: return QApplication::translate("AssetItemType", "Engine core");
        case NauEditorFileType::Project: return QApplication::translate("AssetItemType", "Editor project file");
        case NauEditorFileType::Config: return QApplication::translate("AssetItemType", "Configuration files");
        case NauEditorFileType::Texture: return QApplication::translate("AssetItemType", "Textures");
        case NauEditorFileType::Material: return QApplication::translate("AssetItemType", "Materials");
        case NauEditorFileType::Model: return QApplication::translate("AssetItemType", "Models(Meshes)");
        case NauEditorFileType::Shader: return QApplication::translate("AssetItemType", "Shader");
        case NauEditorFileType::Script: return QApplication::translate("AssetItemType", "Script");
        case NauEditorFileType::VirtualRomFS: return QApplication::translate("AssetItemType", "Virtual ROM FS");
        case NauEditorFileType::Scene: return QApplication::translate("AssetItemType", "Scene");
        case NauEditorFileType::Action: return QApplication::translate("AssetItemType", "Input Action");
        case NauEditorFileType::AudioContainer: return QApplication::translate("AssetItemType", "Audio Container");
        case NauEditorFileType::RawAudio: return QApplication::translate("AssetItemType", "Raw Audio");
        case NauEditorFileType::VFX: return QApplication::translate("AssetItemType", "Visual Effects");
        case NauEditorFileType::Animation: return QApplication::translate("AssetItemType", "Animation");
        case NauEditorFileType::UI: return QApplication::translate("AssetItemType", "UI");
        case NauEditorFileType::Font: return QApplication::translate("AssetItemType", "Font");
        case NauEditorFileType::PhysicsMaterial: return QApplication::translate("AssetItemType", "Physics Material");
        default:
            NED_ASSERT(!"Not implemented");
    }
    return {};
}
