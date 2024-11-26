// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Interface for asset editors implementation

#pragma once

#include "nau/assets/nau_file_types.hpp"

#include "nau/rtti/rtti_object.h"
#include "nau/app/nau_editor_interface.hpp"


// ** NauAssetEditorInterface
//
// Interface for asset editors

class NAU_ABSTRACT_TYPE NauAssetEditorInterface : virtual public nau::IRefCounted
{
    NAU_INTERFACE(NauAssetEditorInterface, nau::IRefCounted)

public:
    NauAssetEditorInterface() = default;
    ~NauAssetEditorInterface() = default;

    virtual void initialize(NauEditorInterface* mainEditor) = 0;
    virtual void terminate() = 0;

    // Initialize function calls in other thread, we cannot create windows not in the main thread
    // So we have a postInitialize function that is called on the main thread
    virtual void postInitialize() = 0;

    // Need to unbind module UI from main editor window before it closes.
    // Otherwise, it will be removed from the widget tree of the main window
    virtual void preTerminate() = 0;

    virtual void createAsset(const std::string& assetPath) = 0;
    virtual bool openAsset(const std::string& assetPath) = 0;
    virtual bool saveAsset(const std::string& assetPath) = 0;

    [[nodiscard]] virtual std::string editorName() const = 0;
    [[nodiscard]] virtual NauEditorFileType assetType() const = 0;

    // Temp solution for UI editor
    virtual void startPlay() {};
    virtual void stopPlay() {};

};
