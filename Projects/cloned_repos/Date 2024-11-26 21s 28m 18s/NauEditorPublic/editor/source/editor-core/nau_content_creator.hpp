// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Content creation functions & add content widgets

#pragma once

#include "nau_widget.hpp"

#include <QString>
#include <string>


// ** NauAddContentMenu
//
// Context menu with create content functions

class NauAddContentMenu : public NauMenu
{
    Q_OBJECT
public:
    explicit NauAddContentMenu(const QString&  currentDir, NauWidget* parent = nullptr);

    // TO-DO: create separate class for creation funcs
    static void createMaterial(const QString& path);
    static void createMaterialByTemplate(const QString& newMaterialName, const QString& path, const QString& name, const std::filesystem::path& materialTemplatePath);
    static void createInputAction(const QString& path, const QString& name);
    static void createAnimation(const QString& path, const QString& name);
    static void createAudioContainer(const QString& path, const QString& name);
    static void createUI(const QString& path, const QString& name);
    static void createVFX(const QString& path, const QString& name);

private:
    static void createAsset(const QString& path, const QString& assetName, const QString& editorName);
};