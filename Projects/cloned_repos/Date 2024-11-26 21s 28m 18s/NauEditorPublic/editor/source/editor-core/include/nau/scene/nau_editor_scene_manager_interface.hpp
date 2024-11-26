// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor scene manager interface

#pragma once

#include <string>


// ** NauEditorSceneManagerInterface

class NauEditorSceneManagerInterface
{
public:
    virtual bool createSceneFile(const std::string& path) = 0;
    virtual bool loadScene(const std::string& path) = 0;
    virtual bool unloadCurrentScene() = 0;

    virtual bool saveCurrentScene() = 0;
    virtual bool isCurrentSceneSaved() = 0;

    virtual std::string currentScenePath() const = 0;
};
