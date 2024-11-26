// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Supported editor file types enum

#pragma once


// ** NauEditorFileType
//
// Each file system's item belongs to particular group(type) and only one.

enum class NauEditorFileType
{
    Unrecognized = 0,
    EngineCore,
    Project,
    Config,
    Texture,
    Model,
    Shader,
    Script,
    VirtualRomFS,
    Scene,
    Material,
    Action,
    VFX,
    Animation,
    AudioContainer,
    RawAudio,
    UI,
    Font,
    PhysicsMaterial,
};
