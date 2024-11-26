// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor modes enums

#pragma once


// ** NauEditorMode

enum class NauEditorMode
{
    Editor, //Default editor mode without game logic
    Play, // Starts game in viewport
    Simulate // Editor mode + simulating some engine systems
};


// ** NauEditingMode

enum class NauEditingMode
{
    Select,
    Translate,
    Rotate,
    Scale
};