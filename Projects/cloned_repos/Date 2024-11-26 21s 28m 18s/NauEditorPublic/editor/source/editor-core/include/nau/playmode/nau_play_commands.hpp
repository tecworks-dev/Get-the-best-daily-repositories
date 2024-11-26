// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// UI commands for play mode

#pragma once

#include "nau/nau_editor_config.hpp"


// ** NauPlayCommands
// This is commands for ui. TODO: Make an template interface for UI commands and their registration

class NAU_EDITOR_API NauPlayCommands
{
public:
    NauPlayCommands() = delete;

    static void startPlay();
    static void pausePlay(const bool isPaused);
    static void stopPlay();
};