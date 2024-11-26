// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Usd scene synchronizer definitions

#pragma once

#if !defined(NAU_EDITOR_STATIC_RUNTIME)

    #ifdef NAU_USD_SCENE_SYNCHRONIZER_BUILD_DLL
        #define NAU_USD_SCENE_SYNCHRONIZER_API __declspec(dllexport)
    #else
        #define NAU_USD_SCENE_SYNCHRONIZER_API __declspec(dllimport)
    #endif

#else
    #define NAU_USD_SCENE_SYNCHRONIZER_API
#endif