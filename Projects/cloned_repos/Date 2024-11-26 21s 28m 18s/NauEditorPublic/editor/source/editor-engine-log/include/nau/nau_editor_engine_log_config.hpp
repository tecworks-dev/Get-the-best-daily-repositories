// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor-Engine log definitions

#pragma once

#if !defined(NAU_EDITOR_STATIC_RUNTIME)

    #ifdef NAU_EDITOR_ENGINE_LOG_BUILD_DLL
        #define NAU_EDITOR_ENGINE_LOG_API __declspec(dllexport)
    #else
        #define NAU_EDITOR_ENGINE_LOG_API __declspec(dllimport)
    #endif

#else
    #define NAU_EDITOR_ENGINE_LOG_API
#endif