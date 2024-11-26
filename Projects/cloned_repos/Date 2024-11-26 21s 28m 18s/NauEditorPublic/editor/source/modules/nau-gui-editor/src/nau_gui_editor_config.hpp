// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor definitions

#pragma once
#include <QtCore/QtGlobal>

#if !defined(NAU_EDITOR_STATIC_RUNTIME)

    #ifdef NAU_GUI_EDITOR_BUILD_DLL
        #define NAU_GUI_EDITOR_API Q_DECL_EXPORT
    #else
        #define NAU_GUI_EDITOR_API Q_DECL_IMPORT
    #endif

#else
    #define NAU_EDITOR_API
#endif
