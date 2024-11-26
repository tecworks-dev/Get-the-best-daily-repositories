// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Hash functors for various internal types.

#pragma once

#include "baseWidgets/nau_widget_utility.hpp"


namespace std{
    template <>
    struct hash<NauKeySequence>
    {
        size_t operator()(const NauKeySequence& sequ) const
        {
            return qHash(sequ, QHashSeed::globalSeed());
        }
    };
}