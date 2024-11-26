// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_theme.hpp"
#include "nau_default_theme.hpp"

namespace Nau
{
    namespace Theme
    {
        NauAbstractTheme& current()
        {
            // TODO: At this moment we have only one theme. Implement theme repository.
            static NauDefaultTheme globalTheme;
            
            return globalTheme;
        }
    }
}
