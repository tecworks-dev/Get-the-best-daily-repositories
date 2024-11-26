// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_gui_outliner_client.hpp"


//TODO: Refine the api architecture of client-widget interaction in the future
// ** NauUsdOutlinerClient

NauGuiOutlinerClient::NauGuiOutlinerClient(NauWorldOutlinerWidget* outlinerWidget, NauWorldOutlineTableWidget& outlinerTab, const NauUsdSelectionContainerPtr& selectionContainer)
    : NauUsdOutlinerClient(outlinerWidget, outlinerTab, selectionContainer)
{

}
