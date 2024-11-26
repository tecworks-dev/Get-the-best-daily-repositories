// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/assets/nau_usd_asset_processor.hpp"
#include "nau/module/module.h"
#include "nau/nau_editor_plugin_manager.hpp"
#include "nau/usd_meta_tools/usd_meta_manager.h"

namespace nau
{
    struct UsdAssetProcessorModule : public IModule
    {
        nau::string getModuleName() override
        {
            return "UsdAssetProcessorModule";
        }
        void initialize() override
        {
            nau::loadPlugins();
            Nau::EditorServiceProvider().addClass<NauUsdAssetProcessor>();
        }
        void deinitialize() override
        {
        }
        void postInit() override
        {
        }
    };
}  // namespace nau

IMPLEMENT_MODULE(nau::UsdAssetProcessorModule);
