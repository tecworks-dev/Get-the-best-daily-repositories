// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// The default implementation of item type resolver.

#pragma once

#include "nau_project_browser_item_type.hpp"

#include <QCoreApplication>


// ** NauProjectBrowserItemTypeResolver

class NauProjectBrowserItemTypeResolver : public NauProjectBrowserItemTypeResolverInterface
{
    Q_DECLARE_TR_FUNCTIONS(NauProjectBrowserItemTypeResolver)
public:

    virtual ItemTypeDescription resolve(const QString& completeExtension, const std::optional<std::string> primPath = std::nullopt) const override;
};
