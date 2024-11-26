// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Asset import dialog

#pragma once

#include "nau_widget.hpp"
#include "scene/nau_world.hpp"

#include "QString"

class DataBlock;


// ** NauAssetImporter
//
// Asset importer editor pipeline

class NauAssetImporter : public QObject
{
    Q_OBJECT

public:
    static void proccessAssetImport(const QString& currentDir);

private:

    static void proccessModelImport(const QString& currentDir, const QString& filePath);
    static void fillDialogWithParameters(NauDialog& dialog, NauPropertiesContainer& importParams);
};
