// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// File accessors interfaces

#pragma once

#include <QFileSystemWatcher>
#include <QString>


// ** NauFileAccessorInterface
//
// Editor file accessor interface. Contains open virtual function

class NauFileAccessorInterface
{
public:
    virtual ~NauFileAccessorInterface() noexcept = default;


    // Needed for delayed accessor construction.
    // Because some implementations may not be created due to the programs not being installed
    virtual bool init() = 0;

    // Function for opening a file in any editors that are implemented from this interface
    virtual bool openFile(const QString& path) = 0;
};
