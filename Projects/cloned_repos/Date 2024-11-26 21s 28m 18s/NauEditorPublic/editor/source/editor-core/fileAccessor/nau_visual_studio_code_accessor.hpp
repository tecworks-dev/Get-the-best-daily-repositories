// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Visual studio code accessor class

#pragma once

#include "fileAccessor/nau_file_accessor.hpp"


// ** NauVisualStudioCodeAccessor
// 
// Implements base file accessor interface to open file in Visual Studio Code

class NauVisualStudioCodeAccessor : public NauFileAccessorInterface
{
public:
    NauVisualStudioCodeAccessor() = default;

    virtual bool init() override;
    virtual bool openFile(const QString& path) override;

private:
    QString m_IDEPath;
};