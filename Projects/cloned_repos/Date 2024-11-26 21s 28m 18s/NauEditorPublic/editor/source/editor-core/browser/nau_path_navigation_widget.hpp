// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Represents a chain of labels (breadcrumbs) each of which is a directory of given path.

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_widget_utility.hpp"


// ** NauPathNavigationWidget

class NAU_EDITOR_API NauPathNavigationWidget : public NauWidget
{
    Q_OBJECT

public:
    explicit NauPathNavigationWidget(NauWidget* parent);

    // Rebuilds the navigation chain with given path.
    // path is relative. e.g. (dir1/dir2/dir3/dir4).
    void setNavigationChain(const NauDir& path);

signals:
    // Emitted when user requested to jump back on specified level of path.
    // requestedDir is relative path.
    void changeDirectoryRequested(const NauDir& requestedDir);

private:
    void emitChangeDirectoryRequest(int pathIdx);
    void clearNavigationChain();

private:
    QStringList m_pathParts;
};