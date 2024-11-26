// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Viewport container widget

#pragma once

#include "nau_shortcut_hub.hpp"
#include "nau/viewport/nau_viewport_widget.hpp"


// ** NauViewportContainerWidget
//
// Widget container for viewport and viewport toolbar

class NAU_EDITOR_API NauViewportContainerWidget : public NauWidget
{
    Q_OBJECT

public:
    NauViewportContainerWidget(NauWidget* parent);
    void setViewport(NauViewportWidget* viewport, NauShortcutHub* shortcutHub = nullptr ,NauWidget* toolbar = nullptr);
    auto viewportHandle() const { return m_viewportWidget->windowHandle(); }

signals:
    // TODO: Move to scene editor widget
    void eventLevelLoaded();

    void eventFocusRequested();
    void eventDeleteRequested();
    void eventDuplicateRequested();
    void eventCopyRequested();
    void eventCutRequested();
    void eventPasteRequested();

private:
    NauWidget* m_toolbar;
    NauViewportWidget* m_viewportWidget;
};