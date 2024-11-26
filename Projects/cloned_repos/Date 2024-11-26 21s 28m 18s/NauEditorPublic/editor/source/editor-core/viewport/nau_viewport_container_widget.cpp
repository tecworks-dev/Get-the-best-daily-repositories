// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "viewport/nau_viewport_container_widget.hpp"
#include "nau_log.hpp"

#include <QShortcut>


// ** NauViewportContainerWidget

NauViewportContainerWidget::NauViewportContainerWidget(NauWidget* parent)
    : NauWidget(parent)
    , m_viewportWidget(nullptr)
    , m_toolbar(nullptr)
{
}

void NauViewportContainerWidget::setViewport(NauViewportWidget* viewport, NauShortcutHub* shortcutHub,  NauWidget* toolbar)
{
    if (m_viewportWidget) {
        NED_WARNING("Viewport already set");
        return;
    }

    if (shortcutHub) {
        shortcutHub->addWidgetShortcut(NauShortcutOperation::ViewportFocus, *viewport,
            std::bind(&NauViewportContainerWidget::eventFocusRequested, this));

        shortcutHub->addWidgetShortcut(NauShortcutOperation::ViewportDuplicate, *viewport,
            std::bind(&NauViewportContainerWidget::eventDuplicateRequested, this));

        shortcutHub->addWidgetShortcut(NauShortcutOperation::ViewportCopy, *viewport,
            std::bind(&NauViewportContainerWidget::eventCopyRequested, this));

        shortcutHub->addWidgetShortcut(NauShortcutOperation::ViewportCut, *viewport,
            std::bind(&NauViewportContainerWidget::eventCutRequested, this));

        shortcutHub->addWidgetShortcut(NauShortcutOperation::ViewportPaste, *viewport,
            std::bind(&NauViewportContainerWidget::eventPasteRequested, this));

        shortcutHub->addWidgetShortcut(NauShortcutOperation::ViewportDelete, *viewport,
            std::bind(&NauViewportContainerWidget::eventDeleteRequested, this));
    }

    setFocusPolicy(Qt::StrongFocus);
    m_viewportWidget = viewport;
    m_viewportWidget->setParent(this);

    if (toolbar) {
        m_toolbar = toolbar;
        m_toolbar->setParent(this);
    }

    NauLayoutVertical* layout = new NauLayoutVertical;
    layout->setContentsMargins(0, 0, 0, 0);
    
    if (m_toolbar) {
        layout->addWidget(m_toolbar);
    }

    layout->addWidget(m_viewportWidget);
    setLayout(layout);
}