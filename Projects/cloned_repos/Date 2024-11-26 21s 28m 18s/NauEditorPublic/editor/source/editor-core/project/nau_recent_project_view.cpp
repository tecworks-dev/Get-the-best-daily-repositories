// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_log.hpp"
#include "nau_recent_project_view.hpp"
#include "themes/nau_theme.hpp"
#include "themes/nau_default_theme.hpp"

#include <QMenu>
#include <QDesktopServices>
#include <QUrl>
#include <QPainterPath>


// ** NauRecentProjectView

NauRecentProjectView::NauRecentProjectView(const NauProjectInfo& projectInfo, NauWidget* parent)
    : NauWidget(parent)
    , m_projectInfo(projectInfo)
    , m_missing(!projectInfo.path.isValid())
{
    setFixedHeight(Height);
    setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, &NauRecentProjectView::customContextMenuRequested, this, &NauRecentProjectView::handleContextMenu);
}

void NauRecentProjectView::handleContextMenu(const QPoint& position)
{
    NED_DEBUG("Show project context menu");
    
    QMenu menu(this);
    if (!m_missing) {
        menu.addAction(tr("Open project"), this, [this] { emit eventClicked(m_projectInfo); });
        #if defined(Q_OS_WIN)
        menu.addAction(tr("Show in Explorer"), this, &NauRecentProjectView::handleReveal);
        #elif defined(Q_OS_MAC)
        menu.addAction(tr("Reveal in Finder"), this, &NauRecentProjectView::handleReveal);
        #endif
    }
    menu.addAction(tr("Remove from list"), this, &NauRecentProjectView::eventClear);
    menu.exec(this->mapToGlobal(position));
}

void NauRecentProjectView::handleReveal()
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(m_projectInfo.path.root().absolutePath()));
}

void NauRecentProjectView::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.setRenderHint(QPainter::TextAntialiasing);

    painter.fillRect(rect(), NauColor(19, 21, 22));

    // Highlight
    if (m_highlight) {
        QPainterPath path;
        path.addRoundedRect(rect(), 8, 8);
        painter.fillPath(path, NauColor(25, 29, 33));
    }

    // Draw the project icon
    // TODO: put back once we have scene thumbnails
    // const int iconOffset = (height() - 80) * 0.5;
    // painter.drawPixmap(iconOffset, iconOffset, 80, 80, QPixmap(":/UI/icons/browser/editor.png"));

    // Project title
    painter.setFont(Nau::Theme::current().fontTitleBarTitle());
    painter.setPen(!m_missing ? Qt::white : NauColor(138, 138, 138));
    const auto projectName = m_missing ? tr("Missing project") : m_projectInfo.name.value();
    const auto elidedprojectName = painter.fontMetrics().elidedText(projectName, Qt::ElideRight, width() - 120);   // Project name might be too long
    const auto titleY = 23;
    painter.drawText(12, titleY, elidedprojectName);

    // Project location
    painter.setFont(Nau::Theme::current().fontProjectManagerLocation());
    const auto elidedPath = painter.fontMetrics().elidedText(m_projectInfo.path.root().absolutePath(), Qt::ElideRight, width() - 120);   // Project path might be too long
    painter.setPen(!m_missing ? NauColor(138, 138, 138) : NauColor(78, 78, 78));
    painter.drawText(12, titleY + 22, elidedPath);

    // Project version
    if (!m_missing) {
        painter.setFont(Nau::Theme::current().fontTitleBarTitle());
        painter.setPen(Qt::white);
        painter.drawText(width() - 100, titleY, m_projectInfo.version->asQtString());
    }

    // Missing project overlay
    if (m_missing) {
        painter.fillRect(rect(), QColor(0, 0, 0, 64));
    }
}

void NauRecentProjectView::enterEvent([[maybe_unused]] QEnterEvent* event)
{
    m_highlight = true;
    m_missing = !m_projectInfo.path.isValid();
    update();
}

void NauRecentProjectView::leaveEvent([[maybe_unused]] QEvent* event)
{
    m_highlight = false;
    update();
}

void NauRecentProjectView::mouseDoubleClickEvent([[maybe_unused]] QMouseEvent* event)
{
    if (m_projectInfo.path.isValid()) {
        emit eventClicked(m_projectInfo);
    }
}
