// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Recent projects view

#pragma once

#include "nau_project.hpp"
#include "nau_widget.hpp"

#include <QObject>


// ** NauRecentProjectView

class NAU_EDITOR_API NauRecentProjectView : public NauWidget
{
    Q_OBJECT

public:
    NauRecentProjectView(const NauProjectInfo& projectInfo, NauWidget* parent);

protected:
    void paintEvent(QPaintEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

signals:
    void eventClicked(const NauProjectInfo& info);
    void eventClear();

private slots:
    void handleContextMenu(const QPoint& position);
    void handleReveal();

private:
    inline static constexpr int Height = 61;

private:
    const NauProjectInfo m_projectInfo;

    bool m_highlight = false;
    bool m_missing = false;
};
