// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_about.hpp"
#include "nau_editor_version.hpp"
#include "nau/app/nau_qt_app.hpp"

#include <QClipboard>
#include <QDate>
#include <QLabel>


// ** NauAboutDialog

NauAboutDialog::NauAboutDialog(NauMainWindow* parent)
    : NauDialog(parent)
{
    setWindowTitle(tr("About %1").arg(NauApp::name()));
    auto layout = new NauLayoutVertical(this);
    layout->addWidget(new QLabel(NauApp::name()));
    layout->addWidget(new QLabel(tr("Copyright %1 %2 LLC").arg(QDate::currentDate().year()).arg("N-GINN")));
    
    // Version section
    auto versionWidget = new NauWidget(this);
    auto versionLayout = new NauLayoutHorizontal(versionWidget);
    layout->addWidget(versionWidget);
    const auto versionText = QString("%1 (%2)").arg(QApplication::applicationVersion()).arg(NAU_COMMIT_HASH);
    auto versionCopyButton = new QPushButton(tr("Copy"));
    versionLayout->addWidget(new QLabel(tr("Version %1").arg(versionText)));
    versionLayout->addSpacing(8);
    versionLayout->addWidget(versionCopyButton);
    connect(versionCopyButton, &QPushButton::clicked, [versionText] {
        QApplication::clipboard()->setText(versionText);
    });
}
