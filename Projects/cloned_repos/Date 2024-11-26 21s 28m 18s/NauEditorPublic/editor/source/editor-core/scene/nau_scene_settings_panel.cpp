// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "scene/nau_scene_settings_panel.hpp"
#include "inspector/nau_object_inspector.hpp"
#include "nau_settings.hpp"
#include "nau_widget_utility.hpp"
#include "nau/app/nau_editor_interface.hpp"

#include <QFileDialog>
#include <QDir>
#include <QMessageBox>


// ** NauSceneSettingsPanel

NauSceneSettingsPanel::NauSceneSettingsPanel(NauWidget* parent)
    : NauWidget(parent)
    , m_layout(new NauLayoutVertical(this))
{
}

void NauSceneSettingsPanel::loadSettings(NauSceneSettings* sceneSettings)
{
    m_layout->clear();

    // Add initial scripts path section
    // Buttons
    auto scriptButtonsLayout = new NauLayoutVertical();
    auto addScriptButton = new NauPushButton(this);
    addScriptButton->setText("+");
    addScriptButton->setObjectName("addInitialScriptButton");
    addScriptButton->setToolTip(tr("Add initial script"));
    addScriptButton->setMaximumSize(QSize(20, 20));
    auto deleteScriptButton = new NauPushButton(this);
    deleteScriptButton->setText("-");
    deleteScriptButton->setObjectName("removeInitialScriptButton");
    deleteScriptButton->setToolTip(tr("Remove initial script"));
    deleteScriptButton->setMaximumSize(QSize(20, 20));

    scriptButtonsLayout->addWidget(addScriptButton);
    scriptButtonsLayout->addWidget(deleteScriptButton);
    scriptButtonsLayout->addSpacerItem(new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding));

    // scripts list view
    m_initailScriptsProperty = new QListWidget(this);
    m_initailScriptsProperty->setWordWrap(true);
    m_initailScriptsProperty->setObjectName("initialScriptsListWidget");
    m_initailScriptsProperty->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_initailScriptsProperty->setMaximumHeight(70);
    
    QFont font;
    font.setPointSizeF(9);
    m_initailScriptsProperty->setFont(font);
    for (const auto& script : sceneSettings->initialScripts()) {
        m_initailScriptsProperty->addItem(QString(script.c_str()));
    }

    // label
    auto initialScriptsLabel = new NauLabel(tr("Initial scripts"), this);
    initialScriptsLabel->setObjectName("initialScriptsLabel");
    
    // layout
    auto initialScriptsLayout = new NauLayoutHorizontal();
    initialScriptsLayout->setSpacing(4);
    initialScriptsLayout->setContentsMargins(4, 4, 4, 4);
    initialScriptsLayout->setObjectName("initialScriptsLayout");
    initialScriptsLayout->setAlignment(Qt::AlignLeft);
    initialScriptsLayout->addWidget(initialScriptsLabel);
    initialScriptsLayout->addWidget(m_initailScriptsProperty);
    initialScriptsLayout->addLayout(scriptButtonsLayout);

    // Set buttons handle
    addScriptButton->connect(addScriptButton, &NauPushButton::clicked, [this, sceneSettings]() {
        auto relativePath = getRelativeFilePathFromFileExplorer();
        if (relativePath.isEmpty()) {
            return;
        }

        m_initailScriptsProperty->addItem(relativePath);
        sceneSettings->addInitialScript(relativePath.toUtf8().constData());
    });

    deleteScriptButton->connect(deleteScriptButton, &NauPushButton::clicked, [this, sceneSettings]() {
        if (auto curItem = m_initailScriptsProperty->currentItem()) {
            sceneSettings->removeInitialScript(curItem->text().toUtf8().constData());
            m_initailScriptsProperty->removeItemWidget(curItem);
            delete curItem;
        }
    });

    // Add to main layout
    m_layout->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    m_layout->addSpacing(12);
    m_layout->addLayout(initialScriptsLayout);
    m_layout->addSpacerItem(new QSpacerItem(10, 550, QSizePolicy::Minimum, QSizePolicy::Expanding));
    m_layout->setStretch(2, 1);
}

QString NauSceneSettingsPanel::getRelativeFilePathFromFileExplorer()
{
    const QString scriptsDirPath = NauEditorInterface::currentProject()->defaultScriptsFolder();
    QString fileName = QFileDialog::getOpenFileName(this, tr("Get file"), scriptsDirPath, tr("daScript (*.das)"));
    QDir scriptsDir(scriptsDirPath);
    return scriptsDir.relativeFilePath(fileName);
}