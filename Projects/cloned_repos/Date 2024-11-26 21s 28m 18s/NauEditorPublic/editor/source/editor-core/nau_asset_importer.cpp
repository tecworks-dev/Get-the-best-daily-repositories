// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_asset_importer.hpp"
#include "nau_log.hpp"
#include "inspector/nau_object_inspector.hpp"
#include "nau_content_creator.hpp"
#include "nau/app/nau_editor_interface.hpp"

#include <QFileDialog>
#include <QList>
#include <QSet>


static const QString modelAssetExtension = ".glb";
static const QSet<QString> supportedTextureFormats = { "jpg", "tga" };

// ** NauAssetImporter

void NauAssetImporter::proccessAssetImport(const QString& currentDir)
{
    QString saveDir = currentDir;
    QDir contentDir(NauEditorInterface::currentProject()->defaultContentFolder());
    if (contentDir.relativeFilePath(saveDir).startsWith("..")) {
        saveDir = NauEditorInterface::currentProject()->defaultContentFolder();
        NED_WARNING("Asset Import: Current folder is not a content folder, the file will be saved to {}", saveDir.toUtf8().constData());
    }

    // Get file to export from explorer
    const QString filePath = QFileDialog::getOpenFileName(nullptr,
        QCoreApplication::translate("Asset import", "Choose file to import"));

    if (filePath.isEmpty()) {
        return;
    }

    // Texture importers do not exist now
    QFileInfo fileInfo(filePath);
    if (supportedTextureFormats.contains(fileInfo.completeSuffix())) {
        //NauEngineAssetsAPI::importTextureAsset(saveDir, filePath);
        return;
    }

    proccessModelImport(saveDir, filePath);
}

void NauAssetImporter::proccessModelImport(const QString& currentDir, const QString& filePath)
{
    QFileInfo fileInfo(filePath);
    const QString extension = fileInfo.completeSuffix().toLower();
    const QString objectName = fileInfo.completeBaseName();

    //NauEngineProxyObjectPropertiesPtr proxyImportParams = NauEngineAssetsAPI::importParams(extension);
    //NauPropertiesContainer importParams = NauEditorSceneProxyImporter::importFromProxyProperties(*proxyImportParams);

    //Build dialog with settings
    NauDialog importDialog(nullptr);
    importDialog.setWindowTitle(QCoreApplication::translate("Asset import", "Asset import"));

    //fillDialogWithParameters(importDialog, importParams);

    if (importDialog.showModal() == NauDialog::Rejected) {
        return;
    }

    //NauEngineProxyObjectPropertiesPtr proxyImportChangedParams = NauEditorSceneProxyExporter::exportToProxyProperties(importParams);
    //NauEngineAssetsAPI::importModelAsset(currentDir, filePath, proxyImportChangedParams);
}

void NauAssetImporter::fillDialogWithParameters(NauDialog& importDlg, NauPropertiesContainer& importParams)
{
    auto layout = new NauLayoutVertical(&importDlg);
    layout->setContentsMargins(2, 2, 2, 2);

    const QString desc = QCoreApplication::translate("Asset import", 
        "Importing a model into an asset. A material attached to it will be generated too");

    layout->addWidget(new NauLabel(desc));
    layout->addWidget(new NauSplitter);
    QList<NauPropertyAbstract*> params;
    for (auto param = importParams.begin(), end = importParams.end(); param != end; ++param) {
        auto propertyWidget = NauPropertyFactory::createProperty(param.value().value());

        // TODO: Add property eventValueChanged signal, which returns the value
        QString paramName = param.key();
        connect(propertyWidget, &NauPropertyAbstract::eventValueChanged, &importDlg, [&, paramName, propertyWidget]() {
            importParams[paramName] = NauObjectProperty(paramName, propertyWidget->getValue(), "");
        });

        if (!propertyWidget) continue;
        auto boolWidget = dynamic_cast<NauPropertyBool*>(propertyWidget);
        // TODO: Move label var and setter to abstract widget
        if (boolWidget) {
            boolWidget->setLabel(param.key());
        }
        layout->addWidget(propertyWidget);
        propertyWidget->setValue(param.value().value());
        propertyWidget->setParent(&importDlg);

        params.push_back(propertyWidget);
    }

    NauDialogButtonBox* m_buttons = new NauDialogButtonBox(&importDlg);
    m_buttons->setStandardButtons(NauDialogButtonBox::Ok | NauDialogButtonBox::Cancel);

    auto applyParametrs = [&importDlg]() { importDlg.accept(); };

    m_buttons->connect(m_buttons, &QDialogButtonBox::accepted, applyParametrs);
    m_buttons->connect(m_buttons, &QDialogButtonBox::rejected, &importDlg, &NauDialog::reject);

    layout->addWidget(m_buttons);
}