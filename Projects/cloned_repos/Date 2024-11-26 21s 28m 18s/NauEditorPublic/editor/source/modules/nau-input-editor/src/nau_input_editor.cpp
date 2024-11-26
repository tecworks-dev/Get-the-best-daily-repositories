// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_input_editor.hpp"

#include "usd_proxy/usd_proxy.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/relationship.h>

#include "components/pages/nau_input_editor_page.hpp"


// ** NauInputEditor

NauInputEditor::NauInputEditor() noexcept
    : m_editorWidget(nullptr) 
{
    m_assetWatcher = std::make_unique<NauInputAssetWatcher>();
}

void NauInputEditor::initialize(NauEditorInterface* mainEditor)
{
    m_dockManager = mainEditor->mainWindow().dockManager();

    if (m_assetWatcher) {
        m_assetWatcher->loadFromFile("input_assets.json");
    }
}

void NauInputEditor::terminate()
{
    // TODO: Reset resources
}

void NauInputEditor::postInitialize()
{
    m_editorDockWidget = new NauDockWidget(QObject::tr("Input Editor"), nullptr);
    m_editorDockWidget->setObjectName("DockInputEditor");
    m_editorDockWidget->setMinimumSizeHintMode(ads::CDockWidget::MinimumSizeHintFromContent);
}

void NauInputEditor::preTerminate()
{
    // TODO: unbind ui from NauEditor window

    if (m_assetWatcher) {
        m_assetWatcher->saveToFile("input_assets.json");
    }
}

void NauInputEditor::createAsset(const std::string& assetPath)
{
    if (!assetPath.empty()) {
        auto stage = PXR_NS::UsdStage::CreateInMemory(assetPath);
        auto prim = stage->DefinePrim(PXR_NS::SdfPath("/Input"), PXR_NS::TfToken("InputAction"));
        stage->SetDefaultPrim(prim);
        stage->GetRootLayer()->Export(assetPath);
    }
}

bool NauInputEditor::openAsset(const std::string& assetPath)
{
    if (const auto* inspector = m_dockManager->inspector()) {
        m_stage = PXR_NS::UsdStage::Open(assetPath);
        
        if (m_assetWatcher) {
            m_assetWatcher->makeAssetCurrent(assetPath);
        }

        reset();

        QObject::connect(m_editorWidget, &NauInputEditorPage::eventProcessAsset, [=](bool isAdded)
            {
                if (!isAdded) {
                    if (m_assetWatcher) {
                        m_assetWatcher->addAsset(assetPath);
                        m_assetWatcher->makeAssetCurrent(assetPath);
                    }
                }
                else {
                    if (m_assetWatcher) {
                        m_assetWatcher->removeAsset(assetPath);
                    }
                }

                m_editorWidget->setIsAdded(m_assetWatcher->isAssetAdded(assetPath));
            });

        QObject::connect(m_editorWidget, &NauInputEditorPage::eventInputUpdateRequired, [=]()
            {
                if (m_assetWatcher) {
                    m_assetWatcher->updateAsset(assetPath);
                }
            });

        const std::string actionName = QFileInfo(assetPath.c_str()).baseName().toUtf8().constData();
        m_editorWidget->setName(actionName);
        m_editorWidget->setStage(m_stage);
        m_editorWidget->setIsAdded(m_assetWatcher->isAssetAdded(assetPath));

        m_dockManager->addDockWidgetTabToArea(m_editorDockWidget, inspector->dockAreaWidget());
        NED_TRACE("Input action asset {} opened in the input editor.", assetPath);
        return true;
    }

    if (m_editorDockWidget->dockAreaWidget() == nullptr) {
        NED_ERROR("Failed to open input editor in tab.");
    }
    return false;
}

bool NauInputEditor::saveAsset(const std::string& assetPath)
{
    if (m_stage == nullptr) {
        NED_ERROR("Input action asset is not open.");
        return false;
    }

    NED_TRACE("Input action asset saved to {}.", assetPath);
    return true;
}

std::string NauInputEditor::editorName() const
{
    return "Input Editor";
}

NauEditorFileType NauInputEditor::assetType() const
{
    return NauEditorFileType::Action;
}

void NauInputEditor::handleRemovedAction(const std::string& actionFilePath)
{
    if (m_stage == nullptr || m_editorWidget == nullptr) {
        return;
    }

    const std::string actionName = QFileInfo(actionFilePath.c_str()).baseName().toUtf8().constData();
    const std::string stageName = QFileInfo(m_stage->GetRootLayer()->GetRealPath().c_str()).baseName().toStdString();

    if (stageName != actionName) {
        return;
    }

    m_editorDockWidget->takeWidget();
    m_editorWidget->hide();
    m_editorWidget->deleteLater();
    m_editorWidget = nullptr;

    m_dockManager->removeDockWidget(m_editorDockWidget);
}

void NauInputEditor::reset()
{
    if (m_editorWidget != nullptr) {
        return;
    }

    m_editorWidget = new NauInputEditorPage(nullptr);
    m_editorWidget->setParent(m_editorDockWidget);
    m_editorDockWidget->setWidget(m_editorWidget);
}
