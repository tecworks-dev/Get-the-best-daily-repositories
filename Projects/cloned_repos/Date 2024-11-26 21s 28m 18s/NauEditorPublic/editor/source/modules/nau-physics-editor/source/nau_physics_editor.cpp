// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/physics/nau_physics_editor.hpp"
#include "nau/nau_editor_plugin_manager.hpp"
#include "nau/physics/physics_material.h"

#include "usd_proxy/usd_proxy.h"
#include "usd_proxy/usd_property_proxy.h"
#include "usd_proxy/usd_prim_proxy.h"


// ** NauPhysicsEditor

void NauPhysicsEditor::initialize(NauEditorInterface* editorWindowAbstract)
{
    m_mainEditor = editorWindowAbstract;
    m_editorDockManger = m_mainEditor->mainWindow().dockManager();
}

void NauPhysicsEditor::postInitialize()
{
    m_editorDockWidget = new NauDockWidget(QObject::tr(editorName().data()), nullptr);
    m_editorDockWidget->setObjectName("DockPhysicsEditor");

    auto assetManager = m_mainEditor->assetManager();
    assetManager->addClient({NauEditorFileType::PhysicsMaterial}, this);
}

void NauPhysicsEditor::createAsset(const std::string& assetPath)
{
    if (!assetPath.empty()) {
        auto stage = PXR_NS::UsdStage::CreateInMemory(assetPath);
        auto prim = stage->DefinePrim(PXR_NS::SdfPath("/PhysicsMaterial"), PXR_NS::TfToken("PhysicsMaterial"));
        stage->SetDefaultPrim(prim);
        stage->GetRootLayer()->Export(assetPath);
    }
}

bool NauPhysicsEditor::openAsset(const std::string& assetPath)
{
    if (QFile::exists(QString::fromStdString(assetPath))) {
        if (const auto* inspector = m_editorDockManger->inspector()) {
            m_stage = PXR_NS::UsdStage::Open(assetPath);
            auto prim = m_stage->DefinePrim(PXR_NS::SdfPath("/PhysicsMaterial"), PXR_NS::TfToken("PhysicsMaterial"));

            m_editorWidget = new NauPhysicsMaterialEditWidget(assetPath, prim);
            m_editorWidget->setParent(m_editorDockWidget);
            m_editorDockWidget->setWidget(m_editorWidget);
            m_editorDockManger->addDockWidgetTabToArea(m_editorDockWidget, inspector->dockAreaWidget());
        
            QObject::connect(m_editorWidget, &NauPhysicsMaterialEditWidget::eventMaterialChanged, [this](const PXR_NS::UsdPrim& prim) {
                m_stage->Save();
            });

            return true;
        }
    }
    NED_ERROR("Can't create editor form physics material {}. "
        "Unable to find Inspector tab in docking system.", assetPath);

    return false;
}

bool NauPhysicsEditor::saveAsset(const std::string& assetPath)
{
    return true;
}

std::string NauPhysicsEditor::editorName() const
{
    return "PhysicsEditor";
}

NauEditorFileType NauPhysicsEditor::assetType() const
{
    return NauEditorFileType::PhysicsMaterial;
}

void NauPhysicsEditor::handleSourceAdded(const std::string& sourcePath)
{
}

void NauPhysicsEditor::handleSourceRemoved(const std::string& sourcePath)
{
    if (m_stage == nullptr || m_editorWidget == nullptr) {
        return;
    }

    const std::string name = QFileInfo(sourcePath.c_str()).baseName().toStdString();
    const std::string stageName = QFileInfo(m_stage->GetRootLayer()->GetRealPath().c_str()).baseName().toStdString();

    if (stageName != name) {
        return;
    }

    m_editorDockWidget->takeWidget();
    m_editorWidget->hide();
    m_editorWidget->deleteLater();
    m_editorWidget = nullptr;

    m_editorDockManger->removeDockWidget(m_editorDockWidget);
}
