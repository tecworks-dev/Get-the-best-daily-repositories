// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_static_text_label.hpp"
#include "baseWidgets/nau_spoiler.hpp"

#include <pxr/usd/usd/prim.h>


// ** NauPhysicsEditorPageHeader

class NauPhysicsEditorPageHeader : public NauWidget
{
public:
    NauPhysicsEditorPageHeader(const QString& title, const QString& subtitle, NauWidget* parent = nullptr);

};


// ** NauPhysicsMaterialEditWidget

class NauPhysicsMaterialEditWidget : public NauWidget
{
    Q_OBJECT

public:
    NauPhysicsMaterialEditWidget(const std::string& assetPath, PXR_NS::UsdPrim prim, NauWidget* parent = nullptr);

signals:
    void eventMaterialChanged(const PXR_NS::UsdPrim& prim);

private:
    NauLineEdit* createTextEditor();
    QWidget* createLabel(const QString& text, const QString& tooltip);
    NauMultiValueDoubleSpinBox* createSpinBox();
    void emitResultPrim();
    void initValues();

private:
    inline static constexpr int Spacing = 8;
    inline static constexpr int LabelHeight = 16;

private:
    PXR_NS::UsdPrim m_prim;
    NauPhysicsEditorPageHeader* m_header = nullptr;
    NauLineEdit* m_nameEditor = nullptr;
    NauMultiValueDoubleSpinBox* m_frictionSpinBox = nullptr;
    NauMultiValueDoubleSpinBox* m_restitutionSpinBox = nullptr;
};
