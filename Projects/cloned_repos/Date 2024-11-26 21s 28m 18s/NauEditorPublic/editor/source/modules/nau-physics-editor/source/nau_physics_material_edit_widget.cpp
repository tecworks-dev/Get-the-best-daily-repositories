// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/physics/nau_physics_material_edit_widget.hpp"
#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_label.hpp"
#include "themes/nau_theme.hpp"

#include "usd_proxy/usd_proxy.h"
#include <pxr/base/tf/token.h>


// ** NauPhysicsEditorPageHeader

NauPhysicsEditorPageHeader::NauPhysicsEditorPageHeader(const QString& title, const QString& subtitle, NauWidget* parent)
    : NauWidget(parent)
{
    constexpr int WidgetHeight = 80;
    constexpr int IconSize = 48; 
    constexpr int Spacing = 16;
    constexpr int LabelHeight = 16;

    const NauAbstractTheme& theme = Nau::Theme::current();
    setFixedHeight(WidgetHeight);

    auto* layout = new NauLayoutGrid(this);
    layout->setSpacing(Spacing);

    auto* iconLabel = new NauLabel(this);
    iconLabel->setPixmap(Nau::Theme::current().iconPhysicsMaterialEditor().pixmap(IconSize));
    iconLabel->setFixedSize(IconSize, IconSize);

    auto titleLabel = new NauStaticTextLabel(title, this);
    titleLabel->setFixedHeight(LabelHeight);

    auto subtitleLabel = new NauStaticTextLabel(subtitle, this);
    subtitleLabel->setFixedHeight(LabelHeight);

    layout->addWidget(iconLabel, 0, 0, 2, 1);
    layout->addWidget(titleLabel, 0, 1, Qt::AlignLeft);
    layout->addWidget(subtitleLabel, 1, 1, Qt::AlignLeft);
    layout->addWidget(new NauSpacer(Qt::Horizontal, 1, this), 2, 0, 1, 2);
}


// ** NauPhysicsMaterialEditWidget

NauPhysicsMaterialEditWidget::NauPhysicsMaterialEditWidget(const std::string& assetPath,
    PXR_NS::UsdPrim prim, NauWidget* parent)
    : NauWidget(parent)
    , m_prim(prim)
{
    auto mainLayout = new NauLayoutVertical(this);
    auto layout = new NauLayoutGrid();
    layout->setSpacing(Spacing);
    mainLayout->setSpacing(Spacing);
    mainLayout->setContentsMargins(QMargins(16, 16, 16, 16));

    m_nameEditor = createTextEditor();
    m_frictionSpinBox = createSpinBox();
    m_restitutionSpinBox = createSpinBox();

    initValues();

    layout->addWidget(createLabel(tr("Name"), tr("Name of this physics material")), 0, 0);
    layout->addWidget(m_nameEditor, 0, 2);

    layout->addWidget(createLabel(tr("Friction"), tr("Force that presses the two bodies together")), 1, 0);
    layout->addWidget(m_frictionSpinBox, 1, 2);

    layout->addWidget(createLabel(tr("Restitution"), tr("Degree of elasticity of a collision response")), 2, 0);
    layout->addWidget(m_restitutionSpinBox, 2, 2);

    mainLayout->addWidget(new NauPhysicsEditorPageHeader(tr("Physics Material Editor"),
        QFileInfo(assetPath.c_str()).baseName(), this));

    mainLayout->addLayout(layout);
    mainLayout->addStretch(1);
}

NauLineEdit* NauPhysicsMaterialEditWidget::createTextEditor()
{
    auto editor = new NauLineEdit(this);
    editor->setFixedHeight(24);
    connect(editor, &NauLineEdit::editingFinished, this,  &NauPhysicsMaterialEditWidget::emitResultPrim);

    return editor;
}

QWidget* NauPhysicsMaterialEditWidget::createLabel(const QString& text, const QString& tooltip)
{
    auto label = new NauStaticTextLabel(text);
    label->setToolTip(tooltip);
    label->setFixedHeight(LabelHeight);
    label->setFont(Nau::Theme::current().fontObjectInspector());

    return label;
}

NauMultiValueDoubleSpinBox* NauPhysicsMaterialEditWidget::createSpinBox()
{
    auto spinBox = new NauMultiValueDoubleSpinBox(this, 1);
    spinBox->setMinimum(0.0);
    spinBox->setMaximum(1.0);
    connect(spinBox, &NauMultiValueDoubleSpinBox::eventValueChanged, this, &NauPhysicsMaterialEditWidget::emitResultPrim);

    return spinBox;
}

void NauPhysicsMaterialEditWidget::emitResultPrim()
{
    auto proxyPrim = UsdProxy::UsdProxyPrim(m_prim);

    auto frictionProperty = proxyPrim.getProperty("friction"_tftoken);
    auto restitutionProperty = proxyPrim.getProperty("restitution"_tftoken);
    auto nameProperty = proxyPrim.getProperty("materialName"_tftoken);

    frictionProperty->setValue(pxr::VtValue(static_cast<float>((*m_frictionSpinBox)[0]->value())));
    restitutionProperty->setValue(pxr::VtValue(static_cast<float>((*m_restitutionSpinBox)[0]->value())));
    nameProperty->setValue(pxr::VtValue(m_nameEditor->text().toStdString()));

    emit eventMaterialChanged(m_prim);
}

void NauPhysicsMaterialEditWidget::initValues()
{
    QSignalBlocker block1{m_nameEditor};
    QSignalBlocker block2{m_frictionSpinBox};
    QSignalBlocker block3{m_restitutionSpinBox};

    auto proxyPrim = UsdProxy::UsdProxyPrim(m_prim);
    pxr::VtValue value;

    auto frictionProperty = proxyPrim.getProperty("friction"_tftoken);
    frictionProperty->getValue(&value);
    (*m_frictionSpinBox)[0]->setValue(value.Get<float>());

    auto restitutionProperty = proxyPrim.getProperty("restitution"_tftoken);
    restitutionProperty->getValue(&value);
    (*m_restitutionSpinBox)[0]->setValue(value.Get<float>());

    auto nameProperty = proxyPrim.getProperty("materialName"_tftoken);
    nameProperty->getValue(&value);

    m_nameEditor->setText(QString::fromStdString(value.Get<std::string>()));
}
