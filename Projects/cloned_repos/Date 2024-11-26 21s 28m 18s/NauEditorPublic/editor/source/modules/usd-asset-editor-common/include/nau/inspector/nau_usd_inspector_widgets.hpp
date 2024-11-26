// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Widgets for usd properties

#pragma once

#include "nau/nau_usd_asset_editor_common_config.hpp"

#include "nau/assets/nau_asset_manager.hpp"
#include "nau/nau_editor_plugin_manager.hpp"

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_label.hpp"
#include "nau_view_attributes.hpp"

#include "pxr/base/vt/value.h"
#include "pxr/base/vt/array.h"

#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/gf/matrix4d.h"

#include "nau_resource_view_widget.hpp"


// At the resource visualization level,
// the visual container is filled with data for the selected resource type

#pragma region RESOURCE VIEW LEVEL

// ** NauInspectorUSDResourceComboBox

class NAU_USD_ASSET_EDITOR_COMMON_API NauInspectorUSDResourceComboBox : public NauResourceComboBox
{
    Q_OBJECT

public:

    enum ShowTypes
    {
        ShowFileNameMode,
        ShowPrimNameMode
    };

    explicit NauInspectorUSDResourceComboBox(ShowTypes type = ShowFileNameMode, NauWidget* parent = nullptr);

    void showPopup() override;

    [[deprecated("Replaced by setAssetTypes")]]
    void setAssetType(NauEditorFileType assetType);
    void setAssetTypes(std::set<NauEditorFileType> assetType);

    void updateItems();

private:
    std::set<NauEditorFileType> m_assetTypes;
    ShowTypes m_viewType;
};
#pragma endregion


// At the abstract level, the top-level interaction of the widget with the outside world is defined.
// There can be only one such class, and it is also used by the factory to spawn a specific widget
// based on the received data.

#pragma region ABSTRACT LEVEL

// ** NauUsdPropertyAbstract

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyAbstract : public NauWidget
{
    Q_OBJECT

public:

    NauUsdPropertyAbstract(NauWidget* parent = nullptr);

    template <typename T>
    void setValue(T value) {
        setValueInternal(PXR_NS::VtValue(value));
    }

    virtual PXR_NS::VtValue getValue() = 0;

    virtual void setWidgetEnabled(bool isEnabled) = 0;

    // TODO: virtual void update() = 0;

signals:
    void eventValueChanged();

protected:
    virtual void setValueInternal(const PXR_NS::VtValue& value) = 0;

};
#pragma endregion


// PropertyBase are essentially the basis of properties.
// They are essentially interfaces that define the basic visual representation of an arbitrary data type,
// that can be displayed in an inspector.
// All property widgets are built from a base.

#pragma region BASE LEVEL

// ** NauUsdSingleRowPropertyBase
//
// Base for properties to be displayed as a single line

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdSingleRowPropertyBase : public NauUsdPropertyAbstract
{
    Q_OBJECT

public:
    NauUsdSingleRowPropertyBase(const std::string& propertyTitle, NauWidget * parent = nullptr);

    void setWidgetEnabled(bool isEnabled) override;

    // TODO: virtual void setTitle(std::string& propertyTitle) = 0
    QString label()
    {
        return m_label->text();
    }

    void setLabel(const QString& label) { m_label->setText(label); }

protected:
    NauLayoutGrid* m_contentLayout;
    NauLayoutVertical* m_layout;
    NauLabel* m_label;

private:
    inline static constexpr int Height = 32;
};


// ** NauUsdMultiRowPropertyBase
//
// Base for properties to be displayed as multiple lines

class NauUsdMultiRowPropertyBase : public NauUsdPropertyAbstract
{
    Q_OBJECT

public:
    NauUsdMultiRowPropertyBase(NauWidget* parent = nullptr);

    void setWidgetEnabled(bool isEnabled) override;

protected:
    NauLayoutVertical* m_layout;
};


// ** NauUsdSingleTogglePropertyBase
//
// A base for properties to be displayed as a single line with Check Box

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdSingleTogglePropertyBase : public NauUsdSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauUsdSingleTogglePropertyBase(const std::string& propertyTitle, NauWidget* parent = nullptr);

    // TODO: virtual void setTitle(std::string& propertyTitle) = 0

protected:
    NauCheckBox* m_checkBox;
};


// ** NauUsdMultiValueSpinBoxPropertyBase
//
// A base for properties that should be displayed as a single line with an arbitrary number of spin box fields in it

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdMultiValueSpinBoxPropertyBase : public NauUsdSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauUsdMultiValueSpinBoxPropertyBase(const std::string& propertyTitle, QStringList valuesNames = QStringList(), NauWidget* parent = nullptr);

protected:
    std::unique_ptr<NauMultiValueSpinBox> m_multiLineEdit;

private:
    inline static constexpr int Height = 56;
};


// ** NauUsdMultiValueDoubleSpinBoxPropertyBase
//
// Base for properties to be displayed as a single string with an arbitrary number of spin box fields with a double value

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdMultiValueDoubleSpinBoxPropertyBase : public NauUsdSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauUsdMultiValueDoubleSpinBoxPropertyBase(const std::string& propertyTitle, QStringList valuesNames = QStringList(), NauWidget* parent = nullptr);

protected:
    std::unique_ptr<NauMultiValueDoubleSpinBox> m_multiLineEdit;

private:
    inline static constexpr int Height = 56;
};

#pragma endregion


// At the property level, we already define how to render a particular type.
// When one property is inherited, another can only be lured by:
// the method of receiving and passing data, the validator,
// and an element can also be added to the visual data container (but the visual container cannot change!).
// It is property widgets that are rarely created by the factory.

#pragma region PROPERTY LEVEL

// ** NauAssetProperty
//
// Resource widget for any assets ()

class NauAssetProperty: public NauUsdSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauAssetProperty(const std::string& propertyTitle = "Asset Property", const std::string& metaInfo = "", bool clearable = false, NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Asset Property", const std::string& metaInfo = "") { return new NauAssetProperty(propertyTitle, metaInfo); };

signals:
    void eventClearRequested();

private:
    NauEditorFileType m_assetType;
    NauInspectorUSDResourceComboBox* m_comboBox;

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;
};


// ** NauReferenceProperty
//
// Resource widget for meshes

class NauReferenceProperty : public NauUsdSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauReferenceProperty(const std::string& propertyTitle = "Reference Mesh Property", const std::string& metaInfo = "", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Reference Mesh Property", const std::string& metaInfo = "") { return new NauReferenceProperty(propertyTitle, metaInfo); };

private:
    NauEditorFileType m_assetType;
    NauInspectorUSDResourceComboBox* m_comboBox;

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;
};


// ** NauPropertyBool
//
// Visual display for Bool property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyBool : public NauUsdSingleTogglePropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyBool(const std::string& propertyTitle = "Bool Property", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Bool Property", const std::string& metaInfo = "") { return new NauUsdPropertyBool(propertyTitle); };

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;
};


// ** NauPropertyString
//
// Visual display for String property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyString : public NauUsdSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyString(const std::string& propertyTitle = "String Property", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "String Property", const std::string& metaInfo = "") { return new NauUsdPropertyString(propertyTitle); };

protected:
    NauLineEdit* m_lineEdit;

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;
};

// ** NauPropertyColor
//
// Visual display for Color property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyColor : public NauUsdSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyColor(const std::string& propertyTitle = "Color Property", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Color Property", const std::string& metaInfo = "") { return new NauUsdPropertyColor(propertyTitle); };

protected:
    NauPushButton* m_colorButton;
    NauColorDialog* m_colorDialog;

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;
};


// ** NauPropertyInt
//
// Visual display for Int property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyInt : public NauUsdMultiRowPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyInt(const std::string& propertyTitle = "Int Property", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Int Property", const std::string& metaInfo = "") { return new NauUsdPropertyInt(propertyTitle); };

    void setRange(int minimum, int maximum);

protected:
    NauLayoutGrid* m_contentLayout;

    std::vector<NauLabel*> m_label;
    std::vector<NauSpinBox*> m_spinBox;

    std::string m_titleText;
    int m_minValue = std::numeric_limits<int>::min();
    int m_maxValue = std::numeric_limits<int>::max();

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;

    void createRow(const std::string& titleText, int value, int row);
};


// ** NauPropertyFloat
//
// Visual display for Float property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyFloat : public NauUsdSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyFloat(const std::string& propertyTitle = "Float Property", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Float Property", const std::string& metaInfo = "") { return new NauUsdPropertyFloat(propertyTitle); };

    void setRange(float minimum, float maximum);
    void setSingleStep(float step);

protected:
    NauDoubleSpinBox* m_spinBox;

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;
};


// ** NauPropertyDouble
//
// Visual display for Double property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyDouble : public NauUsdSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyDouble(const std::string& propertyTitle = "Double Property", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Double Property", const std::string& metaInfo = "") { return new NauUsdPropertyDouble(propertyTitle); };

    void setRange(double minimum, double maximum);
    void setSingleStep(double step);

protected:
    NauDoubleSpinBox* m_spinBox;

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;
};


// ** NauPropertyInt2
//
// Visual display for Int2 property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyInt2 : public NauUsdMultiValueSpinBoxPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyInt2(const std::string& propertyTitle = "Int2 Property", const QString& xTitle = "X", const QString& yTitle = "Y", NauWidget* parent = nullptr);

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Int2 Property", const std::string& metaInfo = "") { return new NauUsdPropertyInt2(propertyTitle); };

    PXR_NS::VtValue getValue() override;

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;

protected:
    std::unique_ptr<NauSpinBox> m_x;
    std::unique_ptr<NauSpinBox> m_y;
};


// ** NauPropertyDouble2
//
// Visual display for Double2 property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyDouble2 : public NauUsdMultiValueDoubleSpinBoxPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyDouble2(const std::string& propertyTitle = "Double2 Property", const QString& xTitle = "X", const QString& yTitle = "Y", NauWidget* parent = nullptr);

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Double2 Property", const std::string& metaInfo = "") { return new NauUsdPropertyDouble2(propertyTitle); };

    PXR_NS::VtValue getValue() override;

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;

protected:
    std::unique_ptr<NauDoubleSpinBox> m_x;
    std::unique_ptr<NauDoubleSpinBox> m_y;
};


// TODO: Make all types that describe Points generic


// ** NauPropertyInt3
//
// Visual display for Int3 property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyInt3 : public NauUsdMultiValueSpinBoxPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyInt3(const std::string& propertyTitle = "Int3 Property", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Int3 Property", const std::string& metaInfo = "") { return new NauUsdPropertyInt3(propertyTitle); };

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;

protected:
    std::unique_ptr<NauSpinBox> m_x;
    std::unique_ptr<NauSpinBox> m_y;
    std::unique_ptr<NauSpinBox> m_z;
};


// ** NauPropertyDouble3
//
// Visual display for Double3 property

class NauUsdPropertyDouble3 : public NauUsdMultiValueDoubleSpinBoxPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyDouble3(const std::string& propertyTitle = "Double3 Property", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Double3 Property", const std::string& metaInfo = "") { return new NauUsdPropertyDouble3(propertyTitle); };

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;

protected:
    std::unique_ptr<NauDoubleSpinBox> m_x;
    std::unique_ptr<NauDoubleSpinBox> m_y;
    std::unique_ptr<NauDoubleSpinBox> m_z;
};


// ** NauPropertyInt4
//
// Visual display for Int4 property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyInt4 : public NauUsdMultiValueSpinBoxPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyInt4(const std::string& propertyTitle = "Int4 Property", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Int4 Property", const std::string& metaInfo = "") { return new NauUsdPropertyInt4(propertyTitle); };

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;

protected:
    std::unique_ptr<NauSpinBox> m_x;
    std::unique_ptr<NauSpinBox> m_y;
    std::unique_ptr<NauSpinBox> m_z;
    std::unique_ptr<NauSpinBox> m_w;
};


// ** NauPropertyDouble4
//
// Visual display for Double4 property

class NauUsdPropertyDouble4 : public NauUsdMultiValueDoubleSpinBoxPropertyBase
{
    Q_OBJECT

public:
    NauUsdPropertyDouble4(const std::string& propertyTitle = "Double4 Property", NauWidget* parent = nullptr);
    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Double4 Property", const std::string& metaInfo = "") { return new NauUsdPropertyDouble4(propertyTitle); };

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;

protected:
    std::unique_ptr<NauDoubleSpinBox> m_x;
    std::unique_ptr<NauDoubleSpinBox> m_y;
    std::unique_ptr<NauDoubleSpinBox> m_z;
    std::unique_ptr<NauDoubleSpinBox> m_w;
};


// ** NauPropertyMatrix
//
// Visual display for Matrix property

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyMatrix : public NauUsdMultiRowPropertyBase
{
    Q_OBJECT

public:
    // TODO: propertyTitle is now not used in any way and has been added for uniform work with properties
    // Fix it later
    NauUsdPropertyMatrix(const std::string& propertyTitle = "Matrix Property", NauWidget* parent = nullptr);

    PXR_NS::VtValue getValue() override;

    static NauUsdPropertyAbstract* Create(const std::string& propertyTitle = "Matrix Property", const std::string& metaInfo = "") { return new NauUsdPropertyMatrix(propertyTitle); };

protected:
    void setValueInternal(const PXR_NS::VtValue& value) override;

protected:
    std::unique_ptr<NauMultiValueDoubleSpinBox> m_position;
    std::unique_ptr<NauMultiValueDoubleSpinBox> m_rotation;
    std::unique_ptr<NauMultiValueDoubleSpinBox> m_scale;

private:
    inline static constexpr int OuterMargin = 6;
    inline static constexpr int VerticalSpacer = 14;
};

#pragma endregion
