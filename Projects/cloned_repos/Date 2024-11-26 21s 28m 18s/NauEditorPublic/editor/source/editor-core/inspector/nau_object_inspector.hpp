// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Set of panels and widgets that allow to inspect and modify game objects.

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_spoiler.hpp"
#include "scene/nau_world.hpp"
#include "baseWidgets/nau_label.hpp"
#include "nau/nau_constants.hpp"
#include "nau_view_attributes.hpp"
#include "nau_resource_view_widget.hpp"
#include "baseWidgets/nau_slider_value.hpp"

#include <concepts>
#include <type_traits>

#include <QTimer>


// At the resource visualization level,
// the visual container is filled with data for the selected resource type

#pragma region RESOURCE VIEW LEVEL

// ** NauResourceComboBox

class NAU_EDITOR_API NauInspectorResourceComboBox : public NauResourceComboBox
{
    Q_OBJECT

public:
    explicit NauInspectorResourceComboBox(NauStringViewAttributeType type, NauWidget* parent = nullptr);

    void showPopup() override;

    void updateItems();

private:
    const NauStringViewAttributeType m_type;
};
#pragma endregion


// TODO: Made class for view attributes
using NauAttributesMap = QHash<QString, QString>;
using NauViewAttributes = QHash<QString, NauAttributesMap>;

// At the abstract level, the top-level interaction of the widget with the outside world is defined.
// There can be only one such class, and it is also used by the factory to spawn a specific widget
// based on the received data.

#pragma region ABSTRACT LEVEL

// ** NauPropertyAbstract

class NAU_EDITOR_API NauPropertyAbstract : public NauWidget
{
    Q_OBJECT

public:

    NauPropertyAbstract(NauWidget* parent = nullptr);

    template <typename T>
    void setValue(T value) {
        setValueInternal(NauAnyType(value));
    }

    virtual NauAnyType getValue() = 0;

    virtual void setWidgetEnabled(bool isEnabled) = 0;

    // TODO: virtual void update() = 0;

signals:
    void eventValueChanged();

protected:
    virtual void setValueInternal(const NauAnyType& value) = 0;

};
#pragma endregion


// PropertyBase are essentially the basis of properties.
// They are essentially interfaces that define the basic visual representation of an arbitrary data type,
// that can be displayed in an inspector.
// All property widgets are built from a base.

#pragma region BASE LEVEL

// ** NauSingleRowPropertyBase
//
// Base for properties to be displayed as a single line

class NAU_EDITOR_API NauSingleRowPropertyBase : public NauPropertyAbstract
{
    Q_OBJECT

public:
    NauSingleRowPropertyBase(const std::string& propertyTitle, NauWidget * parent = nullptr);

    virtual void setWidgetEnabled(bool isEnabled) override;

    // TODO: virtual void setTitle(std::string& propertyTitle) = 0
    QString label()
    {
        return m_label ? m_label->text() : QString();
    }

    void setLabel(const QString& label) { m_label->setText(label); }

protected:
    NauLayoutGrid* m_contentLayout;
    NauLayoutVertical* m_layout;
    NauLabel* m_label;

private:
    inline static constexpr int Height = 32;
};


// ** NauSingleTogglePropertyBase
//
// A base for properties to be displayed as a single line with Check Box

class NAU_EDITOR_API NauSingleTogglePropertyBase : public NauSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauSingleTogglePropertyBase(const std::string& propertyTitle, NauWidget* parent = nullptr);

    // TODO: virtual void setTitle(std::string& propertyTitle) = 0

protected:
    NauCheckBox* m_checkBox;
};


// ** NauMultiValueSpinBoxPropertyBase
//
// A base for properties that should be displayed as a single line with an arbitrary number of spin box fields in it

class NAU_EDITOR_API NauMultiValueSpinBoxPropertyBase : public NauSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauMultiValueSpinBoxPropertyBase(const std::string& propertyTitle, QStringList valuesNames = QStringList(), NauWidget* parent = nullptr);

    // TODO: virtual void setTitle(std::string& propertyTitle) = 0

protected:
    std::unique_ptr<NauMultiValueDoubleSpinBox> m_multiLineEdit;

private:
    inline static constexpr int Height = 56;
};


// ** NauMultiRowPropertyBase
//
// Base for properties to be displayed as multiple lines

class NAU_EDITOR_API NauMultiRowPropertyBase : public NauPropertyAbstract
{
    Q_OBJECT

public:
    NauMultiRowPropertyBase(NauWidget* parent = nullptr);

    virtual void setWidgetEnabled(bool isEnabled) override;

protected:
    NauLayoutVertical* m_layout;
};
#pragma endregion


// At the visualization level,
// a specific visual container is selected for the base type already known 
// by this moment for which the widget is designed.

#pragma region VIEWS LEVEL

// ** NauStringAbstractView
//
// An abstract class for all visual containers that contain a string at their base 

class NAU_EDITOR_API NauStringAbstractView : public NauWidget
{
    Q_OBJECT

public:
    NauStringAbstractView(NauWidget* parent = nullptr);

signals:
    // Throws data change events in the visual container further up the parent widget chain
    void eventValueChanged();

public:
    // Retrieves data from a visual container.
    // Usually used in a call: eventValueChanged()
    virtual QString getValue() const = 0;

    // Sets the value if it has been changed by something outside the parent widget
    // The system guarantees error-free data conversion and writing to the visual container.
    // But if something went wrong here, it means there is a serious bug in the program.
    // At the user's discretion can cause: eventValueChanged()
    virtual void setValue(const QString& value) = 0;
};


// ** NauStringLineView
//
// A visual container for representing a string in a single line

class NAU_EDITOR_API NauStringLineView : public NauStringAbstractView
{
    Q_OBJECT

public:
    NauStringLineView(NauWidget* parent = nullptr);

public:
    virtual QString getValue() const override;
    virtual void setValue(const QString& value) override;

private:
    NauLineEdit* m_lineEdit;
};



// ** NauStringResourceSelectorView
//
// A visual container for a string that is a soft link to some resource
 
class NauInspectorResourceComboBox;

class NAU_EDITOR_API NauStringResourceSelectorView : public NauStringAbstractView
{
    Q_OBJECT

public:
    explicit NauStringResourceSelectorView(const NauStringViewAttributeType type, NauWidget* parent = nullptr);

public:
    virtual QString getValue() const override;
    virtual void setValue(const QString& value) override;

    [[nodiscard]] NauAbstractComboBox* getComboBox() const noexcept;

    void updateItems();
    void clearItems();

private:
    NauInspectorResourceComboBox* m_comboBox;
    const NauStringViewAttributeType m_type;
};


// ** NauStringViewsFabric
//
// A factory for visual containers that contain a string at their base

class NAU_EDITOR_API NauStringViewsFactory
{
public:
    static NauStringAbstractView* createStringView(NauStringViewAttributeType type);
};

#pragma endregion


// At the property level, we already define how to render a particular type.
// When one property is inherited, another can only be lured by:
// the method of receiving and passing data, the validator,
// and an element can also be added to the visual data container (but the visual container cannot change!).
// It is property widgets that are rarely created by the factory.

#pragma region PROPERTY LEVEL

// ** NauPropertyBool
//
// Visual display for Bool property

class NAU_EDITOR_API NauPropertyBool : public NauSingleTogglePropertyBase
{
    Q_OBJECT

public:
    NauPropertyBool(const std::string& propertyTitle = "Bool Property", NauWidget* parent = nullptr);
    virtual NauAnyType getValue() override;

protected:
    virtual void setValueInternal(const NauAnyType& value) override;
};


// ** NauPropertyString
//
// Visual display for String property

class NAU_EDITOR_API NauPropertyString : public NauSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauPropertyString(NauStringViewAttributeType type, const std::string& propertyTitle = "String Property", NauWidget* parent = nullptr);
    virtual NauAnyType getValue() override;

protected:
    NauStringAbstractView* m_view;

protected:
    virtual void setValueInternal(const NauAnyType& value) override;
};

// ** NauPropertyColor
//
// Visual display for Color property

class NAU_EDITOR_API NauPropertyColor : public NauSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauPropertyColor(const std::string& propertyTitle = "Color Property", NauWidget* parent = nullptr);
    virtual NauAnyType getValue() override;

protected:
    NauPushButton* m_colorButton;
    NauColorDialog* m_colorDialog;

protected:
    virtual void setValueInternal(const NauAnyType& value) override;
};


// ** NauPropertyInt
//
// Visual display for Int property

class NAU_EDITOR_API NauPropertyInt : public NauSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauPropertyInt(const std::string& propertyTitle = "Int Property", NauWidget* parent = nullptr);
    virtual NauAnyType getValue() override;

    void setRange(int minimum, int maximum);

protected:
    NauSpinBox* m_spinBox;

protected:
    virtual void setValueInternal(const NauAnyType& value) override;
};


// ** NauPropertyReal
//
// Visual display for Real property

class NAU_EDITOR_API NauPropertyReal : public NauSingleRowPropertyBase
{
    Q_OBJECT

public:
    NauPropertyReal(const std::string& propertyTitle = "Real Property", NauWidget* parent = nullptr);
    virtual NauAnyType getValue() override;

    void setRange(float minimum, float maximum);
    void setSingleStep(float step);

protected:
    NauDoubleSpinBox* m_spinBox;

protected:
    virtual void setValueInternal(const NauAnyType& value) override;
};


// ** NauPropertyPoint2
//
// Visual display for Point2 property

class NAU_EDITOR_API NauPropertyPoint2 : public NauMultiValueSpinBoxPropertyBase
{
    Q_OBJECT

public:
    NauPropertyPoint2(const std::string& propertyTitle = "Point2 Property", const QString& xTitle = "X", const QString& yTitle = "Y", NauWidget* parent = nullptr);

    virtual NauAnyType getValue() override;

protected:
    virtual void setValueInternal(const NauAnyType& value) override;

protected:
    std::unique_ptr<NauDoubleSpinBox> m_x;
    std::unique_ptr<NauDoubleSpinBox> m_y;
};


// TODO: Make all types that describe Points generic

// ** NauPropertyPoint3
//
// Visual display for Point3 property

class NAU_EDITOR_API NauPropertyPoint3 : public NauMultiValueSpinBoxPropertyBase
{
    Q_OBJECT

public:
    NauPropertyPoint3(const std::string& propertyTitle = "Point3 Property", NauWidget* parent = nullptr);
    virtual NauAnyType getValue() override;

protected:
    virtual void setValueInternal(const NauAnyType& value) override;

protected:
    std::unique_ptr<NauDoubleSpinBox> m_x;
    std::unique_ptr<NauDoubleSpinBox> m_y;
    std::unique_ptr<NauDoubleSpinBox> m_z;
};


// ** NauPropertyPoint4
//
// Visual display for Point4 property

class NAU_EDITOR_API NauPropertyPoint4 : public NauMultiValueSpinBoxPropertyBase
{
    Q_OBJECT

public:
    NauPropertyPoint4(const std::string& propertyTitle = "Point4 Property", NauWidget* parent = nullptr);
    virtual NauAnyType getValue() override;

protected:
    virtual void setValueInternal(const NauAnyType& value) override;

protected:
    std::unique_ptr<NauDoubleSpinBox> m_x;
    std::unique_ptr<NauDoubleSpinBox> m_y;
    std::unique_ptr<NauDoubleSpinBox> m_z;
    std::unique_ptr<NauDoubleSpinBox> m_w;
};


// ** NauPropertyMatrix
//
// Visual display for Matrix property

class NAU_EDITOR_API NauPropertyMatrix : public NauMultiRowPropertyBase
{
    Q_OBJECT

public:
    // TODO: propertyTitle is now not used in any way and has been added for uniform work with properties
    // Fix it later
    NauPropertyMatrix(const std::string& propertyTitle = "Matrix Property", NauWidget* parent = nullptr);

    virtual NauAnyType getValue() override;

protected:
    virtual void setValueInternal(const NauAnyType& value) override;

protected:
    std::unique_ptr<NauMultiValueDoubleSpinBox> m_position;
    std::unique_ptr<NauMultiValueDoubleSpinBox> m_rotation;
    std::unique_ptr<NauMultiValueDoubleSpinBox> m_scale;

private:
    inline static constexpr int OuterMargin = 6;
    inline static constexpr int VerticalSpacer = 14;
};


// ** NauPropertyRangedInt
//
// Visual display for int property with slider

class NAU_EDITOR_API NauPropertyRangedInt : public NauSingleRowPropertyBase
{
    Q_OBJECT

public:
    // TODO: propertyTitle is now not used in any way and has been added for uniform work with properties
    // Fix it later
    explicit NauPropertyRangedInt(const std::string& propertyTitle = "Ranged Int Property", NauWidget* parent = nullptr);

    NauAnyType getValue() override;

protected:
    void setValueInternal(const NauAnyType& value) override;

protected:
    std::unique_ptr<NauSliderIntValue> m_slider;
};


// ** NauPropertyRangedFloat
//
// Visual display for float property with slider

class NAU_EDITOR_API NauPropertyRangedFloat : public NauSingleRowPropertyBase
{
    Q_OBJECT

public:
    // TODO: propertyTitle is now not used in any way and has been added for uniform work with properties
    // Fix it later
    explicit NauPropertyRangedFloat(const std::string& propertyTitle = "Ranged Float Property", NauWidget* parent = nullptr);

    NauAnyType getValue() override;

protected:
    void setValueInternal(const NauAnyType& value) override;

protected:
    std::unique_ptr<NauSliderFloatValue> m_slider;
};


// ** NauPropertyRangedInt2
//
// Visual display for pair float property with slider

class NAU_EDITOR_API NauPropertyRangedInt2 : public NauSingleRowPropertyBase
{
    Q_OBJECT

public:
    // TODO: propertyTitle is now not used in any way and has been added for uniform work with properties
    // Fix it later
    explicit NauPropertyRangedInt2(const std::string& propertyTitle = "Ranged Int Pair Property", NauWidget* parent = nullptr);

    NauAnyType getValue() override;

protected:
    void setValueInternal(const NauAnyType& value) override;

protected:
    std::unique_ptr<NauSliderIntPair> m_slider;
};


// ** NauPropertyRangedFloat2
//
// Visual display for pair float property with slider

class NAU_EDITOR_API NauPropertyRangedFloat2 : public NauSingleRowPropertyBase
{
    Q_OBJECT

public:
    // TODO: propertyTitle is now not used in any way and has been added for uniform work with properties
    // Fix it later
    explicit NauPropertyRangedFloat2(const std::string& propertyTitle = "Ranged Float Pair Property", NauWidget* parent = nullptr);

    NauAnyType getValue() override;

protected:
    void setValueInternal(const NauAnyType& value) override;

protected:
    std::unique_ptr<NauSliderFloatPair> m_slider;
};

// TODO: Add property widgets for ecs::Array, ecs::Object, ecs::EntityId, ecs::Tag, and etc
#pragma endregion

// The factory creates a visual representation corresponding to the passed type

#pragma region FACTORY

// ** NauPropertyFactory

class NAU_EDITOR_API NauPropertyFactory
{
    using NauTypeMap = std::tuple
        <
        std::pair<bool, NauPropertyBool>,
        std::pair<QString, NauPropertyString>,
        std::pair<int, NauPropertyInt>,
        std::pair<float, NauPropertyReal>,
        std::pair<QMatrix4x3, NauPropertyMatrix>,
        std::pair<QVector2D, NauPropertyPoint2>,
        std::pair<QVector3D, NauPropertyPoint3>,
        std::pair<QVector4D, NauPropertyPoint4>,
        std::pair<QColor, NauPropertyColor>,
        std::pair<NauRangedValue<int>, NauPropertyRangedInt>,
        std::pair<NauRangedValue<float>, NauPropertyRangedFloat>,
        std::pair<NauRangedPair<int>, NauPropertyRangedInt2>,
        std::pair<NauRangedPair<float>, NauPropertyRangedFloat2>
        >;

public:
    template <typename Concrete, typename... Ts>
    static std::unique_ptr<Concrete> constructArgs(Ts&&... params)
    {
        if constexpr (std::is_constructible_v<Concrete, Ts...>) {
            return std::make_unique<Concrete>(std::forward<Ts>(params)...);
        } else {
            return nullptr;
        }
    }

    // If no matching pair is found, this overload will be called
    template<size_t baseTypeSize, typename... Ts>
    static inline NauPropertyAbstract* createProperty(const NauAnyType& value, Ts&&... params) {
        return nullptr;
    }

    template<size_t baseTypeSize = 0, typename... Ts>
    requires (baseTypeSize < std::tuple_size<NauTypeMap>::value)
    static inline NauPropertyAbstract* createProperty(const NauAnyType& value, Ts&&... params)
    {
        // Type of data storage
        using BaseType = typename std::tuple_element<baseTypeSize, NauTypeMap>::type::first_type;

        // Type of visual container for data
        using WidgetType = typename std::tuple_element<baseTypeSize, NauTypeMap>::type::second_type;

        // Unfolds into a view construct when compiled: if{...} else if {...}...
        if (value.canConvert<BaseType>()) {
            return constructArgs<WidgetType, Ts...>(std::forward<Ts>(params)...).release();
        }

        return createProperty<baseTypeSize + 1, Ts...>(value, std::forward<Ts>(params)...);
    }
};
#pragma endregion


// TODO: After updating the folder structure of the editor,
// separate all levels of abstraction into different files for ease of use.
// Very soon they will grow to incredible sizes.


// ** NauInspectorComponentSpoiler
//
// A component spoiler serves as a visual encapsulation of properties

class NAU_EDITOR_API NauInspectorComponentSpoiler : public NauSpoiler
{
    Q_OBJECT

public:
    NauInspectorComponentSpoiler(const QString& title, const int animationDuration = 0, NauWidget* parent = 0);
};


// The inspector panel is responsible for displaying
// the properties of the components of the game entity selected by the user on the stage.
// Only this widget needs to know which entity is currently running, which file is selected, and so on.

#pragma region INSPECTOR PANEL LEVEL


// ** NauEntityInspectorPageHeader

class NauStaticTextLabel;

class NAU_EDITOR_API NauEntityInspectorPageHeader : public NauWidget
{
    Q_OBJECT

public:
    NauEntityInspectorPageHeader(const std::string& title, const std::string& subtitle);

    void changeTitle(const std::string& title);

private:
    inline static constexpr int Height = 80;
    inline static constexpr int OuterMargin = 16;
    inline static constexpr int HorizontalSpacer = 26;

    NauStaticTextLabel* m_title;
};


// ** NauEntityInspectorPage

class NAU_EDITOR_API NauEntityInspectorPage : public NauWidget
{
    Q_OBJECT

    using NauPropertyMap = std::unordered_map<QString, NauPropertyAbstract*>;

public:
    NauEntityInspectorPage(NauWidget* parent);

    void loadEntitiesProperties(std::shared_ptr<std::vector<NauObjectPtr>> objects, const NauViewAttributes& viewAttributes);
    void loadEntityProperties(NauObjectPtr object, const NauAttributesMap& viewAttributes);
    NauInspectorComponentSpoiler* findOrAddComponent(const std::string& componentName);
    void clear();

signals:
    void eventValueChanged(NauObjectGUID guid, const std::string& cids, const NauAnyType& value);

private:
    // TODO: Needed for update system
    void tick();
    void updateProperties();

private:
    NauLayoutVertical* m_layout;

    std::map<std::string, NauInspectorComponentSpoiler*> m_components;

    // TODO: Needed for update system
    NauPropertyMap m_propertyMap;
    NauObjectPtr m_currentObject;
    QTimer m_updateUiTimer;

    inline static constexpr int ErrorLabelOuterMargin = 16;
};

#pragma endregion
