// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "scene/nau_world.hpp"
#include "nau_assert.hpp"
#include "nau_log.hpp"
#include "nau/nau_constants.hpp"


std::string getComponentName(const std::string& propertyName, const std::string& initialComponentName)
{
    if (!initialComponentName.empty()) {
        return initialComponentName;
    }

    // TODO: Temporary solution to show a visual breakdown of the properties that doesn't exist yet
    // Delete in the future
    if (propertyName == NauEcsComponentNames::TRANSFORM) {
        return "Transform";
    } else if (propertyName == NauEcsComponentNames::MATERIAL) {
        return "Material";
    } else if (propertyName == NauEcsComponentNames::RI_EXTRA_NAME) {
        return  "Static Mesh";
    } else if (propertyName == "start_position") {
        return  "Start Position";
    } else if (propertyName == "light__box") {
        return  "Light Box";
    } else if (propertyName == "lightModTm") {
        return  "Light Mode TM";
    } else if (propertyName == "texName") {
        return  "Texture";
    }

    return "Object";
}


// ** NauObjectProperty

NauObjectProperty::NauObjectProperty(const QString& name, const std::string& componentName)
    : m_name(name)
    , m_componentName(getComponentName(name.toUtf8().constData(), componentName))
{
}

NauObjectProperty::NauObjectProperty(const QString& name, const NauAnyType& value, const std::string& componentName)
    : m_name(name)
    , m_value(value)
    , m_componentName(getComponentName(name.toUtf8().constData(), componentName))
{
}

void NauObjectProperty::setValue(const NauAnyType& value)
{
    m_value = value;
}

void NauObjectProperty::setValue(NauAnyType&& value)
{
    m_value = value;
}

void NauObjectProperty::setComponentName(const std::string& componentName)
{
    m_componentName = componentName;
}

const NauAnyType& NauObjectProperty::value() const
{
    return m_value;
}

const QString& NauObjectProperty::name() const
{
    return m_name;
}

const std::string& NauObjectProperty::componentName() const
{
    return m_componentName;
}

NauObjectProperty& NauObjectProperty::operator=(const NauObjectProperty& other)
{
    // Guard self assignment
    if (this == &other) {
        return *this;
    }

    m_name = other.m_name;
    m_value = other.m_value;
    m_componentName = other.m_componentName;

    return *this;
}


// ** NauObject

NauObject::NauObject(const std::string& displayName, NauPropertiesContainer&& properties)
    : m_displayName(displayName)
    , m_properties(properties)
{
}

const std::string& NauObject::displayName() const
{
    return m_displayName;
}

void NauObject::setDisplayName(const std::string& displayName)
{
    if (m_displayName != displayName) {
        m_displayName = displayName;
        eventDisplayNameChanged(m_displayName);
    }
}

void NauObject::addComponent(const NauObjectProperty& component)
{
    m_properties[component.name()] = component;
}

void NauObject::addComponent(NauObjectProperty&& component)
{
    m_properties[component.name()] = component;
}

void NauObject::setPropertyValue(const std::string& componentName, const NauAnyType& value)
{
    m_properties[componentName.c_str()].setValue(value);
    m_isModified = true;
    emit eventComponentChanged(componentName, value);
}

NauAnyType NauObject::getPropertyValue(const std::string& componentName) const
{
    return m_properties.value(componentName.c_str()).value();
}

const NauPropertiesContainer& NauObject::properties() const
{
    return m_properties;
}

std::shared_ptr<NauPropertiesContainer> NauObject::propertiesPtr()
{
    return std::make_shared<NauPropertiesContainer>(m_properties);
}

bool NauObject::hasComponent(const std::string& name) const
{
    return m_properties.find(name.c_str()) != m_properties.end();
}


// ** NauEntity

NauEntity::NauEntity(const std::string& templateName, NauPropertiesContainer&& properties, bool isProtected)
    : NauObject(templateName, std::move(properties))
    , templateName(templateName)
    , m_isProtected(isProtected)
{
}

NauEntity::NauEntity(const std::string& templateName, const std::string& displayName, NauPropertiesContainer&& properties, bool isProtected)
    : NauObject(displayName, std::move(properties))
    , templateName(templateName)
    , m_isProtected(isProtected)
{
}

void NauEntity::updateComponentCompositionFromTemplateInfo(const NauTemplateInfo& templateInfo)
{
    // Initialize the updated component container with the base values corresponding to the template
    // If new properties were added, they would go into the container upon initialization and would not be changed
    auto updatedPropertiesMap = templateInfo.propertiesMap();

    // Iteratively go through the containers with the actual properties
    // Removed properties won't even be considered
    for (auto itComponent = updatedPropertiesMap.begin(); itComponent != updatedPropertiesMap.end(); ++itComponent) {
        
        // If the component was previously created, then simply take its current value
        auto itExistingComponent = m_properties.find(itComponent.key());
        if(itExistingComponent != m_properties.end()) {
            itComponent.value() = itExistingComponent.value();
        }
    }

    // Change the data in the model to updated ones
    m_properties = updatedPropertiesMap;
}


// ** NauEnitityContainer

void NauEnitityContainer::clear()
{
    m_entities.clear();
}

#ifndef NDEBUG 
bool NauEnitityContainer::contains(const NauEntityPtr entity) const
{
    return std::find(m_entities.begin(), m_entities.end(), entity) != m_entities.end();
}
#endif

void NauEnitityContainer::pushBack(const NauEntityPtr entity)
{
    m_entities.push_back(entity);
}

void NauEnitityContainer::pushBack(NauEntityPtr&& entity)
{
    m_entities.push_back(entity);
}


// ** NauDataModel

NauDataModel::NauDataModel()
{
}

const NauEnitityContainer& NauDataModel::objects()
{
    return m_objects;
}

NauObjectGUID NauDataModel::createObject(const std::string& templateName, const std::string& name, const NauTemplatesData& data, const std::shared_ptr<NauPropertiesContainer>& properties)
{
    m_isDirty = true;
    return createObjectInternal(templateName, name, data, properties);
}

void NauDataModel::setObjectPropertyValue(const NauObjectGUID guid, const std::string& componentName, const NauAnyType& value)
{
    auto object = getObject(guid);

    if (!object) {
        NED_WARNING("Trying to change non-existent object!");
        return;
    }

    object->setPropertyValue(componentName, value);
}

void NauDataModel::renameObject(const NauObjectGUID guid, const std::string& newName)
{
    auto object = getObject(guid);
    NED_ASSERT(object);

    object->setDisplayName(newName.c_str());

    m_isDirty = true;
}

void NauDataModel::removeObject(const NauObjectGUID guid)
{
    removeObjectInternal(guid);

    m_isDirty = true;
}

void NauDataModel::clear()
{
    clearInternal();
}

NauEntityPtr NauDataModel::getObject(const NauObjectGUID guid)
{
    for (const NauEntityPtr object : m_objects) {
        if (object->guid == guid) {
            return object;
        }
    }

    return nullptr;
}

void NauDataModel::addObject(NauEntityPtr object)
{
    addObjectInternal(object);
} 

void NauDataModel::addObjectInternal(const NauEntityPtr object, bool notifySelectionChanged)
{
    m_objects.pushBack(object);

    connect(object.get(), &NauObject::eventComponentChanged, this, [this, object](const std::string& componentName, const NauAnyType& value) {
        this->setObjectComponentValueInternal(object, componentName, value);
    });
}

NauObjectGUID NauDataModel::createObjectInternal(const std::string& templateName, const NauTemplatesData& data)
{
    NauEntityPtr object = createObjectFromType(templateName.c_str(), data, std::make_shared<NauPropertiesContainer>());
    if (object) {
        addObjectInternal(object);

        connect(object.get(), &NauObject::eventComponentChanged, this, [this, object](const std::string& componentName, const NauAnyType& value) {
            this->setObjectComponentValueInternal(object, componentName, value);
        });

        return object->guid;
    }
    return NauObjectGUID::invalid();
}

NauObjectGUID NauDataModel::createObjectInternal(const std::string& templateName, const std::string& displayName, const NauTemplatesData& data, const std::shared_ptr<NauPropertiesContainer>& properties)
{
    NauEntityPtr object = createObjectFromType(templateName, data, properties);
    if (!object) {
        return NauObjectGUID::invalid();
    }

    object->setDisplayName(displayName);
    addObjectInternal(object);

    connect(object.get(), &NauObject::eventComponentChanged, this, [this, object](const std::string& componentName, const NauAnyType& value) {
        this->setObjectComponentValueInternal(object, componentName, value);
    });

    return object->guid;    
}

NauEntityPtr NauDataModel::createObjectFromType(const std::string& templateName, const NauTemplatesData& data, const std::shared_ptr<NauPropertiesContainer>& properties)
{
    const NauTemplateInfo& templateInfo = data.getTemplateInfo(templateName);
    NauPropertiesContainer templatePropertiesMap = templateInfo.propertiesMap();
    auto itTransformProperty = templatePropertiesMap.find("transform");
    if (itTransformProperty == templatePropertiesMap.end()) {
        NED_ERROR("Transform property is not presented in this template. The entity creation process has been interrupted.");
        return nullptr;
    }

    // Apply properties to default values
    for (auto _override = properties->begin(), end = properties->end(); _override != end; ++_override) {
        templatePropertiesMap[_override.key()].setValue(_override.value().value());
        templatePropertiesMap[_override.key()].setComponentName(_override.value().componentName());
    }
    
    return std::make_shared<NauEntity>(templateName.c_str(), templateInfo.displayName().c_str(), std::move(templatePropertiesMap), templateInfo.drawBillboard());
}

void NauDataModel::setObjectComponentValueInternal(NauEntityPtr entity, const std::string& componentName, const NauAnyType& value)
{
    m_isDirty = true;
}

void NauDataModel::removeObjectInternal(const NauObjectGUID guid)
{
    const auto itDeleteObject = std::find_if(m_objects.begin(), m_objects.end(), [guid](NauEntityPtr entity) {
        return entity->guid == guid;
    });

    m_objects.erase(itDeleteObject);
}

void NauDataModel::clearInternal()
{
    m_objects.clear();
}

void NauDataModel::updateObjectsFromTemplatesData(const NauTemplatesData& templatesData)
{
    for (auto& object : m_objects) {
        if (templatesData.containsTemplate(object->templateName)) {
            object->updateComponentCompositionFromTemplateInfo(templatesData.getTemplateInfo(object->templateName));
        }
    }
}

// ** NauTemplateInfo

NauTemplateInfo::NauTemplateInfo()
    : m_name("")
    , m_canBeCreatedFromEditor(false)
    , m_propertiesMap(NauPropertiesContainer())
    , m_extends(std::vector<std::string>())
{
}

NauTemplateInfo::NauTemplateInfo(const std::string& name, const std::string& displayName, bool canBeCreatedFromEditor, NauPropertiesContainer&& properties, std::vector<std::string>&& extends, bool drawBillboard)
    : m_name(name)
    , m_displayName(displayName)
    , m_canBeCreatedFromEditor(canBeCreatedFromEditor)
    , m_propertiesMap(properties)
    , m_extends(extends)
    , m_drawbillboard(drawBillboard)
{
}

NauTemplateInfo::NauTemplateInfo(const std::string& name, const std::string& displayName, bool canBeCreatedFromEditor, NauPropertiesContainer&& properties)
    : m_name(name)
    , m_displayName(displayName)
    , m_canBeCreatedFromEditor(canBeCreatedFromEditor)
    , m_propertiesMap(properties)
    , m_drawbillboard(false)
{
}

const std::string& NauTemplateInfo::name() const
{
    return m_name;
}

const std::string& NauTemplateInfo::displayName() const
{
    return m_displayName;
}

const NauPropertiesContainer& NauTemplateInfo::propertiesMap() const
{
    return m_propertiesMap;
}


// ** NauTemplatesData

NauTemplatesData::NauTemplatesData(const NauTemplatesData& templatesData)
{
    m_templates = templatesData.m_templates;
}

NauTemplatesData& NauTemplatesData::operator=(const NauTemplatesData& templatesData)
{
    // Guard self assignment
    if (this == &templatesData) {
        return *this;
    }

    m_templates = templatesData.m_templates;
    return *this;
}

void NauTemplatesData::addTemplate(NauTemplateInfo templateInfo)
{
    m_templates[templateInfo.name()] = templateInfo;
}

bool NauTemplatesData::containsTemplate(const std::string& templateName) const
{
    return m_templates.find(templateName) != m_templates.end();
}

const NauTemplateInfo& NauTemplatesData::getTemplateInfo(const std::string& templateName) const
{
    return m_templates.at(templateName);
}

const std::map<std::string, NauTemplateInfo>& NauTemplatesData::getTemplatesInfo() const
{
    return m_templates;
}

void NauTemplatesData::clear()
{
    m_templates.clear();
}
