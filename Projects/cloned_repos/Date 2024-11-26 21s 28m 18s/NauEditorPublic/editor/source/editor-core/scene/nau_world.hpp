// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Nau World is a data model from the Nau Engine adapted for the purposes of the Nau Editor.

#pragma once

#include "nau/nau_editor_config.hpp"

#include "nau/nau_editor_engine_api.hpp"
#include "nau_concepts.hpp"
#include "src/utility/nau_guid.hpp"

#include <QObject>
#include <QMap>

#include <any>
#include <iterator>
#include <memory>
#include <map>
#include <string>
#include <typeinfo>
#include <vector>

class NauObjectProperty;
class NauTemplateInfo;
class NauTemplatesData;

// Alias for standard editor properties container
using NauPropertiesContainer = QMap<QString, NauObjectProperty>; // TODO: Don't use string as key

// TODO: Probably over time we will need our own data types.
// Then it would make sense to create an enum NauType

// ** NauAnyType
// 
// A supposedly safe container for storing data common to the engine and editor

// TODO: Conduct research. Perhaps template methods are not a waste.
// If so, then it will be possible to rewrite this class with a factory pattern.
class NAU_EDITOR_API NauAnyType
{
public:
    NauAnyType() {};

    template<IsNauType T>
    NauAnyType(T value)
    {
        setValueInternal(value);
    }

    // Perhaps it is worth prohibiting itself, possibly changing the value after creation
    template<IsNauType T>
    void setValue(T&& value, bool force = false)
    {
        setValueInternal(value, force);
    }

    template<IsNauType T>
    T convert() const
    {
        return std::any_cast<T>(m_value);
    }

    template<IsNauType T>
    bool canConvert() const
    {
        return m_value.type() == typeid(T);
    }

    template<IsNauType T>
    bool as(T& value)
    {
        if (!canConvert<T>()) {
            return false;
        }

        value = convert<T>();
        return true;
    }

    // For ease of use, returns a C-string
    const char* getTypeName() const { return m_value.type().name(); }

private:

    template<typename T>
    void setValueInternal(T&& value, bool force = false)
    {
        using type = std::remove_reference_t<T>;

        if (!force && canConvert<type>() && convert<type>() != value) {
            return;
        }

        m_value = std::move(value);
    }

private:
    std::any m_value;
};


// ** NauObjectProperty
// 
// Contains a description of a specific component for a game entity

class NAU_EDITOR_API NauObjectProperty
{
public:
    NauObjectProperty() = default;
    NauObjectProperty(const QString& name, const std::string& componentName = std::string());
    NauObjectProperty(const QString& name, const NauAnyType& value, const std::string& componentName = std::string());

    NauObjectProperty& operator=(const NauObjectProperty& other);

    void setValue(const NauAnyType& value);
    void setValue(NauAnyType&& value);

    void setComponentName(const std::string& componentName);

    const NauAnyType& value() const;
    const QString& name() const;
    const std::string& componentName() const;

private:
    QString       m_name;
    NauAnyType    m_value;
    std::string   m_componentName;
};


// TODO: At this point, all the data in this structure is fake!
// After the transition to USD, this data block should be retrieved from the scene and have the following contents:
// 1) Visibility/Display status
// 2) Edit disabled status
// 3) Name
// 4) Type
// 5) ID
// 6) Modification date
// 7) Tags (in question)
// 8) Layer(in question)
struct NauObjectMeta
{
    bool visibilityStatus = true;
    bool disabledStatus = false;
    QString modificationDate = "01.01.1970"; // Change to a suitable data type for storing time data
    std::vector<QString> tegs = {"_tag0","_tag1","_tag3"};
    QString layer = "Main layer: 0";
};


// ** NauObject
//
// Base class for all objects that can be displayed in the editor

class NAU_EDITOR_API NauObject : public QObject
{
    Q_OBJECT

    friend class NauRendInstUpdateSystem;

public:
    NauObject() = default;

    NauObject(const std::string& displayName, NauPropertiesContainer&& properties = NauPropertiesContainer());

public:
    // Editor scene unique identifier
    NauObjectGUID guid;

    const std::string& displayName() const;
    void setDisplayName(const std::string& displayName);

    void addComponent(const NauObjectProperty& component);
    void addComponent(NauObjectProperty&& component);

    void setPropertyValue(const std::string& componentName, const NauAnyType& value);
    NauAnyType getPropertyValue(const std::string& componentName) const;

    const NauPropertiesContainer& properties() const;
    std::shared_ptr<NauPropertiesContainer> propertiesPtr();

    bool hasComponent(const std::string& name) const;

    // Need for inspector update
    // TODO: Make update widget system
    bool isModified() const { return m_isModified; }
    void handleModified() { m_isModified = false; }

signals:
    void eventComponentChanged(const std::string& componentName, const NauAnyType& value);
    void eventDisplayNameChanged(const std::string& newName);

protected:
    // The name of the object that is displayed in the scene hierarchy
    // TODO: If the object has a "" component, 
    // then the value of this variable must be synchronized with it.
    std::string m_displayName;

    // TODO: it is possible to store properties in the form of a tree
    NauPropertiesContainer m_properties;

    // Need for inspector update
    // TODO: Make update widget system
    bool m_isModified;
};


// ** NauEntity
//
// The base entity class used to render data in the Nau Editor
//
// TODO: Merge with NauObject

class NAU_EDITOR_API NauEntity : public NauObject
{
public:

    NauEntity() = default;

    NauEntity(const std::string& templateName,
        NauPropertiesContainer&& properties = NauPropertiesContainer(), bool isProtected = false);

    NauEntity(const std::string& templateName,
        const std::string& displayName, NauPropertiesContainer&& properties = NauPropertiesContainer(), bool isProtected = false);

    void updateComponentCompositionFromTemplateInfo(const NauTemplateInfo& templateInfo);

    bool isProtected() const { return m_isProtected; }

public:
    // The object has no name in the standard sense. 
    // But there is a strictly defined template name from which this entity was generated.
    const std::string templateName;

    const NauObjectMeta sceneMetaData;

private:

    // You cannot perform basic operations on protected entities: copy, paste, delete, etc.
    // TODO: It's a temporary solution
    // In the future, the entity should have a set of attributes that will describe what operations are available on this entity
    bool m_isProtected;
};


// Alias for NauObject shared pointer
using NauObjectPtr = std::shared_ptr<NauObject>;

// Alias for NauEntity shared pointer
using NauEntityPtr = std::shared_ptr<NauEntity>;


// ** NauEnitityContainer
// 
// Wrapper over a std::vector for working with an array of NauEnitity objects

// TODO: In the future, if the level model is stored in the editor, 
// then this kind of data structures should not be created manually.

class NAU_EDITOR_API NauEnitityContainer
{
    // TODO: In general, the container should be represented as a tree,
    // in order to maintain hierarchical relationships between entities.
    using container = std::vector<NauEntityPtr>;

public:
    NauEnitityContainer() = default;

    void clear();
    bool empty() const { return m_entities.empty(); }

    #ifndef NDEBUG   // Debug only, better not expose in release
    bool contains(const NauEntityPtr entity) const;
    #endif

    void pushBack(const NauEntityPtr entity);
    void pushBack(NauEntityPtr&& entity);

    inline auto begin() { return m_entities.begin(); }
    inline auto begin() const { return m_entities.cbegin(); }
    inline auto end() { return m_entities.end(); }
    inline auto end() const { return m_entities.cend(); }

    // Implicit template causes a linking error due to the synonym "container"
    // Therefore, the full type is specified
    inline std::vector<NauEntityPtr>::size_type size() const { return m_entities.size(); };

    auto& operator [](int index) { return m_entities[index]; }
    auto operator [](int index) const { return m_entities[index]; }

    operator container () { return m_entities; }

    inline auto erase(container::iterator itDeleteElement) { return m_entities.erase(itDeleteElement); }

private:
    container m_entities;
};


// ** NauDataModel
// 
// Editor data model

class NauTemplatesData;

class NAU_EDITOR_API NauDataModel : public QObject
{
    Q_OBJECT

    friend class NauEditorSceneLoader;

public:
    NauDataModel();

    const NauEnitityContainer& objects();

    void setObjectPropertyValue(const NauObjectGUID guid, const std::string& componentName, const NauAnyType& value);  // TODO: use shared_ptr

    NauObjectGUID createObject(const std::string& templateName, const std::string& displayTemplateName, const NauTemplatesData& data, const std::shared_ptr<NauPropertiesContainer>& properties);
    void renameObject(const NauObjectGUID guid, const std::string& newName);
    void removeObject(const NauObjectGUID guid);
    void clear();   // TODO: This doesn't generate a command and shouldn't be here at all!

    NauEntityPtr getObject(const NauObjectGUID guid);

    // TODO: NauDataModel must be unified. Soon such functions should be moved to another place...
    void updateObjectsFromTemplatesData(const NauTemplatesData& templatesData);

    // TODO: Save/load scene inside scene class? To prevent modified control inside scene
    bool isSaved() const { return !m_isDirty; }
    void markSaved() { m_isDirty = false; }

    void addObject(const NauEntityPtr object);

protected:
    // These are for internal usage and don't mark the world dirty!
    // TODO: notifyChanged flag is temporary to solve the problem of calling the selection event when loading a scene
    void addObjectInternal(const NauEntityPtr object, bool notifySelectionChanged = true);
    void removeObjectInternal(const NauObjectGUID guid);

    NauObjectGUID createObjectInternal(const std::string& typeName, const NauTemplatesData& data);
    NauObjectGUID createObjectInternal(const std::string& typeName, const std::string& displayTypeName, const NauTemplatesData& data, const std::shared_ptr<NauPropertiesContainer>& properties);

    void setObjectComponentValueInternal(NauEntityPtr entity, const std::string& componentName, const NauAnyType& value);
    void clearInternal();

    NauEntityPtr createObjectFromType(const std::string& typeName, const NauTemplatesData& data, const std::shared_ptr<NauPropertiesContainer>& properties);

private:
    // TODO: Make the storage of game entities in the form of a tree
    NauEnitityContainer m_objects;

    bool m_isDirty = false;
};


// ** NauTemplateInfo
// 
// Metadata about a specific template

class NAU_EDITOR_API NauTemplateInfo
{

public:
    NauTemplateInfo();
    NauTemplateInfo(const std::string& name, const std::string& displayName, bool canBeCreatedFromEditor, NauPropertiesContainer&& properties, std::vector<std::string>&& extends, bool drawBillboard);
    NauTemplateInfo(const std::string& name, const std::string& displayName, bool canBeCreatedFromEditor, NauPropertiesContainer&& properties);

    const std::string& name() const;
    const std::string& displayName() const;
    const NauPropertiesContainer& propertiesMap() const;

    inline auto canBeCreatedFromEditor() const { return m_canBeCreatedFromEditor; }
    inline auto drawBillboard() const { return m_drawbillboard; }

private:
    std::string m_name;

    // This is the name the user will see in the scene hierarchy
    std::string m_displayName;

    // TODO: In general, this separation exists largely due to the fact that everything is stored in ecs.
    // Because of this, some unwanted systems may try to work and stop the editor from working.
    // Either this problem will be solved by natural evolution of ecs or something will have to be done about it...
    bool m_canBeCreatedFromEditor;  // Can a template be created from the editor

    // TODO: Well, you need to introduce some kind of filter on the order of the properties
    NauPropertiesContainer m_propertiesMap;

    // Not used yet, but could be of great help in the future,
    // since the processing extends order is important
    std::vector<std::string> m_extends;

    bool m_drawbillboard;
};


// ** NauTemplatesData
// 
// Container with metadata about all templates in an open project

class NAU_EDITOR_API NauTemplatesData : public QObject
{
    Q_OBJECT

public:
    NauTemplatesData() = default;
    NauTemplatesData(const NauTemplatesData& templatesData);
    NauTemplatesData& operator=(const NauTemplatesData& templatesData);

    void addTemplate(NauTemplateInfo templateInfo);
    bool containsTemplate(const std::string& templateName) const;
    const NauTemplateInfo& getTemplateInfo(const std::string& templateName) const;
    const std::map<std::string, NauTemplateInfo>& getTemplatesInfo() const;

    void clear();

private:
    // A regular dictionary is used here, because...
    // Processing templates order is important!
    std::map<std::string, NauTemplateInfo> m_templates; // TODO: Don't use string as key
};