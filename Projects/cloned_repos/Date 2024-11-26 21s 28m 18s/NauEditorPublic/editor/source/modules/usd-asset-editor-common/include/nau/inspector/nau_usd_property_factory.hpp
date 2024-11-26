// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Usd properties factory

#pragma once

#include "nau/nau_usd_asset_editor_common_config.hpp"
#include "nau_log.hpp"

#include <tuple>
#include <string>
#include <unordered_map>

#include "nau_usd_inspector_widgets.hpp"


// ** NauUsdPropertyFactory

class NAU_USD_ASSET_EDITOR_COMMON_API NauUsdPropertyFactory
{
    using NauUsdTypeMap = std::tuple
        <
        std::pair<bool, NauUsdPropertyBool>,
        std::pair<std::string, NauUsdPropertyString>,
        std::pair<int, NauUsdPropertyInt>,
        std::pair<float, NauUsdPropertyFloat>,
        std::pair<pxr::GfMatrix4d, NauUsdPropertyMatrix>,
        std::pair<pxr::GfVec2d, NauUsdPropertyDouble2>,
        std::pair<pxr::GfVec3d, NauUsdPropertyDouble3>,
        std::pair<pxr::GfVec4d, NauUsdPropertyDouble4>
        // TODO: Implememt color widget for USD property (there is no color type in USD)
        //std::pair<QColor, NauPropertyColor>
        >;

    typedef NauUsdPropertyAbstract* (*FactoryType)(const std::string&, const std::string&);

private:

    // Basic usd data types
    // https://openusd.org/dev/api/_usd__page__datatypes.html 

    // TODO:
    // - Add support for commented types in the future;
    // - There should be a common widget for all real ones, where the data type constraints will be set;
    // - Arrays should be templated in the future, and they should not be in this dictionary;
    // - For the distant future, most creators should be template - based. Now unfortunately we are stuck in standard Qt views.

    static inline std::unordered_map<std::string, FactoryType> m_factoryTypeMap = {
        // Logical data types
        {"bool",     NauUsdPropertyBool::Create},
        //{"uchar",},

        // Integer data types
        {"int",        NauUsdPropertyInt::Create},
        //{"uint",},
        //{"int64",},
        //{"uint64",},

        // Real data types
        //{"half",},
        //{"float",},
        {"float",      NauUsdPropertyFloat::Create},
        {"double",     NauUsdPropertyDouble::Create},

        // String data types
        {"string",     NauUsdPropertyString::Create},

        // Different data types
        //{"timecode",},
        //{"token",},
        //{"opaque",},

        // Matrix data types
        //{"matrix2d",},
        //"matrix3d",},
        {"matrix4d",   NauUsdPropertyMatrix::Create},

        // Quaternions
        //{"quatd",},
        //{"quatf",},
        //{"quath",},

        // Vec2 data types
        {"double2",    NauUsdPropertyDouble2::Create},
        //{"float2",},
        //{"half2",},
        {"int2",       NauUsdPropertyInt2::Create},

        // Vec3 data types
        {"double3",    NauUsdPropertyDouble3::Create},
        //{"float3",},
        //{"half3",},
        {"int3",       NauUsdPropertyInt3::Create},

        // Vec4 data types
        {"double4",    NauUsdPropertyDouble4::Create},
        //{"float4",},
        //{"half4",},
        {"int4",       NauUsdPropertyInt4::Create},

        // Roles

        // Point3 data types
        {"point3d",    NauUsdPropertyDouble3::Create},
        //{"point3f",},
        //{"point3h",},

        // Normal3 data types
        {"normal3d",   NauUsdPropertyDouble3::Create},
        //{"normal3f",},
        //{"normal3h",},

        // Vector3 data types
        {"vector3d",   NauUsdPropertyDouble3::Create},
        //{"vector3f",},
        //{"vector3h",},

        // Color RGB data types
        //{"color3d",},
        //{"color3f",},
        //{"color3h",},

        // Color RGBA data types
        {"color4d",    NauUsdPropertyColor::Create},
        //{"color4f",},
        //{"color4h",},

        // Frame data type
        {"frame4d",    NauUsdPropertyMatrix::Create},

        // UV data types
        {"texCoord2d", NauUsdPropertyDouble2::Create},
        //{"texCoord2f",},
        //{"texCoord2h",},

        // UVW data types
        {"texCoord3d", NauUsdPropertyDouble3::Create},
        //{"texCoord3f",},
        //{"texCoord3h",},

        // Group data type
        //{"group",},

        // Asset data type
        {"asset",      NauAssetProperty::Create},

        // Reference data type
        {"reference",  NauReferenceProperty::Create},
    };
    
public:

    // Widget factory where the data type is known.

    static inline NauUsdPropertyAbstract* createPropertyWidget(const std::string& typeName, const std::string& propertyName, const std::string& metaInfo)
    {
        //if (auto factoryIt = m_factoryTypeMap.find(propertyName); factoryIt != m_factoryTypeMap.end()) {
        //    return factoryIt->second(propertyName, metaInfo);
        //}

        if (auto factoryIt = m_factoryTypeMap.find(typeName); factoryIt != m_factoryTypeMap.end()) {
            return factoryIt->second(propertyName, metaInfo);
        }

        NED_WARNING("Inspector: there is no widget view for either {} nor for {}", propertyName, typeName);
        return nullptr;
    }

    static inline bool addPropertyWidgetCreator(const std::string& role, FactoryType creator)
    {
        if (m_factoryTypeMap.contains(role)) {
            NED_WARNING("A creator already exists for this key. You can add namespace to the type name.");
            return false;
        }

        m_factoryTypeMap[role] = creator;

        NED_INFO("Property widget for <{}> registered successfully", role);
        return true;
    }

    // A widget factory where the data type is not known.
    // (Information about usd alias will be lost)

    template <typename Concrete, typename... Ts>
    static std::unique_ptr<Concrete> constructArgs(Ts&&... params)
    {
        if constexpr (std::is_constructible_v<Concrete, Ts...>) {
            return std::make_unique<Concrete>(std::forward<Ts>(params)...);
        }

        return nullptr;
    }

    // If no matching pair is found, this overload will be called
    template<size_t baseTypeSize, typename... Ts>
    static inline NauUsdPropertyAbstract* createProperty(const PXR_NS::VtValue& value, Ts&&... params) {
        return nullptr;
    }

    template<size_t baseTypeSize = 0, typename... Ts>
    requires (baseTypeSize <std::tuple_size<NauUsdTypeMap>::value)
    static inline NauUsdPropertyAbstract* createProperty(const PXR_NS::VtValue& value, Ts&&... params)
    {
        // Type of data storage
        using BaseType = typename std::tuple_element<baseTypeSize, NauUsdTypeMap>::type::first_type;

        // Type of visual container for data
        using WidgetType = typename std::tuple_element<baseTypeSize, NauUsdTypeMap>::type::second_type;

        // Unfolds into a view construct when compiled: if{...} else if {...}...
        if (value.IsHolding<BaseType>()) {
            return constructArgs<WidgetType, Ts...>(std::forward<Ts>(params)...).release();
        }

        return createProperty<baseTypeSize + 1, Ts...>(value, std::forward<Ts>(params)...);
    }
};
