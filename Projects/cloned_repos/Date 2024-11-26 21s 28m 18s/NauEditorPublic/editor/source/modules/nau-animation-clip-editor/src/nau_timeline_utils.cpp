// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/nau_timeline_utils.hpp"

#include "nau_assert.hpp"

#include <QTextStream>
#include <cmath>


// ** NauAnimationPropertyDataTraits

template<typename EngineDataType>
struct NauAnimationPropertyDataTraits {};


template<>
struct NauAnimationPropertyDataTraits<bool>
{
    using UsdDataType = bool;
    using EngineDataType = bool;

    [[nodiscard]]
    static EngineDataType convertFromUsd(UsdDataType data) noexcept { return data; }
    [[nodiscard]]
    static UsdDataType convertToUsd(EngineDataType data) noexcept { return data; }
};


template<>
struct NauAnimationPropertyDataTraits<int>
{
    using UsdDataType = int;
    using EngineDataType = int;

    [[nodiscard]]
    static EngineDataType convertFromUsd(UsdDataType data) noexcept { return data; }
    [[nodiscard]]
    static UsdDataType convertToUsd(EngineDataType data) noexcept { return data; }
};


template<>
struct NauAnimationPropertyDataTraits<float>
{
    using UsdDataType = float;
    using EngineDataType = float;

    [[nodiscard]]
    static EngineDataType convertFromUsd(UsdDataType data) noexcept { return data; }
    [[nodiscard]]
    static UsdDataType convertToUsd(EngineDataType data) noexcept { return data; }
};


template<>
struct NauAnimationPropertyDataTraits<nau::math::vec3>
{
    using UsdDataType = pxr::GfVec3f;
    using EngineDataType = nau::math::vec3;

    [[nodiscard]]
    static EngineDataType convertFromUsd(const UsdDataType& data) noexcept { return { data[0], data[1], data[2] }; }
    [[nodiscard]]
    static UsdDataType convertToUsd(const EngineDataType& data) noexcept { return { data[0], data[1], data[2] }; }
};


template<>
struct NauAnimationPropertyDataTraits<nau::math::quat>
{
    using UsdDataType = pxr::GfQuatf;
    using EngineDataType = nau::math::quat;

    [[nodiscard]]
    static EngineDataType convertFromUsd(const UsdDataType& data) noexcept
    {
        const auto imaginary = data.GetImaginary();
        return { imaginary[0], imaginary[1], imaginary[2], data.GetReal() };
    }
    [[nodiscard]]
    static UsdDataType convertToUsd(const EngineDataType& data) noexcept { return { data[3], data[0], data[1], data[2] }; }
};


namespace nau
{
    int CalculateTimelineSizeInPixels(float timeValue, float timeStep, float trackWidth) noexcept
    {
        return static_cast<int>(std::round(timeValue / timeStep * trackWidth));
    }

    float CalculateTimeFromPosition(int timelinePosition, float timeStep, int trackWidth) noexcept
    {
        return static_cast<float>(timelinePosition) / static_cast<float>(trackWidth) * timeStep;
    }

    QString CalculateTimeToString(float time, bool useMilliseconds)
    {
        const int minutes = static_cast<int>(time) / 60;
        const int seconds = static_cast<int>(time) % 60;
        QString timeString;
        QTextStream out(&timeString);
        out << (minutes < 10 ? "0" : "") << minutes << ":" << (seconds < 10 ? "0" : "") << seconds;
        if (useMilliseconds) {
            const float milliseconds = std::roundf((time - static_cast<float>(seconds)) * 1000.f);
            out << "." << (milliseconds >= 100.f ? "" : (milliseconds < 10.f ? "00" : "0"));
            out << static_cast<int>(milliseconds);
        }
        return timeString;
    }


    template<class T>
    static void SetPropertyData(const T& data, float time, const pxr::UsdAttribute& attribute)
    {
        using Traits = NauAnimationPropertyDataTraits<T>;
        attribute.Set(Traits::convertToUsd(data), time);
    }


    template<class T, class U = NauAnimationPropertyDataTraits<T>::UsdDataType>
    static void SetPropertyData(const T& data, float time, const pxr::UsdAttribute& attribute, int jointIndex, pxr::VtArray<U>& cache)
    {
        using Traits = NauAnimationPropertyDataTraits<T>;

        attribute.Get(&cache, time);
        cache[jointIndex] = Traits::convertToUsd(data);
        attribute.Set(cache, time);
    }


    template<class T>
    static auto GetPropertyData(float time, const std::vector<double>& samples, const pxr::UsdAttribute& attribute)
    {
        using Traits = NauAnimationPropertyDataTraits<T>;
        using ReturnType = typename Traits::UsdDataType;

        auto rightIt = std::upper_bound(samples.begin(), samples.end(), time);
        if (rightIt == samples.end()) {
            --rightIt;
        }

        ReturnType rightData;
        attribute.Get(&rightData, *rightIt);

        const float rightFrameTime = *rightIt;
        if ((rightFrameTime <= time) || (rightIt == samples.begin())) {
            return Traits::convertFromUsd(rightData);
        }

        auto leftIt = --rightIt;
        ReturnType leftData;
        attribute.Get(&leftData, *leftIt);

        const float leftFrameTime = *leftIt;
        const float timeValue = std::clamp((time - leftFrameTime) / (rightFrameTime - leftFrameTime), 0.f, 1.f);
        const ReturnType currentData = leftData + timeValue * (rightData - leftData);

        return Traits::convertFromUsd(currentData);
    }


    template<class T, class U = NauAnimationPropertyDataTraits<T>::UsdDataType>
    static auto GetPropertyData(float time, const std::vector<double>& samples, const pxr::UsdAttribute& attribute, int jointIndex, pxr::VtArray<U>& cache)
    {
        using Traits = NauAnimationPropertyDataTraits<T>;
        using ReturnType = typename Traits::UsdDataType;

        auto rightIt = std::upper_bound(samples.begin(), samples.end(), time);
        if (rightIt == samples.end()) {
            --rightIt;
        }
        attribute.Get(&cache, *rightIt);
        const ReturnType rightData = cache[jointIndex];

        const float rightFrameTime = *rightIt;
        if ((rightFrameTime <= time) || (rightIt == samples.begin())) {
            return Traits::convertFromUsd(rightData);
        }

        auto leftIt = --rightIt;
        attribute.Get(&cache, *leftIt);
        if (cache.empty()) {
            return Traits::convertFromUsd(rightData);
        }
        const ReturnType leftData = cache[jointIndex];

        const float leftFrameTime = *leftIt;
        const float timeValue = std::clamp((time - leftFrameTime) / (rightFrameTime - leftFrameTime), 0.f, 1.f);
        const ReturnType currentData = leftData + timeValue * (rightData - leftData);

        return Traits::convertFromUsd(currentData);
    }
}


// ** NauAnimationPropertyData

NauAnimationPropertyData::NauAnimationPropertyData(const NauAnimationPropertyData& rhs)
    : variant(rhs.variant)
{
}

NauAnimationPropertyData NauAnimationPropertyData::defaultPropertyValue(const NauAnimationProperty& property) noexcept
{
    switch (property.type()) {
        case NauAnimationTrackDataType::Bool:
            return false;
        case NauAnimationTrackDataType::Int:
            return 0;
        case NauAnimationTrackDataType::Float:
            return 0.f;
        case NauAnimationTrackDataType::Vec3:
            return nau::math::vec3(0.f, 0.f, 0.f);
        case NauAnimationTrackDataType::Quat:
            return nau::math::quat::identity();
    }
    NAU_ASSERT(false, "Unsupported animation property data type!");
    return NauAnimationPropertyData();
}

NauAnimationPropertyData& NauAnimationPropertyData::operator=(const NauAnimationPropertyData& rhs)
{
    if (this != &rhs) {
        variant = rhs.variant;
    }
    return *this;
}


// ** NauAnimationProperty

NauAnimationProperty::NauAnimationProperty(std::string name, NauAnimationTrackDataType type, bool selected)
    : m_name(std::move(name))
    , m_type(type)
    , m_selected(selected)
{
}

void NauAnimationProperty::setKeyframesAttribute(pxr::UsdAttribute attribute) noexcept
{
    m_keyframesAttribute = std::move(attribute);
    m_keyframesAttribute.GetTimeSamples(&m_sampleList);
}

void NauAnimationProperty::reset()
{
    m_sampleList.clear();
    m_keyframesAttribute = {};
    m_prim = {};
    m_selected = false;
}

const pxr::SdfValueTypeName& NauAnimationProperty::typeName() const noexcept
{
    switch (m_type) {
        case NauAnimationTrackDataType::Bool:
            return pxr::SdfValueTypeNames->Bool;
        case NauAnimationTrackDataType::Int:
            return pxr::SdfValueTypeNames->Int;
        case NauAnimationTrackDataType::Float:
            return pxr::SdfValueTypeNames->Float;
        case NauAnimationTrackDataType::Vec3:
            return pxr::SdfValueTypeNames->Float3;
        case NauAnimationTrackDataType::Quat:
            return pxr::SdfValueTypeNames->Quatf;
    }
    NAU_ASSERT(false, "Unsupported animation property data type!");
    return pxr::SdfValueTypeNames->Bool;
}


// ** NauAnimationClipProperty

NauAnimationClipProperty::NauAnimationClipProperty(std::string name, NauAnimationTrackDataType type, bool selected)
    : NauAnimationProperty(std::move(name), type, selected)
{
}

NauAnimationPropertyData NauAnimationClipProperty::dataForTime(float time) const
{
    if (m_sampleList.empty()) {
        return NauAnimationPropertyData::defaultPropertyValue(*this);
    }
    switch (m_type) {
        case NauAnimationTrackDataType::Bool:
            return nau::GetPropertyData<bool>(time, m_sampleList, m_keyframesAttribute);
        case NauAnimationTrackDataType::Int:
            return nau::GetPropertyData<int>(time, m_sampleList, m_keyframesAttribute);
        case NauAnimationTrackDataType::Float:
            return nau::GetPropertyData<float>(time, m_sampleList, m_keyframesAttribute);
        case NauAnimationTrackDataType::Vec3:
            return nau::GetPropertyData<nau::math::vec3>(time, m_sampleList, m_keyframesAttribute);
        case NauAnimationTrackDataType::Quat:
            return nau::GetPropertyData<nau::math::quat>(time, m_sampleList, m_keyframesAttribute);
    }
    return NauAnimationPropertyData::defaultPropertyValue(*this);
}

void NauAnimationClipProperty::setKeyframeData(float time, const NauAnimationPropertyData& data)
{
    if (!m_keyframesAttribute.IsValid()) {
        return;
    }
    switch (m_type) {
        case NauAnimationTrackDataType::Bool:
            nau::SetPropertyData(std::get<bool>(data.variant), time, m_keyframesAttribute);
            break;
        case NauAnimationTrackDataType::Int:
            nau::SetPropertyData(std::get<int>(data.variant), time, m_keyframesAttribute);
            break;
        case NauAnimationTrackDataType::Float:
            nau::SetPropertyData(std::get<float>(data.variant), time, m_keyframesAttribute);
            break;
        case NauAnimationTrackDataType::Vec3:
            nau::SetPropertyData(std::get<nau::math::vec3>(data.variant), time, m_keyframesAttribute);
            break;
        case NauAnimationTrackDataType::Quat:
            nau::SetPropertyData(std::get<nau::math::quat>(data.variant), time, m_keyframesAttribute);
            break;
        default:
            NED_ASSERT(false);
            return;
    }
    m_keyframesAttribute.GetTimeSamples(&m_sampleList);
}

void NauAnimationClipProperty::deleteKeyframe(float time)
{
    if (!m_keyframesAttribute.IsValid()) {
        return;
    }
    m_keyframesAttribute.ClearAtTime(time);
    m_keyframesAttribute.GetTimeSamples(&m_sampleList);
}

void NauAnimationClipProperty::changeKeyframeTime(float timeOld, float timeNew)
{
    if (!m_keyframesAttribute.IsValid()) {
        return;
    }
    if (std::none_of(m_sampleList.begin(), m_sampleList.end(), [timeOld](double time) { return timeOld == time; })) {
        return;
    }
    auto&& data = dataForTime(timeOld);
    deleteKeyframe(timeOld);
    setKeyframeData(timeNew, data);
}


// ** NauAnimationSkelProperty

NauAnimationSkelProperty::NauAnimationSkelProperty(std::string name, NauAnimationTrackDataType type, uint32_t jointIndex, bool selected)
    : NauAnimationProperty(std::move(name), type, selected)
    , m_jointIndex(jointIndex)
{
}

NauAnimationPropertyData NauAnimationSkelProperty::dataForTime(float time) const
{
    if (m_sampleList.empty() || (m_jointIndex == INVALID_JOINT_INDEX)) {
        return NauAnimationPropertyData::defaultPropertyValue(*this);
    }
    if (m_type == NauAnimationTrackDataType::Vec3) {
        pxr::VtArray<pxr::GfVec3f> dataList;
        m_keyframesAttribute.Get(&dataList, m_sampleList.front());
        if (dataList.empty()) {
            return NauAnimationPropertyData::defaultPropertyValue(*this);
        }
        return nau::GetPropertyData<nau::math::vec3>(time, m_sampleList, m_keyframesAttribute, m_jointIndex, dataList);
    }
    if (m_type == NauAnimationTrackDataType::Quat) {
        pxr::VtArray<pxr::GfQuatf> dataList;
        m_keyframesAttribute.Get(&dataList, m_sampleList.front());
        if (dataList.empty()) {
            return NauAnimationPropertyData::defaultPropertyValue(*this);
        }
        return nau::GetPropertyData<nau::math::quat>(time, m_sampleList, m_keyframesAttribute, m_jointIndex, dataList);
    }
    NAU_ASSERT(false, "Unsupported skeleton data type!");
    return NauAnimationPropertyData::defaultPropertyValue(*this);
}

void NauAnimationSkelProperty::setKeyframeData(float time, const NauAnimationPropertyData& data)
{
    if (!m_keyframesAttribute.IsValid() || (m_jointIndex == INVALID_JOINT_INDEX)) {
        return;
    }

    switch (m_type) {
        case NauAnimationTrackDataType::Vec3: {
            pxr::VtVec3fArray cache;
            nau::SetPropertyData(std::get<nau::math::vec3>(data.variant), time, m_keyframesAttribute, m_jointIndex, cache);
        }    break;
        case NauAnimationTrackDataType::Quat: {
            pxr::VtQuatfArray cache;
            nau::SetPropertyData(std::get<nau::math::quat>(data.variant), time, m_keyframesAttribute, m_jointIndex, cache);
        }   break;
        case NauAnimationTrackDataType::Bool:
        case NauAnimationTrackDataType::Int:
        case NauAnimationTrackDataType::Float:
        default:
            NAU_ASSERT(false, "Unsupported skeleton data type!");
            break;
    }
    m_keyframesAttribute.GetTimeSamples(&m_sampleList);
}

void NauAnimationSkelProperty::deleteKeyframe(float time)
{
    m_keyframesAttribute.ClearAtTime(time);
    m_keyframesAttribute.GetTimeSamples(&m_sampleList);
}

void NauAnimationSkelProperty::changeKeyframeTime(float timeOld, float timeNew)
{
    NAU_ASSERT(false, "No implementation!");
}


// ** NauAnimationPropertyList

void NauAnimationPropertyList::setRefillFlag(bool flag) noexcept
{
    m_refilled = flag;
    if (flag) {
        m_clipPropertyList.clear();
        m_skelPropertyList.clear();
    }
}

void NauAnimationPropertyList::setReadOnly(bool flag) noexcept
{
    const auto pred = [flag](NauAnimationProperty& property) { property.setReadOnly(flag); };
    std::for_each(m_clipPropertyList.begin(), m_clipPropertyList.end(), pred);
    std::for_each(m_skelPropertyList.begin(), m_skelPropertyList.end(), pred);
}

void NauAnimationPropertyList::setTimeAnimation(float start, float end) noexcept
{
    m_startAnimation = std::min(start, end);
    m_endAnimation = std::max(m_startAnimation, end);
}

void NauAnimationPropertyList::setFrameDuration(float value) noexcept
{
    if (m_skelPropertyList.empty()) {
        m_frameDuration = std::max(1.f / 120.f, value);
    } else {
        m_frameDuration = 1.f / 60.f; // TODO: fix constant in AnimationController
    }
}

bool NauAnimationPropertyList::empty() const noexcept
{
    return m_clipPropertyList.empty() && m_skelPropertyList.empty();
}

size_t NauAnimationPropertyList::size() const noexcept
{
    return m_clipPropertyList.size() + m_skelPropertyList.size();
}

bool NauAnimationPropertyList::refilled() const noexcept
{
    return m_refilled;
}

NauAnimationProperty* NauAnimationPropertyList::propertyByIndex(int propertyIndex) noexcept
{
    if (!m_clipPropertyList.empty()) {
        return m_clipPropertyList.data() + propertyIndex;
    }
    if (!m_skelPropertyList.empty()) {
        return m_skelPropertyList.data() + propertyIndex;
    }
    return nullptr;
}
