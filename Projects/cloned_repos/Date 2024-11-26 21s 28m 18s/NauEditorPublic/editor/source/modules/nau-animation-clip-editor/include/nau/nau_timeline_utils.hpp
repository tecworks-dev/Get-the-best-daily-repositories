// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Timeline utility structs and functions

#pragma once

#include "nau/math/math.h"

#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/prim.h>
#include <QString>
#include <memory>
#include <set>
#include <string>
#include <variant>


namespace nau
{
    int CalculateTimelineSizeInPixels(float timeValue, float timeStep, float trackWidth) noexcept;

    float CalculateTimeFromPosition(int timelinePosition, float timeStep, int trackWidth) noexcept;

    QString CalculateTimeToString(float time, bool useMilliseconds);
}


enum class NauAnimationTrackDataType
{
    Bool,
    Int,
    Float,
    Vec3,
    Quat
};


class NauAnimationProperty;


// ** NauAnimationPropertyData

struct NauAnimationPropertyData
{
    // TODO: Make abstract factory
    using Variant = std::variant<bool, int, float, nau::math::vec3, nau::math::quat>;

    NauAnimationPropertyData() noexcept = default;
    NauAnimationPropertyData(const NauAnimationPropertyData& rhs);

    template<class T>
    NauAnimationPropertyData(const T& data)  noexcept : variant(data) {}

    static NauAnimationPropertyData defaultPropertyValue(const NauAnimationProperty& property) noexcept;

    NauAnimationPropertyData& operator = (const NauAnimationPropertyData& rhs);

    Variant variant{};
};


// ** NauAnimationProperty

class NauAnimationProperty
{
public:
    NauAnimationProperty(std::string name, NauAnimationTrackDataType type, bool selected);

    NauAnimationProperty(const NauAnimationProperty& rhs) = default;
    NauAnimationProperty(NauAnimationProperty&& rhs) noexcept = default;

    NauAnimationProperty& operator = (const NauAnimationProperty& rhs) = default;
    NauAnimationProperty& operator = (NauAnimationProperty&& rhs) noexcept = default;

    virtual ~NauAnimationProperty() noexcept = default;

    void setPrim(pxr::UsdPrim prim) noexcept { m_prim = std::move(prim); }
    void setKeyframesAttribute(pxr::UsdAttribute attribute) noexcept;
    void setSelected(bool flag) noexcept { m_selected = flag; }
    void setReadOnly(bool flag) noexcept { m_readOnly = flag; }
    void reset();

    [[nodiscard]]
    const std::string& name() const noexcept { return m_name; }
    [[nodiscard]]
    const pxr::UsdPrim& prim() const noexcept { return m_prim; }
    [[nodiscard]]
    NauAnimationTrackDataType type() const noexcept { return m_type; }
    [[nodiscard]]
    const std::vector<double>& timeSamples() const noexcept { return m_sampleList; }
    [[nodiscard]]
    const pxr::SdfValueTypeName& typeName() const noexcept;
    [[nodiscard]]
    bool selected() const noexcept { return m_selected; }
    [[nodiscard]]
    bool isReadOnly() const noexcept { return m_readOnly; }

    [[nodiscard]]
    virtual NauAnimationPropertyData dataForTime(float time) const = 0;
    virtual void setKeyframeData(float time, const NauAnimationPropertyData& data) = 0;
    virtual void deleteKeyframe(float time) = 0;
    virtual void changeKeyframeTime(float timeOld, float timeNew) = 0;

protected:
    std::string m_name;
    std::vector<double> m_sampleList;
    pxr::UsdPrim m_prim;
    pxr::UsdAttribute m_keyframesAttribute;
    NauAnimationTrackDataType m_type;
    bool m_selected = false;
    bool m_readOnly = false;
};


// ** NauAnimationClipProperty

class NauAnimationClipProperty : public NauAnimationProperty
{
public:
    NauAnimationClipProperty(std::string name, NauAnimationTrackDataType type, bool selected);

    [[nodiscard]]
    NauAnimationPropertyData dataForTime(float time) const override;

    void setKeyframeData(float time, const NauAnimationPropertyData& data) override;
    void deleteKeyframe(float time) override;
    void changeKeyframeTime(float timeOld, float timeNew) override;
};


// ** NauAnimationClipProperty

class NauAnimationSkelProperty : public NauAnimationProperty
{
public:
    NauAnimationSkelProperty(std::string name, NauAnimationTrackDataType type, uint32_t jointIndex, bool selected);

    [[nodiscard]]
    NauAnimationPropertyData dataForTime(float time) const override;

    void setKeyframeData(float time, const NauAnimationPropertyData& data) override;
    void deleteKeyframe(float time) override;
    void changeKeyframeTime(float timeOld, float timeNew) override;

private:
    static constexpr uint32_t INVALID_JOINT_INDEX = std::numeric_limits<uint32_t>::max();

    uint32_t m_jointIndex = INVALID_JOINT_INDEX;
};


// ** NauAnimationPropertyList

struct NauAnimationPropertyList
{
public:
    using ClipList = std::vector<NauAnimationClipProperty>;
    using SkelList = std::vector<NauAnimationSkelProperty>;

    void setRefillFlag(bool flag) noexcept;
    void setReadOnly(bool flag) noexcept;
    void setTimeAnimation(float start, float end) noexcept;
    void setFrameDuration(float value) noexcept;

    [[nodiscard]]
    bool empty() const noexcept;
    [[nodiscard]]
    size_t size() const noexcept;
    [[nodiscard]]
    bool refilled() const noexcept;
    [[nodiscard]]
    float startAnimation() const noexcept { return m_startAnimation; }
    [[nodiscard]]
    float endAnimation() const noexcept { return m_endAnimation; }
    [[nodiscard]]
    float frameDuration() const noexcept { return m_frameDuration; }
    [[nodiscard]]
    NauAnimationProperty* propertyByIndex(int propertyIndex) noexcept;
    [[nodiscard]]
    ClipList& clipList() noexcept { return m_clipPropertyList; }
    [[nodiscard]]
    SkelList& skelList() noexcept { return m_skelPropertyList; }

    template <class Pred>
    void forEach(Pred pred)
    {
        std::for_each(m_clipPropertyList.begin(), m_clipPropertyList.end(), pred);
        std::for_each(m_skelPropertyList.begin(), m_skelPropertyList.end(), pred);
    }

private:
    ClipList m_clipPropertyList;
    SkelList m_skelPropertyList;
    float m_startAnimation = 0.f;
    float m_endAnimation = 0.f;
    float m_frameDuration = 1.f / 60.f;
    bool m_refilled = true;
};


using NauAnimationPropertyListPtr = std::shared_ptr<NauAnimationPropertyList>;
using NauAnimationNameList = std::vector<std::string>;
